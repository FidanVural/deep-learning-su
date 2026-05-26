"""
part2_comm.py

Part 2: Can two machines design a communication protocol?

Interactive AWGN communication system with Transformer-based TX/RX.
- Message: m ∈ {1,...,8}^4 (4 symbols from alphabet of size 8)
- T=4 communication rounds
- Forward channel: AWGN with σ²=0.25
- Feedback: noiseless relay (received noisy symbols sent back)
- Power constraint: E||x(t)||² ≤ 1 per round
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================
ALPHABET_SIZE = 8       # |S_0| = 8
NUM_SYMBOLS = 4         # message length
NUM_ROUNDS = 4          # T = 4 communication rounds
NOISE_VAR = 0.25        # σ² = 0.25
D_MODEL = 64            # transformer model dimension
N_HEADS = 4             # number of attention heads
N_LAYERS = 2            # transformer layers
DIM_FF = 128            # feedforward dimension
CODED_DIM = 1           # each coded symbol is a scalar


# ============================================================
# Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# Transformer Block
# ============================================================
class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm (as specified in the PDF)."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual (post-norm as in PDF eqs 2-3)
        h = self.norm1(x + self.attn(x, x, x, need_weights=False)[0])
        h = self.norm2(h + self.ffn(h))
        return h


# ============================================================
# TX Encoder
# ============================================================
class TXEncoder(nn.Module):
    """
    Transmitter encoder using Transformer.

    At each round t:
    1. Preprocess: map (message_embedding, prev_transmissions, prev_feedback) → d_model
    2. Transformer: self-attention over 4 symbol positions
    3. Output MLP: map to coded symbol + power normalization
    """

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 dim_ff=DIM_FF, num_rounds=NUM_ROUNDS, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_rounds = num_rounds

        # Symbol embedding (8 possible symbols)
        self.symbol_embed = nn.Embedding(ALPHABET_SIZE, d_model // 2)

        # Input preprocessing MLP: maps raw concatenated features → d_model
        # Raw features per position: embedding(d_model//2) + prev_sent(T) + prev_feedback(T)
        raw_dim = d_model // 2 + num_rounds + num_rounds
        self.preprocess = nn.Sequential(
            nn.Linear(raw_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=NUM_SYMBOLS)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output MLP: d_model → coded symbol (scalar)
        self.output_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, CODED_DIM),
        )

    def forward(self, message: torch.Tensor, sent_history: torch.Tensor,
                feedback_history: torch.Tensor, round_idx: int) -> torch.Tensor:
        """
        Args:
            message: (batch, 4) integer symbols [0..7]
            sent_history: (batch, 4, T) previously sent coded symbols
            feedback_history: (batch, 4, T) received feedback
            round_idx: current round (0-indexed)

        Returns:
            coded_symbols: (batch, 4) power-normalized coded symbols
        """
        batch_size = message.size(0)

        # Embed original symbols: (batch, 4, d_model//2)
        sym_emb = self.symbol_embed(message)

        # Concatenate: [embedding, sent_history, feedback_history]
        raw_input = torch.cat([
            sym_emb,
            sent_history,
            feedback_history,
        ], dim=-1)  # (batch, 4, raw_dim)

        # Preprocess to d_model
        z = self.preprocess(raw_input)  # (batch, 4, d_model)

        # Add positional encoding
        z = self.pos_enc(z)

        # Transformer blocks
        for block in self.transformer_blocks:
            z = block(z)

        # Output MLP → coded symbols
        coded = self.output_mlp(z).squeeze(-1)  # (batch, 4)

        # Power normalization: E||x||² ≤ 1
        # Normalize so that average power per round ≤ 1
        # ||x||² / num_symbols ≤ 1 → ||x||² ≤ num_symbols
        power = (coded ** 2).sum(dim=-1, keepdim=True)  # (batch, 1)
        max_power = float(NUM_SYMBOLS)
        coded = coded * torch.sqrt(
            torch.clamp(torch.tensor(max_power, device=coded.device) / (power + 1e-8), max=1.0)
        )

        return coded


# ============================================================
# RX Decoder
# ============================================================
class RXDecoder(nn.Module):
    """
    Receiver decoder using Transformer.

    Runs only at the end (after T rounds), taking all received noisy symbols.
    Output: classification logits for each of 4 symbol positions.
    """

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                 dim_ff=DIM_FF, num_rounds=NUM_ROUNDS, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Input preprocessing: T received values per position → d_model
        self.preprocess = nn.Sequential(
            nn.Linear(num_rounds, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=NUM_SYMBOLS)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head: d_model → 8 classes per position
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, ALPHABET_SIZE),
        )

    def forward(self, received_all: torch.Tensor) -> torch.Tensor:
        """
        Args:
            received_all: (batch, 4, T) all received noisy symbols across T rounds

        Returns:
            logits: (batch, 4, 8) classification logits
        """
        # Preprocess
        z = self.preprocess(received_all)  # (batch, 4, d_model)

        # Positional encoding
        z = self.pos_enc(z)

        # Transformer
        for block in self.transformer_blocks:
            z = block(z)

        # Classify
        logits = self.classifier(z)  # (batch, 4, 8)
        return logits


# ============================================================
# Communication System
# ============================================================
class CommSystem(nn.Module):
    """
    End-to-end trainable communication system.

    TX (Transformer) → AWGN Channel → RX (Transformer)
    with noiseless feedback (relay mechanism).
    """

    def __init__(self, noise_var=NOISE_VAR, num_rounds=NUM_ROUNDS):
        super().__init__()
        self.noise_var = noise_var
        self.noise_std = math.sqrt(noise_var)
        self.num_rounds = num_rounds

        self.encoder = TXEncoder()
        self.decoder = RXDecoder()

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        """
        Full communication protocol.

        Args:
            message: (batch, 4) integer symbols [0..7]

        Returns:
            logits: (batch, 4, 8) classification logits for decoded message
        """
        batch_size = message.size(0)
        device = message.device

        # History buffers
        sent_history = torch.zeros(batch_size, NUM_SYMBOLS, self.num_rounds, device=device)
        feedback_history = torch.zeros(batch_size, NUM_SYMBOLS, self.num_rounds, device=device)
        received_all = torch.zeros(batch_size, NUM_SYMBOLS, self.num_rounds, device=device)

        for t in range(self.num_rounds):
            # TX encodes
            coded = self.encoder(message, sent_history, feedback_history, t)
            # (batch, 4)

            # AWGN channel
            noise = torch.randn_like(coded) * self.noise_std
            received = coded + noise  # (batch, 4)

            # Store
            received_all[:, :, t] = received
            sent_history[:, :, t] = coded

            # Noiseless feedback (relay: send received signal back)
            if t < self.num_rounds - 1:
                feedback_history[:, :, t] = received

        # RX decodes (only at the end)
        logits = self.decoder(received_all)
        return logits


# ============================================================
# Training
# ============================================================
def train_comm_system(
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_train_messages: int = 100000,
    num_test_messages: int = 10000,
    device: str = "cuda",
    report_dir: str = "hw4/reports",
):
    """Train the communication system end-to-end."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(report_dir, exist_ok=True)

    print("=" * 60)
    print("Part 2: Transformer Communication System")
    print("=" * 60)
    print(f"  Alphabet size: {ALPHABET_SIZE}")
    print(f"  Message length: {NUM_SYMBOLS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Noise variance: {NOISE_VAR}")
    print(f"  Device: {device}")
    print()

    model = CommSystem(noise_var=NOISE_VAR, num_rounds=NUM_ROUNDS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Generate random messages for training and testing
    # Messages are random integers in [0, 7] of length 4
    def generate_messages(n):
        return torch.randint(0, ALPHABET_SIZE, (n, NUM_SYMBOLS))

    history = {
        "train_loss": [], "train_ser": [],
        "val_loss": [], "val_ser": [], "val_bler": [],
    }

    # Fixed test set
    test_messages = generate_messages(num_test_messages).to(device)

    print(f"Training for {epochs} epochs...")
    best_val_ser = 1.0

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_messages = generate_messages(num_train_messages).to(device)

        epoch_loss = 0.0
        epoch_errors = 0
        epoch_total = 0
        num_batches = num_train_messages // batch_size

        for i in range(num_batches):
            batch = train_messages[i * batch_size: (i + 1) * batch_size]
            optimizer.zero_grad()

            logits = model(batch)  # (batch, 4, 8)

            # Cross-entropy loss (flatten)
            loss = F.cross_entropy(
                logits.reshape(-1, ALPHABET_SIZE),
                batch.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * batch.size(0)

            # Symbol error rate
            decoded = logits.argmax(dim=-1)
            epoch_errors += (decoded != batch).sum().item()
            epoch_total += batch.numel()

        train_loss = epoch_loss / num_train_messages
        train_ser = epoch_errors / epoch_total
        scheduler.step()

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            val_logits = []
            for i in range(0, num_test_messages, batch_size):
                batch = test_messages[i:i + batch_size]
                val_logits.append(model(batch))
            val_logits = torch.cat(val_logits, dim=0)

            val_loss = F.cross_entropy(
                val_logits.reshape(-1, ALPHABET_SIZE),
                test_messages.reshape(-1),
            ).item()

            val_decoded = val_logits.argmax(dim=-1)
            val_errors = (val_decoded != test_messages).sum().item()
            val_total = test_messages.numel()
            val_ser = val_errors / val_total

            # Block error rate (entire message wrong)
            block_errors = (val_decoded != test_messages).any(dim=1).sum().item()
            val_bler = block_errors / num_test_messages

        history["train_loss"].append(train_loss)
        history["train_ser"].append(train_ser)
        history["val_loss"].append(val_loss)
        history["val_ser"].append(val_ser)
        history["val_bler"].append(val_bler)

        if val_ser < best_val_ser:
            best_val_ser = val_ser
            torch.save(model.state_dict(), os.path.join(report_dir, "../checkpoints/comm_system.pt"))

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Train SER: {train_ser:.4f} | "
                  f"Val SER: {val_ser:.4f} | "
                  f"Val BLER: {val_bler:.4f}")

    print(f"\n  Best Val SER: {best_val_ser:.4f}")

    # --- Final evaluation ---
    model.load_state_dict(torch.load(
        os.path.join(report_dir, "../checkpoints/comm_system.pt"),
        weights_only=True, map_location=device
    ))
    model.eval()

    # Evaluate on larger test set
    final_test = generate_messages(50000).to(device)
    with torch.no_grad():
        final_logits = []
        for i in range(0, 50000, batch_size):
            batch = final_test[i:i + batch_size]
            final_logits.append(model(batch))
        final_logits = torch.cat(final_logits, dim=0)

        final_decoded = final_logits.argmax(dim=-1)
        final_ser = (final_decoded != final_test).float().mean().item()
        final_bler = (final_decoded != final_test).any(dim=1).float().mean().item()

    print(f"\n  Final Test (50k messages):")
    print(f"    Symbol Error Rate (SER): {final_ser:.4f}")
    print(f"    Block Error Rate (BLER): {final_bler:.4f}")
    print(f"    Accuracy per symbol: {1 - final_ser:.4f}")
    print(f"    Message accuracy: {1 - final_bler:.4f}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_ser"], label="Train SER")
    axes[1].plot(history["val_ser"], label="Val SER")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Symbol Error Rate")
    axes[1].set_title("Symbol Error Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["val_bler"], label="Val BLER", color="red")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Block Error Rate")
    axes[2].set_title("Block Error Rate (Full Message)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(report_dir, "part2_comm_training.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves saved: {plot_path}")

    # Save results
    results = {
        "final_ser": final_ser,
        "final_bler": final_bler,
        "final_accuracy_per_symbol": 1 - final_ser,
        "final_message_accuracy": 1 - final_bler,
        "best_val_ser": best_val_ser,
        "epochs_trained": epochs,
        "config": {
            "alphabet_size": ALPHABET_SIZE,
            "message_length": NUM_SYMBOLS,
            "num_rounds": NUM_ROUNDS,
            "noise_variance": NOISE_VAR,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
        },
    }
    results_path = os.path.join(report_dir, "part2_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Part 2: Communication System")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_comm_system(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
