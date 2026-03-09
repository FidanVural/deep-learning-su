"""
visualize_model.py

Generates a computation graph of the MLP using torchviz.
Output saved to outputs/model_graph.pdf and outputs/model_graph.png.
"""

import os
import sys
import torch
from torchviz import make_dot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.mlp import MLP, MLPConfig


def main() -> None:
    os.makedirs("outputs/nn_model_vis", exist_ok=True)

    config = MLPConfig(
        input_size=784,
        hidden_sizes=[512, 256, 128],
        num_classes=10,
        activation="relu",
        dropout=0.3,
        use_bn=True,
    )

    model = MLP(config)
    model.eval()

    x = torch.zeros(1, 1, 28, 28)
    y = model(x)

    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.attr(rankdir="TB", size="20,30", dpi="300")

    dot.render("outputs/nn_model_vis/model_graph", format="png", cleanup=True)
    print("Saved to outputs/nn_model_vis/model_graph.png")

if __name__ == "__main__":
    main()
