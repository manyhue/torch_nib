import torch
import torch.nn as nn
from tnibs.utils import *
from tnibs.modules import ClassifierModule, Module


class MLPConfig(Config):
    n_blks: int  # num primary blocks
    hidden_size: int = 64
    n_classes: int = 1
    dropout: float = 0  # Dropout probability
    activation: ... = nn.GELU
    test_dropout: ... = None


class MLP(Module):
    def __init__(self, c: MLPConfig):
        super().__init__()

        self.save_config(c)
        self.input_layer = nn.Sequential(nn.LazyLinear(c.hidden_size), c.activation())
        self.hidden_layers = self._make_hidden_layers(
            c.n_blks, c.hidden_size, c.dropout, c.activation
        )
        self.output_layer = nn.Linear(c.hidden_size, c.n_classes)

    def _make_hidden_layers(self, n_blks, hidden_size, dropout, activation):
        layers = []
        for _ in range(n_blks):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                activation(),
                nn.Dropout(dropout),  # Add dropout after ReLU
            ]
        if self.test_dropout is not None:
            layers[2] = nn.Dropout(self.test_dropout)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class MLPClassifier(MLP, ClassifierModule):
    def __init__(self, c: MLPConfig):
        super().__init__(c)
