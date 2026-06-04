"""MLP 模型。"""
import torch.nn as nn
from config import hp


class IKMLP(nn.Module):
    def __init__(self):
        super().__init__()
        d = hp["hidden_dims"]
        layers = []
        prev = hp["input_dim"]
        for h in d:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(hp["dropout"]))
            prev = h
        layers.append(nn.Linear(prev, hp["output_dim"]))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
