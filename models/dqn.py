import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, input_size=11, output_size=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.net(x)
