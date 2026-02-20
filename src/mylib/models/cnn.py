import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, output_dim, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=2)
        x = x.permute(0, 2, 1)
        return x
