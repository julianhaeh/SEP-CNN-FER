# Class for our own architecture and model

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)


class OurCNN(nn.Module):
    """
    Standard Baseline CNN für 64x64 Grayscale.
    - 4 Stages, jeweils Downsampling via MaxPool
    - 2 Conv-Blöcke pro Stage
    - GAP + MLP-Head
    """
    def __init__(
        self,
        num_classes: int,
        channels=(32, 64, 128, 256),
        dropout2d=0.0,
        dropout_fc=0.3,
        fc_dim=256,
    ):
        super().__init__()

        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            ConvBNReLU(1, c1, dropout=dropout2d),
            ConvBNReLU(c1, c1, dropout=dropout2d),
            nn.MaxPool2d(2),  # 64 -> 32
        )

        self.stage2 = nn.Sequential(
            ConvBNReLU(c1, c2, dropout=dropout2d),
            ConvBNReLU(c2, c2, dropout=dropout2d),
            nn.MaxPool2d(2),  # 32 -> 16
        )

        self.stage3 = nn.Sequential(
            ConvBNReLU(c2, c3, dropout=dropout2d),
            ConvBNReLU(c3, c3, dropout=dropout2d),
            nn.MaxPool2d(2),  # 16 -> 8
        )

        self.stage4 = nn.Sequential(
            ConvBNReLU(c3, c4, dropout=dropout2d),
            ConvBNReLU(c4, c4, dropout=dropout2d),
            nn.MaxPool2d(2),  # 8 -> 4
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B, c4, 1, 1)
        self.head = nn.Sequential(
            nn.Flatten(),  # (B, c4)
            nn.Linear(c4, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.head(x)
        return x
