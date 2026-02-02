# Class for our own architecture and model

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dropout=0.0, max_groups=8, residual=True, second_conv=True, use_se=True):
        super().__init__()

        def gn(ch):
            g = min(max_groups, ch)
            while ch % g != 0:
                g -= 1
            return nn.GroupNorm(g, ch)

        self.residual = residual and second_conv #only useful with at least 2 conv-layers

        # Main-Layer
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            gn(out_ch),
            nn.SiLU(inplace=True),
        ]

        # optional sec-conv-layer
        if second_conv:
            layers += [
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                gn(out_ch),
                nn.SiLU(inplace=True),
                SEBlock(out_ch) if use_se else nn.Identity(),
            ]

        self.net = nn.Sequential(*layers)

        if self.residual:
            if (in_ch != out_ch) or (s != 1):
                self.skip = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=s, padding=0, bias=False),
                    gn(out_ch),
                )
            else:
                self.skip = nn.Identity()
        else:
            self.skip = None

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            return out + self.skip(x)
        return out


class SEBlock(nn.Module): # Squeeze-and-Excitation; lets the network itself decide which feature maps are more relevant than others
    """Squeeze-and-Excitation: oft guter Accuracy-Boost bei wenig Overhead."""
    def __init__(self, ch, r=8):
        super().__init__()
        hidden = max(ch // r, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, ch),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.net(x).view(x.size(0), x.size(1), 1, 1)
        return x * w
    
class OurCNN(nn.Module):

    def __init__(
        self,
        num_classes: int = 6,
        channels=(32, 64, 128, 256),
        dropout2d=0.05,
        dropout_fc=0.3,
        fc_dim=256,
    ):
        super().__init__()

        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            ConvBNReLU(1, c1, s=2, dropout=dropout2d, residual=True, second_conv=True, use_se=False),
            ConvBNReLU(c1, c1, dropout=dropout2d, residual=True, second_conv=True, use_se=False),
            ConvBNReLU(c1, c1, dropout=dropout2d, residual=True, second_conv=True, use_se=False),
        )

        self.stage2 = nn.Sequential(
            ConvBNReLU(c1, c2, s=2,dropout=dropout2d, residual=True, second_conv=True, use_se=False),
            ConvBNReLU(c2, c2, dropout=dropout2d, residual=True, second_conv=True, use_se=False),
            ConvBNReLU(c2, c2, dropout=dropout2d, residual=True, second_conv=True, use_se=True),
        )

        self.stage3 = nn.Sequential(
            ConvBNReLU(c2, c3, s=2,dropout=dropout2d, residual=True, second_conv=True, use_se=False),
            ConvBNReLU(c3, c3, dropout=dropout2d, residual=True, second_conv=True, use_se=True),
            ConvBNReLU(c3, c3, dropout=dropout2d, residual=True, second_conv=True, use_se=False),
        )

        self.stage4 = nn.Sequential(
            ConvBNReLU(c3, c4, s=2,dropout=dropout2d, residual=True, second_conv=True, use_se=True),
            ConvBNReLU(c4, c4, dropout=dropout2d, residual=True, second_conv=True, use_se=False),
            ConvBNReLU(c4, c4, dropout=dropout2d, residual=True, second_conv=True, use_se=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B, c4, 1, 1)

        self.head = nn.Sequential(
            nn.Flatten(),  # (B, c4)
            nn.Linear(c4, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.SiLU(inplace=True),
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
