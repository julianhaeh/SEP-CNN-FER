# CNN model to compare our model with, src:
# https://github.com/Xiaoccer/MobileFaceNet_Pytorch/blob/master/core/model.py
#
# Corresponding paper of the architecture:
# MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices (Chen et al., 2018)
#
# ADAPTATION GOAL:
# - Original repo expects 3-channel input and (typically) 112×96 resolution.
# - This version is adapted to 1-channel (grayscale) 64×64 input.
#
# NOTE: Every change vs. the original code is explicitly marked with "CHANGED:" comments.

from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import math
from torch.nn import Parameter


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1,
                      groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]

Mobilenetv2_bottleneck_setting = [
    # t, c, n, s
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


class MobileFacenet(nn.Module):
    def __init__(self, bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFacenet, self).__init__()

        # CHANGED (1/3): input channels 3 -> 1 for grayscale.
        # Original: ConvBlock(3, 64, 3, 2, 1)
        
        self.conv1 = ConvBlock(1, 64, 3, 2, 1)

        # Unchanged: still operates on 64 channels produced by conv1

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0)

        # CHANGED (2/3): spatial kernel (7,6) -> (4,4) to match 64×64 input.
        #
        # Reason:
        # - With original input 112×96, the feature map before this layer is 7×6.
        # - With 64×64 input, the feature map before this layer becomes 4×4.
        #
        # Original: ConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)

        self.linear7 = ConvBlock(512, 512, (4, 4), 1, 0, dw=True, linear=True)

        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        x = x.view(x.size(0), -1)
        return x

""" ArcMarginProduct (ArcFace) is primarily designed for face identity recognition / verification, not for emotion classification. We dont need it.
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        # (Already adapted in your version) device-safe one_hot (works on CPU/CUDA)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
"""

if __name__ == "__main__":
    # CHANGED (3/3): test input tensor shape to match grayscale 64×64.
    # Original: (2, 3, 112, 96)
    input = Variable(torch.FloatTensor(2, 1, 64, 64))

    net = MobileFacenet()
    print(net)
    x = net(input)
    print(x.shape)  # expected: torch.Size([2, 128])
