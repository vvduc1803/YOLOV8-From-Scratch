# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
import torch
import torch.nn as nn
from torchinfo import summary


class CNNBlock(nn.Module):
    """Base block in CNN"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bn_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not bn_act)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            x = self.silu(self.bn(self.conv(x)))
            return x

        else:
            return self.conv(x)

class BottleNeckBlock(nn.Module):
    def __init__(self, channels, short_cut=True):
        super().__init__()
        self.short_cut = short_cut
        self.Conv = nn.Sequential(CNNBlock(channels, channels//2, 3, 1, 1),
                                  CNNBlock(channels//2, channels, 3, 1, 1))

    def forward(self, x):
        if self.short_cut:
            return self.Conv(x) + x
        else:
            return self.Conv(x)

class C2FBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=3, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv = CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Conv_end = CNNBlock(int(0.5*(repeat+2)*out_channels), out_channels, kernel_size=1, stride=1, padding=0)
        self.BottleNeck = BottleNeckBlock(out_channels//2, **kwargs)
        self.repeat = repeat

    def forward(self, x):
        x = self.Conv(x)
        x, x1 = torch.split(x, self.out_channels//2, dim=1)
        x = torch.cat([x, x1], dim=1)
        for i in range(self.repeat):
            x2 = self.BottleNeck(x1)
            x = torch.cat([x, x2], dim=1)
            x1 = x2
        x = self.Conv_end(x)
        return x

class SPPFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.Conv = CNNBlock(channels, channels, kernel_size=1, stride=1, padding=0)
        self.Conv_end = CNNBlock(4*channels, channels, kernel_size=1, stride=1, padding=0)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.Conv(x)
        x = torch.cat([x, self.MaxPool(x), self.MaxPool(self.MaxPool(x)), self.MaxPool(self.MaxPool(self.MaxPool(x)))],
                      dim=1)
        x = self.Conv_end(x)
        return x

class DetectBlock(nn.Module):
    def __init__(self, in_chan, num_classes=80):
        super().__init__()

        self.detect = nn.Sequential(CNNBlock(in_chan, in_chan, kernel_size=3, stride=1, padding=1),
                                    CNNBlock(in_chan, in_chan, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(in_chan, num_classes + 4, kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        x = self.detect(x)
        return x

class MyModel(nn.Module):
    def __init__(self, in_chan, hidden=64, num_classes=80):
        super().__init__()
        self.num_classes = num_classes

        self.Block1 = nn.Sequential(CNNBlock(in_chan, hidden),
                                    CNNBlock(hidden, hidden*2),
                                    C2FBlock(hidden*2, hidden*2),
                                    CNNBlock(hidden*2, hidden*4),
                                    C2FBlock(hidden*4, hidden*4, repeat=6))

        self.Block2 = nn.Sequential(CNNBlock(hidden*4, hidden*8),
                                    C2FBlock(hidden*8, hidden*8, repeat=6))

        self.Block3 = nn.Sequential(CNNBlock(hidden*8, hidden*8),
                                    C2FBlock(hidden*8, hidden*8),
                                    SPPFBlock(hidden*8))

        self.Block4 = nn.Upsample(scale_factor=2)

        self.Block5 = C2FBlock(hidden*16, hidden*8, repeat=3, short_cut=False)

        self.Block6 = nn.Upsample(scale_factor=2)

        self.Block7 = C2FBlock(hidden*12, hidden*4, short_cut=False)

        self.Block8 = CNNBlock(hidden*4, hidden*4)

        self.Block9 = C2FBlock(hidden*12, hidden*8, short_cut=False)

        self.Block10 = CNNBlock(hidden*8, hidden*8)

        self.Block11 = C2FBlock(hidden*16, hidden*8, short_cut=False)

        self.Detect_L = DetectBlock(hidden*4)

        self.Detect_M = DetectBlock(hidden*8)

        self.Detect_S = DetectBlock(hidden*8)

    def forward(self, x):

        x = self.Block1(x)

        x1 = self.Block2(x)

        x2 = self.Block3(x1)

        x3 = self.Block4(x2)

        x1 = torch.cat([x1, x3], dim=1)

        x1 = self.Block5(x1)

        x3 = self.Block6(x1)

        x = torch.cat([x, x3], dim=1)

        x = self.Block7(x)

        x3 = self.Block8(x)

        x1 = torch.cat([x1, x3], dim=1)

        x1 = self.Block9(x1)

        x3 = self.Block10(x1)

        x2 = torch.cat([x2, x3], dim=1)

        x2 = self.Block11(x2)

        outL = self.Detect_L(x)
        outM = self.Detect_M(x1)
        outS = self.Detect_S(x2)

        return [outL, outM, outS]

def test():
    x = torch.rand((1, 3, 640, 640))
    model = MyModel(3, )
    y = model(x)
    print(y[0].shape)
    # summary(model, (1, 3, 640, 640))

if __name__ == '__main__':
    test()

