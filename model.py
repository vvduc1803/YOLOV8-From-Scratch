# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    """Base block in CNN"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn_act=True):
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
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv = CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Conv_end = CNNBlock(int(0.5*(1+2)*out_channels), out_channels, kernel_size=1, stride=1, padding=0)
        self.BottleNeck = BottleNeckBlock(out_channels//2, **kwargs)

    def forward(self, x):
        x = self.Conv(x)
        x, x1 = torch.split(x, self.out_channels//2, dim=1)
        x2 = self.BottleNeck(x1)
        x = torch.cat([x, x1, x2], dim=1)
        x = self.Conv_end(x)
        return x

class C2F_2_Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Conv = CNNBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Conv_end = CNNBlock(int(0.5*(2+2)*out_channels), out_channels, kernel_size=1, stride=1, padding=0)
        self.BottleNeck = BottleNeckBlock(out_channels//2, **kwargs)

    def forward(self, x):
        x = self.Conv(x)
        x, x1 = torch.split(x, self.out_channels//2, dim=1)
        x2 = self.BottleNeck(x1)
        x3 = self.BottleNeck(x2)
        x = torch.cat([x, x1, x2, x3], dim=1)
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

class Classifier(nn.Module):
    def __init__(self, num_classes=500):
        super().__init__()
        self.Conv = nn.Sequential(CNNBlock(512, 1280, kernel_size=1, stride=1, padding=0))
        self.Flatten = nn.Flatten()
        self.Linear = nn.Sequential(nn.Linear(62720, num_classes))

    def forward(self, x):
        x = self.Conv(x)
        x = self.Flatten(x)
        x = self.Linear(x)
        return x

class Yolov8_cls(nn.Module):
    """Model architecture based page: https://blog.roboflow.com/whats-new-in-yolov8/
       and the ONNX file of yolov8_cls.onnx"""

    def __init__(self, in_channels, num_classes=500):
        super().__init__()
        self.Block1 = nn.Sequential(CNNBlock(in_channels, 32, 3, 2, 1),
                                    CNNBlock(32, 64, 3, 2, 1))

        self.Block2 = C2FBlock(64, 64)

        self.Block3 = nn.Sequential(CNNBlock(64, 128, 3, 2, 1),
                                    C2F_2_Block(128, 128))

        self.Block4 = nn.Sequential(CNNBlock(128, 256, 3, 2, 1),
                                    C2F_2_Block(256, 256))

        self.Block5 = nn.Sequential(CNNBlock(256, 512, 3, 2, 1),
                                    C2F_2_Block(512, 512))

        self.Block6 = Classifier(num_classes)

    def forward(self, x):
        x = self.Block1(x)
        x = self.Block2(x)
        x = self.Block3(x)
        x = self.Block4(x)
        x = self.Block5(x)
        x = self.Block6(x)
        return x

class AnchorFreeYOLOv5(nn.Module):
    def __init__(self, num_classes=80, num_anchor_boxes=3):
        super(AnchorFreeYOLOv5, self).__init__()

        # Replace anchor boxes with center points
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, num_anchor_boxes * (num_classes + 4), kernel_size=1, stride=1)
        )

        self.num_classes = num_classes
        self.num_anchor_boxes = num_anchor_boxes

    def forward(self, x):
        x = self.head(x)

        batch_size, _, height, width = x.shape

        # Reshape output to match expected format
        x = x.view(batch_size, self.num_anchor_boxes, self.num_classes + 4, height, width)

        # Split the predicted outputs into objectness, class probabilities, and bounding box coordinates
        objectness = torch.sigmoid(x[:, :, 0:1, :, :])
        class_probs = torch.sigmoid(x[:, :, 1:self.num_classes + 1, :, :])
        bbox_center = torch.exp(x[:, :, -4:, :, :])
        bbox_size = torch.exp(x[:, :, -3:-1, :, :])
        bbox_offset = x[:, :, -1:, :, :]

        # Calculate the actual bounding box coordinates using the center points and offsets
        strides = [8, 16, 32]
        grid_x, grid_y = torch.meshgrid(torch.arange(height), torch.arange(width))
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().to(x.device)

        bbox_xy = (bbox_center * strides + grid_xy + bbox_offset) * strides
        bbox_wh = bbox_size * strides

        # Combine the predicted outputs into a single tensor
        bbox = torch.cat([bbox_xy, bbox_wh], dim=-1)
        output = torch.cat([objectness, bbox, class_probs], dim=-1)

        return output.permute(0, 3, 4, 1, 2)

model = AnchorFreeYOLOv5()
input_data = torch.randn(1, 256, 32, 32)
output = model(input_data)
print(output)
