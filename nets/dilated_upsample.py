import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义膨胀卷积和上采样操作
def dilated_conv_upsample(input, dilation_rate, upsample_size):
    input = input.cuda()
    conv = nn.Conv2d(input.shape[1], input.shape[1], kernel_size=3, stride=1, padding=dilation_rate, dilation=dilation_rate)
    conv = conv.cuda()
    output = F.relu(conv(input))
    upsampled_output = F.interpolate(output, size=upsample_size, mode='bilinear', align_corners=False)
    return upsampled_output

# 定义融合操作
class FusionLayer(nn.Module):
    def __init__(self, in_channels):
        super(FusionLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1)
        self.conv1 = self.conv1.cuda()
    def forward(self, x):
        out = F.relu(self.conv1(x))
        return out
class FusionLayer1(nn.Module):
    def __init__(self, in_channels):
        super(FusionLayer1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1024, kernel_size=1)
        self.conv1 = self.conv1.cuda()
    def forward(self, x):
        out = F.relu(self.conv1(x))
        return out

