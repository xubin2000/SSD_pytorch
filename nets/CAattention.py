import torch
import torch.nn as nn





class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16, out_channels=None):
        super(CA_Block, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        self.out_channels = out_channels

    def forward(self, x):
        # batch_size, c, h, w
        _, _, h, w = x.size()

        # batch_size, c, h, w => batch_size, c, h, 1 => batch_size, c, 1, h
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        # batch_size, c, h, w => batch_size, c, 1, w
        x_w = torch.mean(x, dim=2, keepdim=True)

        # batch_size, c, 1, w cat batch_size, c, 1, h => batch_size, c, 1, w + h
        # batch_size, c, 1, w + h => batch_size, c / r, 1, w + h
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        # batch_size, c / r, 1, w + h => batch_size, c / r, 1, h and batch_size, c / r, 1, w
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        # batch_size, c / r, 1, h => batch_size, c / r, h, 1 => batch_size, c, h, 1
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        # batch_size, c / r, 1, w => batch_size, c, 1, w
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
