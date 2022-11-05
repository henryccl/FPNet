import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


SRM_npy = np.load(os.path.join(os.path.dirname(__file__), 'SRM_Kernels.npy'))


class SRMConv2d(nn.Module):

    def __init__(self, stride=1, padding=0):
        super(SRMConv2d, self).__init__()
        self.in_channels = 1
        self.out_channels = 30
        self.kernel_size = (5, 5)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding
        self.dilation = (1, 1)
        self.transpose = False
        self.output_padding = (0,)
        self.groups = 1
        self.weight = Parameter(torch.Tensor(30, 1, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(30), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class ConvModule(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 1)
        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Conv_layer(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, s=1):
        super(Conv_layer, self).__init__()
        if k == 1:
            self.conv = nn.Conv2d(in_dim, out_dim, k, stride=s)
        elif k == 3:
            self.conv = nn.Conv2d(in_dim, out_dim, k, padding=1, stride=s)

        self.norm = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class FPB(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5):
        super(FPB, self).__init__()
        mid_channels = int(out_dim * expand_ratio)

        self.main_conv = Conv_layer(in_dim, mid_channels, 1)
        self.short_conv = Conv_layer(in_dim, mid_channels, 1)
        self.final_conv = Conv_layer(mid_channels*2, out_dim, 1)
        self.conv1 = Conv_layer(mid_channels, mid_channels, 1)
        self.conv2 = Conv_layer(mid_channels, mid_channels, 3)

    def forward(self, x):
        x_main = self.main_conv(x)
        x_short = self.short_conv(x)
        res = x_main
        x_main = self.conv1(x_main)
        x_main = self.conv2(x_main)
        x_main = x_main + res
        x = torch.cat([x_short, x_main], dim=1)
        x = self.final_conv(x)
        return x


class ADM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ADM, self).__init__()
        self.conv = ConvModule(in_channels * 4, out_channels)
        self.att_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.act = nn.Sigmoid()
        self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        x_att = x.mean((2, 3), keepdim=True)
        x_att = self.att_conv(x_att)
        x_att = self.act(x_att)
        x = x*x_att
        x = self.norm(x)

        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


class FPNet(nn.Module):
    def __init__(self):
        super(FPNet, self).__init__()
        self.srm = SRMConv2d(1, 0)
        self.bn1 = nn.BatchNorm2d(30)
        self.act = nn.ReLU(inplace=True)

        self.fpb1 = FPB(30, 30)
        self.fpb2 = FPB(30*2, 30)
        self.fpb3 = FPB(30, 30)
        self.fpb4 = FPB(30*2, 30)

        self.adm1 = ADM(30, 64)
        self.fpb5 = FPB(64, 64)
        self.adm2 = ADM(64, 128)
        self.fpb6 = FPB(128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = x.float()
        x = self.srm(x)
        x = self.bn1(x)
        x = self.act(x)

        res1 = x
        x = self.fpb1(x)
        x = torch.cat([x, res1], dim=1)
        x = self.fpb2(x)
        res2 = x
        x = self.fpb3(x)
        x = torch.cat([x, res2], dim=1)
        x = self.fpb4(x)

        x = self.adm1(x)
        x = self.fpb5(x)

        x = self.adm2(x)
        x = self.fpb6(x)

        out = self.avgpool(x)
        out_flatten = out.view(out.size(0), out.size(1))
        out_flatten = self.dropout(out_flatten)
        out = self.fc(out_flatten)
        return out


if __name__ == '__main__':
    from thop import profile

    x = torch.randn(1, 1, 256, 256)

    net = FPNet()

    flops, params = profile(net, (x,))
    print('flops: ', flops, 'params: ', params)