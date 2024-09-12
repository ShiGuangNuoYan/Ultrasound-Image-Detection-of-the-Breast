import torch.nn as nn
import torch.nn.functional as F
import torch
from util import ResidualBlockClass

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        dim = opt.dim
        self.conv_1 = nn.Conv2d(in_channels=4, out_channels=1 * dim, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.batchnormlize = nn.BatchNorm2d(1 * dim)
        self.residual_block_1 = ResidualBlockClass('G.Res1', 1 * dim, 2 * dim, resample='down')
        # 128x128
        self.residual_block_2 = ResidualBlockClass('G.Res2', 2 * dim, 4 * dim, resample='down')
        # 64x64
        #         self.residual_block_2_1  = ResidualBlockClass('G.Res2_1', 4*dim, 4*dim, resample='down')
        # 64x64
        # self.residual_block_2_2  = ResidualBlockClass('G.Res2_2', 4*dim, 4*dim, resample=None)
        # 64x64
        self.residual_block_3 = ResidualBlockClass('G.Res3', 4 * dim, 4 * dim, resample='down')
        # 32x32
        self.residual_block_4 = ResidualBlockClass('G.Res4', 4 * dim, 6 * dim, resample='down')
        # 16x16
        self.residual_block_5 = ResidualBlockClass('G.Res5', 6 * dim, 6 * dim, resample=None)
        # 16x16
        self.residual_block_6 = ResidualBlockClass('G.Res6', 6 * dim, 6 * dim, resample=None)

    def forward(self, x):
        x = self.conv_1(x)
        x1 = self.residual_block_1(x)  # x1:2*dimx128x128
        x2 = self.residual_block_2(x1)  # x2:4*dimx64x64
        #         x = self.residual_block_2_1(x)
        # x = self.residual_block_2_2(x)
        x3 = self.residual_block_3(x2)  # x3:4*dimx32x32
        x = self.residual_block_4(x3)  # x4:6*dimx16x16
        x = self.residual_block_5(x)
        x = self.residual_block_6(x)
        x = F.tanh(x)
        return x, x1, x2, x3


class Discriminator(nn.Module):
    def __init__(self, opt):
        dim = opt.dim
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=6 * dim, out_channels=6 * dim, kernel_size=3, stride=1, padding=1)
        # 16x16
        self.conv_2 = nn.Conv2d(in_channels=6 * dim, out_channels=6 * dim, kernel_size=3, stride=1, padding=1)

        self.conv_3 = nn.Conv2d(in_channels=6 * dim, out_channels=4 * dim, kernel_size=3, stride=1, padding=1)

        self.bn_1 = nn.BatchNorm2d(6 * dim)

        self.conv_4 = nn.Conv2d(in_channels=4 * dim, out_channels=4 * dim, kernel_size=3, stride=2, padding=1)
        # 8x8
        self.conv_5 = nn.Conv2d(in_channels=4 * dim, out_channels=4 * dim, kernel_size=3, stride=1, padding=1)
        # 8x8
        self.conv_6 = nn.Conv2d(in_channels=4 * dim, out_channels=2 * dim, kernel_size=3, stride=2, padding=1)
        # 4x4
        self.bn_2 = nn.BatchNorm2d(2 * dim)

        self.conv_7 = nn.Conv2d(in_channels=2 * dim, out_channels=2 * dim, kernel_size=3, stride=1, padding=1)
        # 4x4
        self.conv_8 = nn.Conv2d(in_channels=2 * dim, out_channels=1 * dim, kernel_size=3, stride=1, padding=1)
        # 4x4
        # self.conv_9 = nn.Conv2d(in_channels=1*dim, out_channels=1, kernel_size=4, stride=1, padding=(0, 1), dilation=(1, 3))
        # 1x1

    def forward(self, x):
        x = F.leaky_relu(self.conv_1(x), negative_slope=0.02)
        x = F.leaky_relu(self.conv_2(x), negative_slope=0.02)
        x = F.leaky_relu(self.conv_3(x), negative_slope=0.02)
        #         x = F.leaky_relu(self.bn_1(x), negative_slope=0.02)
        x = F.leaky_relu(self.conv_4(x), negative_slope=0.02)
        x = F.leaky_relu(self.conv_5(x), negative_slope=0.02)
        x = F.leaky_relu(self.conv_6(x), negative_slope=0.02)
        #         x = F.leaky_relu(self.bn_2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv_7(x), negative_slope=0.02)
        x = F.leaky_relu(self.conv_8(x), negative_slope=0.02)
        # x = self.conv_9(x)
        x = torch.mean(x, dim=[1, 2, 3])
        x = F.sigmoid(x)

        return x.view(-1, 1).squeeze()
