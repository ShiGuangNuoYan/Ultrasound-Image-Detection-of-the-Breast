import torch.nn as nn
import torch.nn.functional as F
import torch
from util import ResidualBlockClass
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim,opt,activation=None):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // opt.sa_scale, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // opt.sa_scale, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X (W*H) X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class UpProject(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpProject, self).__init__()
        # self.batch_size = batch_size

        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv1_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, (2, 3))
        self.conv2_3 = nn.Conv2d(in_channels, out_channels, (3, 2))
        self.conv2_4 = nn.Conv2d(in_channels, out_channels, 2)

        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # b, 10, 8, 1024
        batch_size = x.shape[0]
        out1_1 = self.conv1_1(nn.functional.pad(x, (1, 1, 1, 1)))
        out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 0, 1)))  # right interleaving padding
        # out1_2 = self.conv1_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github
        out1_3 = self.conv1_3(nn.functional.pad(x, (0, 1, 1, 1)))  # right interleaving padding
        # out1_3 = self.conv1_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github
        out1_4 = self.conv1_4(nn.functional.pad(x, (0, 1, 0, 1)))  # right interleaving padding
        # out1_4 = self.conv1_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        out2_1 = self.conv2_1(nn.functional.pad(x, (1, 1, 1, 1)))
        out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 0, 1)))  # right interleaving padding
        # out2_2 = self.conv2_2(nn.functional.pad(x, (1, 1, 1, 0)))#author's interleaving pading in github
        out2_3 = self.conv2_3(nn.functional.pad(x, (0, 1, 1, 1)))  # right interleaving padding
        # out2_3 = self.conv2_3(nn.functional.pad(x, (1, 0, 1, 1)))#author's interleaving pading in github
        out2_4 = self.conv2_4(nn.functional.pad(x, (0, 1, 0, 1)))  # right interleaving padding
        # out2_4 = self.conv2_4(nn.functional.pad(x, (1, 0, 1, 0)))#author's interleaving pading in github

        height = out1_1.size()[2]
        width = out1_1.size()[3]

        out1_1_2 = torch.stack((out1_1, out1_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out1_3_4 = torch.stack((out1_3, out1_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out1_1234 = torch.stack((out1_1_2, out1_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out2_1_2 = torch.stack((out2_1, out2_2), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)
        out2_3_4 = torch.stack((out2_3, out2_4), dim=-3).permute(0, 1, 3, 4, 2).contiguous().view(
            batch_size, -1, height, width * 2)

        out2_1234 = torch.stack((out2_1_2, out2_3_4), dim=-3).permute(0, 1, 3, 2, 4).contiguous().view(
            batch_size, -1, height * 2, width * 2)

        out1 = self.bn1_1(out1_1234)
        out1 = self.relu(out1)
        out1 = self.conv3(out1)
        out1 = self.bn2(out1)

        out2 = self.bn1_2(out2_1234)

        out = out1 + out2
        out = self.relu(out)

        return out


class Fcrn_encode(nn.Module):
    def __init__(self, opt):
        dim = opt.dim
        super(Fcrn_encode, self).__init__()
        self.alpha = opt.alpha
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, stride=1, padding=1)
        self.residual_block_1_down_1 = ResidualBlockClass('Detector.Res1', 1 * dim, 2 * dim, resample='down',
                                                          activate='leaky_relu')

        # 128x128
        self.residual_block_2_down_1 = ResidualBlockClass('Detector.Res2', 2 * dim, 4 * dim, resample='down',
                                                          activate='leaky_relu')
        # 64x64
        self.residual_block_3_down_1 = ResidualBlockClass('Detector.Res3', 4 * dim, 4 * dim, resample='down',
                                                          activate='leaky_relu')
        # 32x32
        self.residual_block_4_down_1 = ResidualBlockClass('Detector.Res4', 4 * dim, 6 * dim, resample='down',
                                                          activate='leaky_relu')
        # 16x16
        self.residual_block_5_none_1 = ResidualBlockClass('Detector.Res5', 6 * dim, 6 * dim, resample=None,
                                                          activate='leaky_relu')

    def forward(self, x, n1=0, n2=0, n3=0):
        x1 = self.conv_1(x)  # x1:dimx256x256
        x2 = self.residual_block_1_down_1(x1)  # x2:2dimx128x128
        x3 = self.residual_block_2_down_1((1 - self.alpha) * x2 + self.alpha * n1)  # x3:4dimx64x64
        x4 = self.residual_block_3_down_1((1 - self.alpha) * x3 + self.alpha * n2)  # x4:4dimx32x32
        x = self.residual_block_4_down_1((1 - self.alpha) * x4 + self.alpha * n3)
        feature = self.residual_block_5_none_1(x)
        x = F.tanh(feature)
        return x, x2, x3, x4
    
    
class Fcrn_decode(nn.Module):
    def __init__(self, opt):
        dim = opt.dim
        super(Fcrn_decode, self).__init__()

        self.conv_2 = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.residual_block_6_none_1 = ResidualBlockClass('Detector.Res6', 6 * dim, 6 * dim, resample=None,
                                                          activate='leaky_relu')
        #         self.residual_block_7_up_1       = ResidualBlockClass('Detector.Res7', 6*dim, 6*dim, resample='up', activate='leaky_relu')
        self.sa_0 = Self_Attn(6 * dim,opt)
        # 32x32
        self.UpProject_1 = UpProject(6 * dim, 4 * dim)
        self.residual_block_8_up_1 = ResidualBlockClass('Detector.Res8', 6 * dim, 4 * dim, resample='up',
                                                        activate='leaky_relu')
        self.sa_1 = Self_Attn(4 * dim,opt)
        # 64x64
        self.UpProject_2 = UpProject(2 * 4 * dim, 4 * dim)
        self.sa_2 = Self_Attn(4 * dim,opt)
        self.residual_block_9_up_1 = ResidualBlockClass('Detector.Res9', 4 * dim, 4 * dim, resample='up',
                                                        activate='leaky_relu')
        # 128x128
        self.UpProject_3 = UpProject(2 * 4 * dim, 2 * dim)
        self.sa_3 = Self_Attn(2 * dim,opt)
        self.residual_block_10_up_1 = ResidualBlockClass('Detector.Res10', 4 * dim, 2 * dim, resample='up',
                                                         activate='leaky_relu')
        # 256x256
        self.UpProject_4 = UpProject(2 * 2 * dim, 1 * dim)
        self.sa_4 = Self_Attn(1 * dim,opt)
        self.residual_block_11_up_1 = ResidualBlockClass('Detector.Res11', 2 * dim, 1 * dim, resample='up',
                                                         activate='leaky_relu')

    def forward(self, x, x2, x3, x4):
        x = self.residual_block_6_none_1(x)
        x = self.UpProject_1(x)
        x = self.sa_1(x)
        x = self.UpProject_2(torch.cat((x, x4), dim=1))
        x = self.sa_2(x)
        x = self.UpProject_3(torch.cat((x, x3), dim=1))
        #         x = self.sa_3(x)
        x = self.UpProject_4(torch.cat((x, x2), dim=1))
        #         x = self.sa_4(x)
        x = F.normalize(x, dim=[0, 2, 3])
        x = F.leaky_relu(x)
        x = self.conv_2(x)
        x = F.sigmoid(x)

        return x