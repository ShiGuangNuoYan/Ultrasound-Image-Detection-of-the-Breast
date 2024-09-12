import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockClass(nn.Module):
    def __init__(self, name, input_dim, output_dim, resample=None, activate='relu'):
        super(ResidualBlockClass, self).__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        self.batchnormlize_1 = nn.BatchNorm2d(input_dim)
        self.activate = activate
        if resample == 'down':
            self.conv_0 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.conv_shortcut = nn.AvgPool2d(3, stride=2, padding=1)
            self.conv_1 = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=2, padding=1)
            self.batchnormlize_2 = nn.BatchNorm2d(input_dim)
        elif resample == 'up':
            self.conv_0 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.conv_shortcut = nn.Upsample(scale_factor=2)
            self.conv_1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.ConvTranspose2d(in_channels=output_dim, out_channels=output_dim, kernel_size=3, stride=2,
                                             padding=2,
                                             output_padding=1, dilation=2)
            self.batchnormlize_2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            self.conv_shortcut = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1,
                                           padding=1)
            self.conv_1 = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, stride=1, padding=1)
            self.conv_2 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1)
            self.batchnormlize_2 = nn.BatchNorm2d(input_dim)
        else:
            raise Exception('invalid resample value')

    def forward(self, inputs):
        if self.output_dim == self.input_dim and self.resample == None:
            shortcut = inputs
        elif self.resample == 'down':
            x = self.conv_0(inputs)
            shortcut = self.conv_shortcut(x)
        elif self.resample == None:
            x = inputs
            shortcut = self.conv_shortcut(x)
        else:
            x = self.conv_0(inputs)
            shortcut = self.conv_shortcut(x)
        if self.activate == 'relu':
            x = inputs
            x = self.batchnormlize_1(x)
            x = F.relu(x)
            x = self.conv_1(x)
            x = self.batchnormlize_2(x)
            x = F.relu(x)
            x = self.conv_2(x)
            return shortcut + x
        else:
            x = inputs
            x = self.batchnormlize_1(x)
            x = F.leaky_relu(x)
            x = self.conv_1(x)
            x = self.batchnormlize_2(x)
            x = F.leaky_relu(x)
            x = self.conv_2(x)
            return shortcut + x