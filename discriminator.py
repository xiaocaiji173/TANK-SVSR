import torch
import torch.nn as nn
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights. 网络权重初始化

    Parameters:
        net (network)   -- network to be initialized  网络
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal 网络初始化类型
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal. 缩放因子

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies. 只适用于正态分布
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    return net

class D_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1, BN=True):
        super(D_block, self).__init__()
        if BN:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(ch_out),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x):
        out = self.conv(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = D_block(3, 64, 3, 1, 1, False)
        self.conv1_2 = D_block(3, 64, 3, 1, 1, False)
        self.conv2 = D_block(128, 128, 3, 2, 1, True)
        # self.conv2 = D_block(128, 128, 4, 4, 0, True)
        self.conv3 = D_block(128, 128, 3, 1, 1, True)
        self.conv4 = D_block(128, 256, 3, 2, 1, True)
        self.conv5 = D_block(256, 256, 3, 1, 1, True)
        self.conv6 = D_block(256, 512, 3, 2, 1, True)
        self.conv7 = D_block(512, 512, 3, 1, 1, True)
        self.conv8 = D_block(512, 512, 3, 2, 1, True)
        self.conv9 = D_block(512, 8, 3, 2, 1, False)
        self.fc = nn.Sequential(
            nn.Linear(8*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, x,label=0):
        out1_1 = self.conv1(x)        # 64 256 256
        out1_2 = self.conv1(label)  # 64 256 256
        out1 = torch.cat((out1_1, out1_2), dim=1)
        out2 = self.conv2(out1)     # 128 128 128
        out3 = self.conv3(out2)     # 128 128 128
        out4 = self.conv4(out3)     # 256 64 64
        out5 = self.conv5(out4)     # 256 64 64
        out6 = self.conv6(out5)     # 512 32 32
        out7 = self.conv7(out6)     # 512 32 32
        out8 = self.conv8(out7)     # 512 16 16
        out9 = self.conv9(out8)     # 8 8 8
        out9 = out9.view(out9.size(0), -1)  # 512
        out10 = self.fc(out9)
        return out1, out4, out6, out8, out10

    def get_perception(self, x):
        out1 = self.conv1(x)        # 64 256 256
        out2 = self.conv2(out1)     # 128 128 128
        out3 = self.conv3(out2)     # 128 128 128
        out4 = self.conv4(out3)     # 256 64 64
        out5 = self.conv5(out4)     # 256 64 64
        out6 = self.conv6(out5)     # 512 32 32
        out7 = self.conv7(out6)     # 512 32 32
        out8 = self.conv8(out7)     # 512 16 16
        out9 = self.conv9(out8)     # 8 8 8
        return out1, out4, out6, out8

def get_Perception(real_feature, fake_feature, lam_p):
    # return lam_p * torch.mean(torch.abs(real_feature-fake_feature))
    return lam_p * torch.mean((real_feature - fake_feature))