import model.common as common
import torch
import torch.nn as nn
import scipy.io as sio


def make_model(args, parent=False):
    return RED30(args)

class RED30(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RED30, self).__init__()
        n_feats = args.n_feats
        kernel_size = 3
        nColor = args.n_colors
        self.layerNum = 7
        act = nn.ReLU(True)

        self.scale_idx = 0

        self.Rednet = nn.ModuleList()
        self.Rednet.append(Conv_block(n_feats,nColor,kernel_size))
        for i in range(1,self.layerNum):
            self.Rednet.append(Conv_block(n_feats,n_feats,kernel_size))
        self.Rednet.append(Mid_block(n_feats,n_feats,kernel_size))
        for i in range(1,self.layerNum):
            self.Rednet.append(Deconv_block(n_feats,n_feats,kernel_size))
        self.Rednet.append(Deconv_block(nColor,n_feats,kernel_size))

    def forward(self, x):
        x0 = self.Rednet[0](x)
        x1 = self.Rednet[1](x0)
        x2 = self.Rednet[2](x1)
        x3 = self.Rednet[3](x2)        
        x4 = self.Rednet[4](x3)
        x5 = self.Rednet[5](x4)        
        x6 = self.Rednet[6](x5)

        x7 = self.Rednet[7](x6)

        x8 = self.Rednet[8](x7+x6)
        x9 = self.Rednet[9](x8+x5)
        x10 = self.Rednet[10](x9+x4)
        x11 = self.Rednet[11](x10+x3)
        x12 = self.Rednet[12](x11+x2)
        x13 = self.Rednet[13](x12+x1)
        x14 = self.Rednet[14](x13+x0)

        return x14

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

class Conv_block(nn.Module):
    def __init__(self, channel_out, channel_in, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):
        super(Conv_block, self).__init__()

        conv_1 = []
        conv_1.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2))
        conv_1.append(act)
        conv_1.append(nn.Conv2d(in_channels=channel_out, out_channels=channel_out, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2))
        conv_1.append(act)

        self.b1 = nn.Sequential(*conv_1)

    def forward(self, x):
        x1 = self.b1(x)
        return x1


class Mid_block(nn.Module):
    def __init__(self, channel_out, channel_in, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):
        super(Mid_block, self).__init__()

        conv_1 = []
        conv_1.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, stride=1,
                                padding=kernel_size // 2))
        conv_1.append(act)
        conv_1.append(
            nn.ConvTranspose2d(in_channels=channel_out, out_channels=channel_out, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2))
        conv_1.append(act)

        self.b1 = nn.Sequential(*conv_1)

    def forward(self, x):
        x1 = self.b1(x)
        return x1


class Deconv_block(nn.Module):
    def __init__(self,channel_out, channel_in, kernel_size,
                 bias=True, bn=False, act=nn.ReLU(True)):
        super(Deconv_block, self).__init__()

        conv_1 = []
        conv_1.append(nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_in, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2))
        conv_1.append(act)
        conv_1.append(nn.ConvTranspose2d(in_channels=channel_in, out_channels=channel_out, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2))
        conv_1.append(act)

        self.b1 = nn.Sequential(*conv_1)

    def forward(self, x):
        x1 = self.b1(x)
        return x1
