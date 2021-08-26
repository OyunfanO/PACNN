from model import common
import torch
import torch.nn as nn
import scipy.io as sio

def make_model(args, parent=False):
    return DDUNET(args)

class DDUNET(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DDUNET, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale_idx = 0
        nColor = args.n_colors
        self.n_unets = args.n_resblocks

        act = nn.ReLU(True)
        self.head = common.BBlock(conv, nColor, n_feats, kernel_size, act=act)
        
        
        
        self.UNets = nn.ModuleList()
        
        for k in range(self.n_unets):
            self.UNets.append(DenseUNet(conv, n_feats, kernel_size, n_th = k+1))
        
        fusion = []
        fusion.append(conv((self.n_unets + 1) * n_feats, n_feats, kernel_size))
        fusion.append(common.BBlock(conv, n_feats, n_feats, kernel_size, act=act))
        self.fusion = nn.Sequential(*fusion)
        self.tail = conv(n_feats, nColor, kernel_size)
        
    def forward(self, x):
        #print(x.size())
        x0 = self.head(x)
        
        x0_list = [x0]
        x0, x1, x2, x3 = self.UNets[0](x0)
        
        x0_list.append(x0)
        x1_list = [x1]
        x2_list = [x2]
        x3_list = [x3]
        for i in range(1, self.n_unets):
            x0 = torch.cat(x0_list, 1)
            x1 = torch.cat(x1_list, 1)
            x2 = torch.cat(x2_list, 1)
            x3 = torch.cat(x3_list, 1)
            
            x0, x1, x2, x3 = self.UNets[i](x0 ,x1, x2, x3)
            
            x0_list.append(x0)
            x1_list.append(x1)
            x2_list.append(x2)
            x3_list.append(x3)
        x0 = self.fusion(torch.cat(x0_list, 1))
        x = self.tail(x0) + x

        return x
    
class DenseUNet(nn.Module):
    def __init__(self, conv, channel_in, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), n_th=1):
        super(DenseUNet, self).__init__()
        
        self.DWT = common.DWT()
        self.IWT = common.IWT()

        
        self.d_l00 = common.BBlock(conv, channel_in * n_th, channel_in, kernel_size, act=act, bn=False)
        self.d_l01 = self.make_layer(DCR_block, channel_in, kernel_size)


        self.d_l10 = common.BBlock(conv, channel_in * (3 + n_th), channel_in, kernel_size, act=act, bn=False)
        self.d_l11 = self.make_layer(DCR_block, channel_in, kernel_size)

        self.d_l20 = common.BBlock(conv, channel_in * (3 + n_th), channel_in, kernel_size, act=act, bn=False)
        self.d_l21 = self.make_layer(DCR_block, channel_in, kernel_size)
        
        
        self.pro_l30 = common.BBlock(conv, channel_in * (3 + n_th), channel_in, kernel_size, act=act, bn=False)
        self.pro_l31 = self.make_layer(DCR_block, channel_in, kernel_size)
        self.pro_l32 = self.make_layer(DCR_block, channel_in, kernel_size)
        
        self.i_l2 = nn.Sequential(common.BBlock(conv, channel_in * 5 // 4, channel_in, kernel_size, act=act, bn=False),
                                   self.make_layer(DCR_block, channel_in, kernel_size)
                                   )
        
        self.i_l1 = nn.Sequential(common.BBlock(conv, channel_in * 5 // 4, channel_in, kernel_size, act=act, bn=False),
                                   self.make_layer(DCR_block, channel_in, kernel_size)
                                   )
        
        self.i_l0 = nn.Sequential(common.BBlock(conv, channel_in * 5 // 4, channel_in, kernel_size, act=act, bn=False),
                                   self.make_layer(DCR_block, channel_in, kernel_size)
                                   )
        
    def make_layer(self, block, channel_in, kernel_size,
                   bias=True, bn=False, act=nn.ReLU(True)):
        layers = []
        layers.append(block(channel_in, kernel_size, bias=True, bn=False, act=act))
        return nn.Sequential(*layers)


    def forward(self, x, c1=None, c2=None, c3=None):
        #print(x.size())
        d0 = self.d_l00(x)
        d0 = self.d_l01(d0)
        
        d1 = self.DWT(d0)
        if c1 is not None: d1 = torch.cat([d1, c1], dim = 1)
        d1 = self.d_l11(self.d_l10(d1))
        
        
        d2 = self.DWT(d1)
        if c2 is not None: d2 = torch.cat([d2, c2], dim = 1)
        d2 = self.d_l21(self.d_l20(d2))
        
        
        d3 = self.DWT(d2)
        if c3 is not None: d3 = torch.cat([d3, c3], dim = 1)
        d3 = self.pro_l32(self.pro_l31(self.pro_l30(d3)))
        
        i2 = torch.cat([self.IWT(d3), d2], 1)
        i2 = self.i_l2(i2)
        
        i1 = torch.cat([self.IWT(i2), d1], 1)
        i1 = self.i_l1(i1)
        
        i0 = torch.cat([self.IWT(i1), d0], 1)
        i0 = self.i_l0(i0)
        
        return  i0, i1, i2, d3
        
        
        
class DCR_block(nn.Module):
    def __init__(self, channel_in, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True)):
        super(DCR_block, self).__init__()        

        conv_1 = []
        conv_1.append(nn.Conv2d(in_channels=channel_in, out_channels=channel_in//2, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
        conv_1.append(act)
        conv_1.append(nn.Conv2d(in_channels=channel_in//2, out_channels=channel_in//2, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
        conv_1.append(act)
        
        conv_2 = []
        conv_2.append(nn.Conv2d(in_channels=3*channel_in//2, out_channels=channel_in//2, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
        conv_2.append(act)
        conv_2.append(nn.Conv2d(in_channels=channel_in//2, out_channels=channel_in//2, kernel_size=kernel_size, stride=1, padding=kernel_size//2))
        conv_2.append(act)
        
        
        fusion = []
        fusion.append(nn.Conv2d(in_channels = 2 * channel_in, out_channels = channel_in, kernel_size = kernel_size, stride = 1, padding = kernel_size//2))
        
        self.b1 = nn.Sequential(*conv_1)
        self.b2 = nn.Sequential(*conv_2)
        self.fusion = nn.Sequential(*fusion)
        
        
        

    def forward(self, x):
        x1 = self.b1(x)
        x2 = self.b2(torch.cat((x, x1), dim=1))
        out = self.fusion(torch.cat((x, x1, x2), dim=1)) + x
        
        return out
       
class DWTBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DWTBlock, self).__init__()
        m = []
        m.append(nn.Conv2d(in_channels, out_channels, kernel_size*2, stride=2, 
                           padding=(kernel_size*2//3), bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x   
    

    