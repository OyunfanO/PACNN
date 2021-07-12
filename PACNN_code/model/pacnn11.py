from model import common
import torch
import torch.nn as nn
import scipy.io as sio
import model.msrd as MSRDBlock

#msrd3, 128,128,128,128, d12v2
def make_model(args, parent=False):
    return PACNN11(args)

class PACNN11(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(PACNN11, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        self.scale_idx = 0
        nColor = args.n_colors

        act = nn.PReLU()

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        n = 1
        m_head = [common.BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(self.make_layer(MSRDBlock.MSRD3, n_feats))
        d_l0a = [common.BBlock(conv, n_feats, n_feats//2 , kernel_size, act=act, bn=False)]
        d_l0a.append(self.make_layer(MSRDBlock.CAttention, n_feats//2))


        d_l1 = [common.BBlock(conv, n_feats * 4, n_feats , kernel_size, act=act, bn=False)]
        #d_l1.append(common.DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False))
        d_l1.append(self.make_layer(MSRDBlock.MSRD3, n_feats))
        d_l1a = [common.BBlock(conv, n_feats, n_feats//2, kernel_size, act=act, bn=False)]
        d_l1a.append(self.make_layer(MSRDBlock.CAttention, n_feats//2))
        
        d_l121 = [common.BBlock(conv, n_feats, n_feats//2 , kernel_size, act=act, bn=False)]
        d_l121.append(self.make_layer(MSRDBlock.MSRD3, n_feats//2))
        d_l122 = [common.BBlock(conv, n_feats*3//2, n_feats//2 , kernel_size, act=act, bn=False)]
        d_l122.append(self.make_layer(MSRDBlock.MSRD3, n_feats//2))
        d_l123 = [common.BBlock(conv, n_feats*2, n_feats//2 , kernel_size, act=act, bn=False)]
        d_l123.append(self.make_layer(MSRDBlock.MSRD3, n_feats//2))
        d_l124 = [common.BBlock(conv, n_feats*5//2, n_feats//2 , kernel_size, act=act, bn=False)]
        d_l124.append(self.make_layer(MSRDBlock.MSRD3, n_feats//2))
        d_l125 = [common.BBlock(conv, n_feats*3, n_feats , kernel_size, act=act, bn=False)]


        d_l2 = []
        d_l2.append(common.BBlock(conv, n_feats * 4, n_feats, kernel_size, act=act, bn=False))
        #d_l2.append(common.DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False))
        d_l2.append(self.make_layer(MSRDBlock.MSRD3, n_feats))
        d_l2.append(common.BBlock(conv, n_feats, n_feats, kernel_size, act=act, bn=False))
        d_l2a = [common.BBlock(conv, n_feats, n_feats//2, kernel_size, act=act, bn=False)]
        d_l2a.append(self.make_layer(MSRDBlock.CAttention, n_feats//2))

        pro_l3 = []
        pro_l3.append(common.BBlock(conv, n_feats * 4, n_feats, kernel_size, act=act, bn=False))
        #pro_l3.append(common.DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        #pro_l3.append(common.DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(self.make_layer(MSRDBlock.MSRD3, n_feats))
        pro_l3.append(self.make_layer(MSRDBlock.MSRD3, n_feats))

        pro_l3.append(common.BBlock(conv, n_feats, n_feats * 4, kernel_size, act=act, bn=False))
        pro_l3a = [common.BBlock(conv, n_feats*4, n_feats//2, kernel_size, act=act, bn=False)]
        pro_l3a.append(self.make_layer(MSRDBlock.CAttention, n_feats//2))

        i_l2 = [common.BBlock(conv, n_feats * 2, n_feats, kernel_size, act=act, bn=False)]
        i_l2.append(self.make_layer(MSRDBlock.MSRD3, n_feats))
        i_l2.append(common.BBlock(conv, n_feats, n_feats * 4, kernel_size, act=act, bn=False))
        i_l2a = [common.BBlock(conv, n_feats*4, n_feats//2, kernel_size, act=act, bn=False)]
        i_l2a.append(self.make_layer(MSRDBlock.CAttention, n_feats//2))

        i_l1 = [common.BBlock(conv, n_feats * 3, n_feats, kernel_size, act=act, bn=False)]
        i_l1.append(self.make_layer(MSRDBlock.MSRD3, n_feats))
        i_l1.append(common.BBlock(conv, n_feats, n_feats * 4, kernel_size, act=act, bn=False))
        i_l1a = [common.BBlock(conv, n_feats*4, n_feats//2, kernel_size, act=act, bn=False)]
        i_l1a.append(self.make_layer(MSRDBlock.CAttention, n_feats//2))

        i_l0 = [common.BBlock(conv, n_feats * 2, n_feats, kernel_size, act=act, bn=False)]
        i_l0.append(self.make_layer(MSRDBlock.MSRD3, n_feats))
        i_l0a = [common.BBlock(conv, n_feats, n_feats//2, kernel_size, act=act, bn=False)]
        i_l0a.append(self.make_layer(MSRDBlock.CAttention, n_feats//2))

        m_tail = [conv(n_feats, nColor, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l121 = nn.Sequential(*d_l121)
        self.d_l122 = nn.Sequential(*d_l122)
        self.d_l123 = nn.Sequential(*d_l123)
        self.d_l124 = nn.Sequential(*d_l124)
        self.d_l125 = nn.Sequential(*d_l125)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        
        self.d_l2a = nn.Sequential(*d_l2a)
        self.d_l1a = nn.Sequential(*d_l1a)
        self.d_l0a = nn.Sequential(*d_l0a)
        self.pro_l3a = nn.Sequential(*pro_l3a)
        self.i_l2a = nn.Sequential(*i_l2a)
        self.i_l1a = nn.Sequential(*i_l1a)
        self.i_l0a = nn.Sequential(*i_l0a)
        
        self.tail = nn.Sequential(*m_tail)

    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        x0 = self.d_l0(self.head(x))
        x0a = torch.squeeze(self.d_l0a(x0), dim=1)
        #print(x0.shape, x0a.shape)
        x0 = torch.einsum('bchw,bhw->bchw', x0, x0a)
        
        x1 = self.d_l1(self.DWT(x0))
        x1a = torch.squeeze(self.d_l1a(x1), dim=1)
        x1 = torch.einsum('bchw,bhw->bchw', x1, x1a)
        
        x121 = self.d_l121(x1)
        x122 = self.d_l122(torch.cat((x1, x121),1))
        x123 = self.d_l123(torch.cat((x1, x121, x122),1))
        x124 = self.d_l124(torch.cat((x1, x121, x122, x123),1))
        x125 = self.d_l125(torch.cat((x1, x121, x122, x123, x124),1)) + x1
        
        x2 = self.d_l2(self.DWT(x1))
        x2a = torch.squeeze(self.d_l2a(x2), dim=1)
        x2 = torch.einsum('bchw,bhw->bchw', x2, x2a)
        
        pl3 = self.pro_l3(self.DWT(x2))
        pl3a = torch.squeeze(self.pro_l3a(pl3), dim=1)
        pl3 = torch.einsum('bchw,bhw->bchw', pl3, pl3a)
        x2_ = torch.cat((self.IWT(pl3),x2),1)
        
        i_l2 = self.i_l2(x2_)
        i_l2a = torch.squeeze(self.i_l2a(i_l2), dim=1)
        i_l2 = torch.einsum('bchw,bhw->bchw', i_l2, i_l2a)
        x1_ = torch.cat((self.IWT(i_l2), x1, x125),1)
        
        i_l1 = self.i_l1(x1_)
        i_l1a = torch.squeeze(self.i_l1a(i_l1), dim=1)
        i_l1 = torch.einsum('bchw,bhw->bchw', i_l1, i_l1a)       
        x0_ = torch.cat((self.IWT(i_l1),x0),1)
        
        i_l0 = self.i_l0(x0_)
        i_l0a = torch.squeeze(self.i_l0a(i_l0), dim=1)
        i_l0 = torch.einsum('bchw,bhw->bchw', i_l0, i_l0a)
        x = self.tail(i_l0) + x

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

