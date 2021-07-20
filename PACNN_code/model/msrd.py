from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F

class MSRD_Conv(nn.Module):
    def __init__(self, inChannels, outChannels, kSize=3, dSize = 1): #dsize refers to dilation size
        super(MSRD_Conv, self).__init__()
        Cin = inChannels
        Cout  = outChannels
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, Cout, kSize, padding= dSize*(kSize-1)//2, stride=1,dilation = dSize),
            nn.PReLU(init=0)
        ])

    def forward(self, x):
        out = self.conv(x)
        return out        
        
class Res_Conv(nn.Module):
    def __init__(self, inChannels, kSize=3, dSize = 1): #dsize refers to dilation size
        super(Res_Conv, self).__init__()
        C = inChannels
        self.conv = nn.Sequential(*[
            nn.Conv2d(C, C, kSize, padding= dSize*(kSize-1)//2, stride=1,dilation = dSize),
            nn.PReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return out+x

class Res_Conv1(nn.Module):
    def __init__(self, inChannels, kSize=3, dSize = 1): #dsize refers to dilation size
        super(Res_Conv1, self).__init__()
        C = inChannels
        self.conv = nn.Sequential(*[
            nn.Conv2d(C, C, kSize, padding= dSize*(kSize-1)//2, stride=1,dilation = dSize),
            nn.ReLU(),
            nn.Conv2d(C, C, kSize, padding= dSize*(kSize-1)//2, stride=1,dilation = dSize)
        ])
    def forward(self, x):
        x1 = self.conv(x) + x
        out = F.relu(x1)
        return out

        
class MSRD3(nn.Module):
    def __init__(self, inChannels,kSize=3,dilated = [1,2,3,4]):
        super(MSRD3, self).__init__()
        Cin = inChannels
        Cout  = inChannels
        
        #self.m00 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[0]) #x->m0
        #self.m01 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[1])
        
        self.m10 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[0])
        self.m11 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[1])
        self.m12 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[2])
        self.m13 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[3])       
        
        self.m20 = MSRD_Conv(Cin*2,Cout//2,kSize,dSize=dilated[0]) #concat(m1,x)->m2
        self.m21 = MSRD_Conv(Cin*2,Cout//2,kSize,dSize=dilated[1])
        
        self.s3 = MSRD_Conv(2*Cin,Cout,kSize,dSize=1)  #concat(m2,m1) -> s3
        

    def forward(self, x):
    
        xm10 = self.m10(x)
        xm11 = self.m11(x)
        xm12 = self.m12(x)
        xm13 = self.m13(x)
                
        xm20 = self.m20(torch.cat((xm10,xm11,x),1))
        xm21 = self.m21(torch.cat((xm12,xm13,x),1))
        
        xs3 = self.s3(torch.cat((xm20,xm21,x),1))
        
        return x+xs3
        
class MSRD7(nn.Module):
    def __init__(self, inChannels,kSize=3,dilated = [1,2,3,4]):
        super(MSRD7, self).__init__()
        Cin = inChannels
        Cout  = inChannels
        
        #self.m00 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[0]) #x->m0
        #self.m01 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[1])
        
        self.m10 = MSRD_Conv(Cin,Cout//4,kSize,dSize=dilated[0])
        self.m11 = MSRD_Conv(Cin,Cout//4,kSize,dSize=dilated[1])
        self.m12 = MSRD_Conv(Cin,Cout//4,kSize,dSize=dilated[2])
        self.m13 = MSRD_Conv(Cin,Cout//4,kSize,dSize=dilated[3])       
        
        self.m20 = MSRD_Conv(Cin*2,Cout//2,kSize,dSize=dilated[1]) #concat(m1,x)->m2
        self.m21 = MSRD_Conv(Cin*2,Cout//2,kSize,dSize=dilated[2])
        
        self.s3 = MSRD_Conv(3*Cin,Cout,kSize,dSize=1)  #concat(m2,m1) -> s3
        

    def forward(self, x):
    
        xm10 = self.m10(x)
        xm11 = self.m11(x)
        xm12 = self.m12(x)
        xm13 = self.m13(x)
                
        xm20 = self.m20(torch.cat((xm10,xm11,xm12,xm13,x),1))
        xm21 = self.m21(torch.cat((xm10,xm11,xm12,xm13,x),1))
        
        xs3 = self.s3(torch.cat((xm10,xm11,xm12,xm13,xm20,xm21,x),1))
        
        return x+xs3
        
class MN1(nn.Module):
    def __init__(self, inChannels,kSize=3,dilated = [1,2,3,4]):
        super(MN1, self).__init__()
        Cin = inChannels
        Cout  = inChannels
        
        #self.m00 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[0]) #x->m0
        #self.m01 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[1])
        
        self.s1 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[1])       
        
        self.s2 = MSRD_Conv(Cin*3//2,Cout//2,kSize,dSize=dilated[2]) #concat(m1,x)->m2
        
        self.s3 = MSRD_Conv(2*Cin,Cout,kSize,dSize=dilated[0])  #concat(m2,m1) -> s3
        

    def forward(self, x):
    
        xs1 = self.s1(x)
                
        xs2 = self.s2(torch.cat((xs1,x),1))
        
        xs3 = self.s3(torch.cat((xs1,xs2,x),1))
        
        return xs3+x
        
class CAttention(nn.Module):
    def __init__(self, inChannels,kSize=3):
        super(CAttention, self).__init__()
        Cin = inChannels
        Cout  = 1
        
        att = []
        att.append(self.make_layer(MN1, Cin))
        att.append(MSRD_Conv(Cin,Cout,kSize,dSize=1))
        att.append(nn.Sigmoid())
        self.att = nn.Sequential(*att)
        
    def make_layer(self, block, channel_in):
        layers = []
        layers.append(block(channel_in))
        return nn.Sequential(*layers)


    def forward(self, x):
        
        return self.att(x)
        
class DenBlock(nn.Module):
    def __init__(self, inChannels,kSize=3,dilated = 1):
        super(DenBlock, self).__init__()
        Cin = inChannels
        Cout  = inChannels
        
        #self.m00 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[0]) #x->m0
        #self.m01 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated[1])
        
        self.s1 = MSRD_Conv(Cin,Cout//2,kSize,dSize=dilated)       
        
        self.s2 = MSRD_Conv(Cin*3//2,Cout//2,kSize,dSize=dilated) #concat(m1,x)->m2
        
        self.s3 = MSRD_Conv(2*Cin,Cout,kSize,dSize=dilated)  #concat(m2,m1) -> s3
        

    def forward(self, x):
    
        xs1 = self.s1(x)
                
        xs2 = self.s2(torch.cat((xs1,x),1))
        
        xs3 = self.s3(torch.cat((xs1,xs2,x),1))
        
        return xs3+x

class CAConv(nn.Module):
    def __init__(self, inChannels, outChannels, kSize=3, dSize = 1): #dsize refers to dilation size
        super(CAConv, self).__init__()
        self.Cout = outChannels
        self.Cin = inChannels
        self.kSize = kSize
        self.conv0 = nn.Sequential(*[
            nn.Conv2d(self.Cin, self.Cout, kSize, padding=dSize * (kSize - 1) // 2, stride=2, dilation=dSize),
            nn.PReLU(init=0),
            DenBlock(self.Cout)
        ])

        self.bias0 = nn.Sequential(*[
            nn.Conv2d(self.Cin, self.Cout, kSize, padding=dSize * (kSize - 1) // 2, stride=2, dilation=dSize),
            nn.PReLU(init=0),
            DenBlock(inChannels)
        ])

        self.linear = nn.Linear(self.Cout, self.Cin*self.Cout) #64->128*64

        self.pool1 = torch.nn.AdaptiveAvgPool3d((self.Cout, kSize, kSize)) #(64, 3, 3)
        self.pool2 = torch.nn.AdaptiveAvgPool3d((self.Cout, 1, 1)) #64, 1, 1


    def forward(self, x):#x.shape (batch=8, channel=128, H=48, W=48) outchannel = 64

        k1 = self.conv0(x)
        k2 = torch.zeros([k1.shape[0], self.Cout, self.kSize, self.kSize]).float().cuda() #k2 (8, 64, 3, 3)
        for i in range(0,k1.shape[0]): #i=64
            k2[i,:,:,:] = self.pool1(k1[i:i+1,:,:,:])#(i, 64, 48, 48) to (i, 64, 3, 3)
        k3 = k2.permute(0,2,3,1) #(8, 64, 3, 3) to (8, 3, 3, 64)
        k4 = self.linear(k3) #(8, 3, 3, 128*64)
        k5 = k4.permute(0, 3, 1, 2) #(8, 128*64, 3, 3)
        k6 = torch.reshape(k5, (k1.shape[0], self.Cout, self.Cin, self.kSize, self.kSize)) #(8, 64, 128, 3, 3) kernel size

        b1 = self.bias0(x)
        b2 = torch.zeros([k1.shape[0], self.Cout, 1, 1]).float().cuda() #b2 (8, 64, 1, 1) bias size
        for i in range(0,k1.shape[0]): #i=64
            b2[i,:] = self.pool2(b1[i:i+1,:,:,:]) #(i, 64, 48, 48) to (i, 64, 1, 1)
        #b3 = torch.squeeze(b2)
        return k6, b2
        
class CAConv1(nn.Module):
    def __init__(self, inChannels, outChannels, kSize=3, dSize = 1): #dsize refers to dilation size
        super(CAConv1, self).__init__()
        self.Cout = outChannels
        self.Cin = inChannels
        self.kSize = kSize
        self.conv0 = nn.Sequential(*[
            nn.Conv2d(self.Cin, self.Cout, kSize, padding=dSize * (kSize - 1) // 2, stride=2, dilation=dSize),
            nn.PReLU(init=0),
            nn.Conv2d(self.Cout, self.Cout, kSize, padding=dSize * (kSize - 1) // 2, stride=2, dilation=dSize),
            nn.PReLU(init=0),
            nn.Conv2d(self.Cout, self.Cout, kSize, padding=dSize * (kSize - 1) // 2, stride=2, dilation=dSize),
            nn.PReLU(init=0)
        ])

        self.bias0 = nn.Sequential(*[
            nn.Conv2d(self.Cin, self.Cout, kSize, padding=dSize * (kSize - 1) // 2, stride=2, dilation=dSize),
            nn.PReLU(init=0),
            nn.Conv2d(self.Cout, self.Cout, kSize, padding=dSize * (kSize - 1) // 2, stride=2, dilation=dSize),
            nn.PReLU(init=0),
            nn.Conv2d(self.Cout, self.Cout, kSize, padding=dSize * (kSize - 1) // 2, stride=2, dilation=dSize),
            nn.PReLU(init=0)
        ])

        self.pool1 = torch.nn.AdaptiveAvgPool3d((self.Cout, kSize, kSize)) #(64, 3, 3)
        self.pool2 = torch.nn.AdaptiveAvgPool3d((self.Cout, 1, 1)) #64, 1, 1


    def forward(self, x):#x.shape (batch=8, channel=128, H=48, W=48) outchannel = 64

        k1 = self.conv0(x)
        k2 = torch.zeros([k1.shape[0], self.Cout, self.kSize, self.kSize]).float().cuda() #k2 (8, 64, 3, 3)
        for i in range(0,k1.shape[0]): #i=64
            k2[i,:,:,:] = self.pool1(k1[i:i+1,:,:,:])#(i, 64, 48, 48) to (i, 64, 3, 3)
        k3 = k2.repeat(1, self.Cin, 1, 1)# (8, 64, 3, 3) to (8, 128*64, 3, 3) 
        k6 = torch.reshape(k3, (k1.shape[0], self.Cout, self.Cin, self.kSize, self.kSize)) #(8, 64, 128, 3, 3) kernel size

        b1 = self.bias0(x)
        b2 = torch.zeros([k1.shape[0], self.Cout, 1, 1]).float().cuda() #b2 (8, 64, 1, 1) bias size
        for i in range(0,k1.shape[0]): #i=64
            b2[i,:] = self.pool2(b1[i:i+1,:,:,:]) #(i, 64, 48, 48) to (i, 64, 1, 1)
        #b3 = torch.squeeze(b2)
        return k6, b2
