from unet_orig import TwoConv, Downsample, SmallUp, SmallDown, Upsample, ImgAttn
import torch
from torch import nn
from torch.nn import functional as F

class Encode(nn.Module):

    def __init__(self):
        super().__init__()
        # Input is Nx1x200x200
        self.c1 = TwoConv(1, 4)
        self.d1 = Downsample(4,8) # 8x100x100
        self.d2 = Downsample(8,16) # 16x50x50
        self.su = nn.Sequential(
            SmallUp(16),
            SmallUp(16),
            SmallUp(16)
        ) # 16x56x56
        self.d3 = Downsample(16,32) # 32x28x28
        self.d4 = Downsample(32,64) # 64x14x14
        self.d5 = Downsample(64, 128) # 128x7x7

    def forward(self, input):
        # Start with convolution, expand 1 channels to 4.
        # Then downsample 5 times, saving the result
        top = self.c1(input)
        x1 = self.d1(top)
        x2 = self.d2(x1)
        x3 = self.d3(self.su(x2)) # Here we upsample slightly so that we can downsample with less border artifacts
        x4 = self.d4(x3)
        x5 = self.d5(x4)

        return top,x1,x2,x3,x4,x5

class Decode(nn.Module):

    def __init__(self):
        # Now back up
        super().__init__()
        self.u1 = Upsample(128, 64) # 64x14x14
        self.u2 = Upsample(64, 32) # 32x28x28
        self.u3 = Upsample(32, 16, tweak=nn.Sequential(
            SmallDown(16),
            SmallDown(16),
            SmallDown(16)
        ))  # 16x50x50
        self.u4 = Upsample(16,8) # 8x100x100
        self.u5 = Upsample(8,4) # 4x200x200
        self.final = TwoConv(4, 1, end_tanh=True)

    def forward(self, top,x1,x2,x3,x4,input):
        # At each layer with concatenate with the xi that is the same size as the up after upsampling.
        up = self.u1(input, x4)
        up = self.u2(up, x3)
        up = self.u3(up, x2) # Again, a small downsample here to get back on the proper resolution
        up = self.u4(up, x1)
        up = self.u5(up, top)
        # One last convolution on the result to return to 3 channels from 8, leaving us with the proper 3x200x200
        return self.final(up)
    
class DecodeAttn(nn.Module):

    def __init__(self):
        super().__init__()
        # Now back up
        self.u1 = Upsample(128, 64) # 64x14x14
        self.u2 = Upsample(64, 32) # 32x28x28
        self.u3 = Upsample(32, 16, tweak=nn.Sequential(
            SmallDown(16),
            SmallDown(16),
            SmallDown(16)
        ))  # 16x50x50
        self.u4 = Upsample(16,8) # 8x100x100
        self.u5 = Upsample(8,4) # 4x200x200
        self.final = TwoConv(4, 1, end_tanh=True)

        self.atn1 = ImgAttn(64)
        self.atn2 = ImgAttn(32)
        self.atn3 = ImgAttn(16)

    def forward(self, top,x1,x2,x3,x4,input):
        # At each layer with concatenate with the xi that is the same size as the up after upsampling.
        up = self.u1(input, x4)
        up,_ = self.atn1(up)
        up = self.u2(up, x3)
        up,_ = self.atn2(up)
        up = self.u3(up, x2) # Again, a small downsample here to get back on the proper resolution
        up,_ = self.atn3(up)
        up = self.u4(up, x1)
        up = self.u5(up, top)
        # One last convolution on the result to return to 3 channels from 8, leaving us with the proper 3x200x200
        return self.final(up)
    
class SplitNet(nn.Module):

    def __init__(self, attn=False):
        super().__init__()
        self.K_encode = Encode()
        self.P_encode = Encode()

        self.inside = nn.Sequential(
            TwoConv(256,128),
            TwoConv(128,128)
        )

        if not attn:
            self.K_decode = Decode()
            self.P_decode = Decode()
        else:
            self.K_decode = DecodeAttn()
            self.P_decode = DecodeAttn()

    def forward(self, input):
        k = input[:,0:1]
        p = input[:,1:2]

        kt,k1,k2,k3,k4,k5 = self.K_encode(k)
        pt,p1,p2,p3,p4,p5 = self.P_encode(p)

        merge = torch.concat((k5,p5), dim=1)

        merge = self.inside(merge)

        kf = self.K_decode(kt,k1,k2,k3,k4,merge)
        pf = self.P_decode(pt,p1,p2,p3,p4,merge)

        return torch.concat((kf,pf), dim=1)