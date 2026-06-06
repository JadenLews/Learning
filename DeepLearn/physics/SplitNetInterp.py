from unetFixed import DownSample, UpSample, DoubleConv
import torch
from torch import nn
from torch.nn import functional as F

class Encode(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_convolution_1 = DownSample(1, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

    def forward(self, x):
        # Start with convolution, expand 1 channels to 4.
        # Then downsample 5 times, saving the result
        x1, p1 = self.down_convolution_1(x)
        x2, p2 = self.down_convolution_2(p1)
        x3, p3 = self.down_convolution_3(p2)
        x4, p4 = self.down_convolution_4(p3)

        return x1,x2,x3,x4,p4

class Decode(nn.Module):

    def __init__(self):
        # Now back up
        super().__init__()
        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

    def forward(self, x1,x2,x3,x4,input):
        # At each layer with concatenate with the xi that is the same size as the up after upsampling.
        up_1 = self.up_convolution_1(input, x4)
        up_2 = self.up_convolution_2(up_1, x3)
        up_3 = self.up_convolution_3(up_2, x2)
        up_4 = self.up_convolution_4(up_3, x1)
        # One last convolution on the result to return to 3 channels from 8, leaving us with the proper 3x200x200
        return up_4
        
class SplitNetInterp(nn.Module):

    def __init__(self):

        super().__init__()
        self.K_encode = Encode()
        self.P_encode = Encode()

        self.inside = nn.Sequential(
            DoubleConv(1024,1024),
            DoubleConv(1024,1024)
        )

        self.K_decode = Decode()
        self.P_decode = Decode()

        self.K_end = nn.Conv2d(64,1,1,1,0)
        self.P_end = nn.Conv2d(64,1,1,1,0)

    def forward(self, input):
        input = F.interpolate(
            input, (256,256),
            mode='bilinear',
            align_corners=False
        )

        k = input[:,0:1]
        p = input[:,1:2]
        k1,k2,k3,k4,kb = self.K_encode(k)
        p1,p2,p3,p4,pb = self.P_encode(p)

        merge = torch.concat((kb,pb), dim=1)
        merge = self.inside(merge)

        kf = self.K_decode(k1,k2,k3,k4,merge)
        pf = self.P_decode(p1,p2,p3,p4,merge)

        kf = self.K_end(kf)
        pf = self.P_end(pf)

        out = torch.concat((kf,pf), dim=1)
        return F.interpolate(out, (200,200),
            mode='bilinear',
            align_corners=False
        )