"""
> Maintainer: https://github.com/nsccsuperli/DPGAN

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from network import Resnet
from my_functionals import GatedSpatialConv as gsc
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        if bn:
            self.model = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_size, momentum=0.8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_size, in_size, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_size, momentum=0.8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_size, momentum=0.8),
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 2, 1, 1, bias=False ),
                nn.BatchNorm2d(out_size, momentum=0.8)
            )
            self.l1 = nn.Sequential(
                nn.LeakyReLU(0.2)
            )

        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_size, in_size, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 2, 1, 1, bias=False),
            )
            self.l1 = nn.Sequential(
                nn.LeakyReLU(0.2)
            )
    def forward(self, x):
        # setp 1
        out = self.model(x)
        # setp 2
        out2 =self.shortcut(x)
        # feature map add operation
        out.add(out2)
        # relu operation
        out = self.l1(out)

        return out

class UNetDown_1(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown_1, self).__init__()
        if bn:
            self.model = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_size, momentum=0.8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_size, in_size, 3, 1, 1, bias=False),
                nn.BatchNorm2d(in_size, momentum=0.8),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_size, momentum=0.8),
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1, 1, bias=False ),
                nn.BatchNorm2d(out_size, momentum=0.8)
            )
            self.l1 = nn.Sequential(
                nn.LeakyReLU(0.2)
            )

        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_size, in_size, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
            )

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1, 1, bias=False),
            )
            self.l1 = nn.Sequential(
                nn.LeakyReLU(0.2)
            )
    def forward(self, x):
        # setp 1
        out = self.model(x)
        # setp 2
        out2 =self.shortcut(x)
        # feature map add operation
        out.add(out2)
        # relu operation
        out = self.l1(out)

        return out

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1, 1, bias=False),

        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        x = self.shortcut(x)
        return x
class UNetUp_1(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp_1, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        # self.model = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1, 1, bias=False),

        )
        self.model = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_size, in_size, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
        )

    def forward(self, x, skip_input):
        x= self.model(x)
        x = torch.cat((x, skip_input), 1)
        x = self.shortcut(x)
        return x

class GeneratorDPGAN(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self, in_channels=1, out_channels=3):
        super(GeneratorDPGAN, self).__init__()
        self.res1 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.resnet = UNetDown_1(128, 256)
        self.resnet2 = UNetDown_1(256, 512)
        self.resnet3 = UNetDown_1(512, 1024)
        # self.resnet = Resnet.BasicBlock(128, 256, stride=1, downsample=None)
        # self.resnet2 = Resnet.BasicBlock(256, 512, stride=1, downsample=None)
        # self.resnet3 = Resnet.BasicBlock(512, 1024, stride=1, downsample=None)
        self.d1 = nn.Conv2d(32, 32, 1)
        self.d2 = nn.Conv2d(32, 16, 1)
        self.d3 = nn.Conv2d(16, 8, 1)
        self.dsn1 = nn.Conv2d(125, 1, 1)
        self.dsn3 = nn.Conv2d(128, 1, 1)
        self.dsn4 = nn.Conv2d(256, 1, 1)
        self.dsn7 = nn.Conv2d(512, 1, 1)
        self.gate1 = gsc.GatedSpatialConv2d(32, 32)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 256)
        self.down5 = UNetDown(256, 512, bn=False)

        # decoding layers

        self.up_r31 = UNetUp(128, 64)
        self.up_r32 = UNetUp(64, 32)


        self.up_r1 = UNetUp_1(1024, 512)
        self.up_r2 = UNetUp_1(512, 256)
        self.up_r3 = UNetUp_1(256, 128)
        self.up1 = UNetUp(512, 256)
        self.up2 = UNetUp(256, 128)
        self.up3 = UNetUp(128, 64)
        self.up4 = UNetUp(64, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, out_channels, 4, padding=1),
            nn.Tanh()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,edge):
        x_size = x.size()
        # downsample
        d1 = self.down1(x)# 4,32,64,48
        d2 = self.down2(d1)# 4,64,24,32
        d3 = self.down3(d2)# 128,12,16
        ru3_1 = self.up_r31(d3, d2) #128+64 ----64
        ru3_2 = self.up_r32(ru3_1, d1) #64+32---32
        seg_out2 = self.final(ru3_2)

        rd4 = self.resnet(d3) # 256,12,16
        rd5 = self.resnet2(rd4) #512,12,16
        rd6 = self.resnet3(rd5)# 1024
        # upsampling-3


        ru1 = self.up_r1(rd6, rd5)  # 1024+512 --->512
        ru2 = self.up_r2(ru1, rd4)  # 512+256 --->256
        ru3 = self.up_r3(ru2, d3)  # 256+128 --->128
        ru4 = self.up3(ru3, d2)  # 128+64 --->64
        ru5 = self.up4(ru4, d1)  # 64+32 --->32
        seg_out1 = self.final(ru5)
        # downsample 2

        # ds_2 = self.resnet(d2,)
        # downsample-5
        d4 = self.down4(d3)# 256,6,8
        d5 = self.down5(d4)# 512,3,4
        u1 = self.up1(d5, d4) # 256+512 --->256
        u2 = self.up2(u1, d3) #256 128
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        d1f = F.interpolate(d1, x_size[2:], mode='bilinear', align_corners=True)# 16,32,96,128



        s3 = F.interpolate(self.dsn3(d3), x_size[2:],
                            mode='bilinear', align_corners=True) # 16,1,96,128
        s4 = F.interpolate(self.dsn4(d4), x_size[2:],
                            mode='bilinear', align_corners=True) # [ 2,1,120,101] 16,1,96,128
        s7 = F.interpolate(self.dsn7(d5), x_size[2:],
                            mode='bilinear', align_corners=True)# [ 2,1,120,101]
        canny_edge = edge
        cs = self.res1(d1f)#[ 2,64,120,101] 16,32,96,128
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)#[ 2,64,120,101]
        cs = self.d1(cs)#[ 2,32,120,101]  16,32,96,128
        cs = self.gate1(cs, s3) #[ 2,32,120,101]  [ 2,1,120,101] ----> [ 2,32,120,101]
        cs = self.res2(cs)  #[ 2,32,120,101]
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True) #[ 2,32,120,101]
        cs = self.d2(cs) #[ 2,16,120,101]
        cs = self.gate2(cs, s4)#[ 2,16,120,101] [ 2,1,120,101] --------------->[ 2,16,120,101]
        cs = self.res3(cs)#[ 2,16,120,101]
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)#[ 2,16,120,101]
        cs = self.d3(cs)#[ 2,8,120,101]
        cs = self.gate3(cs, s7)#[ 2,8,120,101] [ 2,1,120,101] --------------->[ 2,8,120,101]
        cs = self.fuse(cs)#[ 2,1,120,101]
        cs = F.interpolate(cs, x_size[2:],
                           mode='bilinear', align_corners=True)#[ 2,1,120,101]
        edge_out = self.sigmoid(cs)# [ 2,1,120,101]
        cat = torch.cat((edge_out, canny_edge), dim=1)#[ 2,2,120,101]
        seg_out =self.final(u45)

        return seg_out2,edge_out



