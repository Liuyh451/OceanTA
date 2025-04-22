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
        skip_input = skip_input.size()
        if skip_input[2] == 51 or skip_input[2] == 101 :
            x = F.interpolate(x, skip_input[2:], mode='bilinear', align_corners=True)
        else:
            x = self.model(x)

        x = torch.cat((x, skip_input), 1)
        x = self.shortcut(x)
        return x
class UNetUp_1(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp_1, self).__init__()

        layers = [
            # nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.Conv2d(in_size, out_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        layers2 = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            # nn.Conv2d(in_size, out_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)
        self.model2 = nn.Sequential(*layers2)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, 1, 1, 1, bias=False),

        )

    def forward(self, x, skip_input):
        skip_input_shape = skip_input.size()
        # x2=self.model2(x)
        # if skip_input[2] == 51 or skip_input[2] == 101:
        x_1 = F.interpolate(x, skip_input_shape[2:], mode='bilinear', align_corners=True)

        x_2 = self.model(x_1)

        x = torch.cat((x_2, skip_input), 1)
        x = self.shortcut(x)
        return x
class UNetUp_2(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp_2, self).__init__()

        layers = [
            # nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.Conv2d(in_size, out_size, 1, 1, 0),

            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)


    def forward(self, x, skip_input):
        skip_input_shape = skip_input.size()
        # x2=self.model2(x)
        # if skip_input[2] == 51 or skip_input[2] == 101:
        x = self.model(x)
        x = F.interpolate(x, skip_input_shape[2:], mode='bilinear', align_corners=True)




        return x
class EddyNet(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self, in_channels=1, out_channels=3):
        super(EddyNet, self).__init__()
        # self.res1 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        # self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        # self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        # self.resnet = UNetDown_1(128, 256)
        # self.resnet2 = UNetDown_1(256, 512)
        # self.resnet3 = UNetDown_1(512, 1024)
        # self.resnet = Resnet.BasicBlock(128, 256, stride=1, downsample=None)
        # self.resnet2 = Resnet.BasicBlock(256, 512, stride=1, downsample=None)
        # self.resnet3 = Resnet.BasicBlock(512, 1024, stride=1, downsample=None)
        # self.d1 = nn.Conv2d(32, 32, 1,1)
        # self.d2 = nn.Conv2d(32, 16, 1)
        # self.d3 = nn.Conv2d(16, 8, 1)
        # self.dsn1 = nn.Conv2d(125, 1, 1)
        # self.dsn3 = nn.Conv2d(128, 1, 1)
        # self.dsn4 = nn.Conv2d(256, 1, 1)
        # self.dsn7 = nn.Conv2d(512, 1, 1)

        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 256)
        self.down5 = UNetDown(256, 512, bn=False)

        # decoding layers
        self.up_r1 = UNetUp_1(256, 128)
        self.up_r2 = UNetUp_1(128, 64)
        self.up_r3 = UNetUp_1(64, 32)
        self.up_final = UNetUp_2(32, 3)

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

    def forward(self, x):
        x_size = x.size()
        d1 = self.down1(x)# 16,32,60,51

        d2 = self.down2(d1)# 64,30,26
        d3 = self.down3(d2)# 128,15,13
        d4 = self.down4(d3)# 256,8,7
        d5 = self.down5(d4)# 512,4,4


        ru1 = self.up_r1(d4, d3)  # 1024+512 --->512,15,13
        ru2 = self.up_r2(ru1, d2)  # 512+256 --->256,15,13
        ru3 = self.up_r3(ru2, d1)  # 256+128 --->128,15,13
        seg_out = self.up_final(ru3, x)  # 256+128 --->128,15,13
        # ru4 = self.up3(ru3, d2)  # 128+64 --->64,30,26
        # ru5 = self.up4(ru4, d1)  # 64+32 --->32
        # # seg_out1 = self.final(ru5)
        #
        # u1 = self.up1(d5, d4) # 256+512 --->256
        # u2 = self.up2(u1, d3) #256 128
        # u3 = self.up3(u2, d2)
        # u45 = self.up4(u3, d1)

        # seg_out =self.final(ru3)

        return seg_out



