from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, xavier_normal_
from .basic_layers import PlainConvBlock as PCB
import math

# Attention-ResNet5 (utterance-based)
class AttenResNet5(nn.Module):
    def __init__(self, atten_activation, atten_channel=16, temperature=1, size1=(257,1091), size2=(249,1075), size3=(233,1043), size4=(201,979), size5=(137,851)):

        super(AttenResNet5, self).__init__()

        self.temperature = temperature # temperature softmax

        self.pre = nn.Sequential( # channel expansion
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(4,8)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(8,16)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(16,32)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(32,64)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(64,128)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)

        if atten_channel == 1:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else:
            self.conv1 = nn.Sequential( # channel compression
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )

        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))

        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(2,2)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding


        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 32*4*6

        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        #print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        #print('size2', x.size())
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        #print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        #print('size5', x.size())
        x = self.att5(self.down5(x))
        #print(x.size())
        ## attention block: top-down 
        x = self.att6(skip4 + self.up5(x))
        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x = self.conv1(self.up1(x))
        weight = self.soft(x/self.temperature) # temperature modeling
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual
        #print(x.size())

        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual
        x = self.cnn4(self.mp1(x))
        ##print(x.size())
        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual
        x = self.cnn7(self.mp2(x))
        ##print(x.size())
        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual
        x = self.cnn10(self.mp3(x))
        ##print(x.size())
        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))
        x += residual
        x = self.cnn13(self.mp4(x))
        ##print(x.size())
        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))
        x += residual
        x = self.cnn16(self.mp5(x))
        ##print(x.size())
        ### classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x += residual
        ###
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x += residual
        ###
        out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        ##print(out.size()) 

        return out, weight

# Attention-ResNet4 (utterance-based)
class AttenResNet4(nn.Module):
    def __init__(self, atten_activation, atten_channel=16, size1=(257,1091), size2=(249,1075), size3=(233,1043), size4=(201,979), size5=(137,851), is_print=False, static_quant=False):

        super(AttenResNet4, self).__init__()

        self.is_print = is_print
        self.static_quant = static_quant

        if self.static_quant:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            self.quant_func = nn.quantized.FloatFunctional()

        self.pre = nn.Sequential( # channel expansion
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(4,8)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(8,16)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(16,32)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(32,64)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(64,128)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)

        if atten_channel == 1:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else:
            self.conv1 = nn.Sequential( # channel compression
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )

        if atten_activation == 'tanh':
            self.soft = nn.Tanh()
        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))

        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(2,2)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding


        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 32*4*6

        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re10= nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11= nn.BatchNorm1d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re11= nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def fuse_model(self):
        torch.quantization.fuse_modules(self.pre, [['0', '1', '2'], ['3', '4', '5']], inplace=True)
        for i in range(1, 10):
            torch.quantization.fuse_modules(getattr(self, 'att' + str(i)), ['0', '1', '2'], inplace=True)

    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        if self.is_print: print('size1', x.size())

        ### For static quantization only
        if self.static_quant: x = self.quant(x)

        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        if self.is_print: print('size2', x.size())
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        if self.is_print: print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        if self.is_print: print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        if self.is_print: print('size5', x.size())
        x = self.att5(self.down5(x))
        if self.is_print: print('att5', x.size())

        if not self.static_quant:
            ## attention block: top-down
            x = self.att6(skip4 + self.up5(x))
            if self.is_print: print('att6', x.size())
            x = self.att7(skip3 + self.up4(x))
            if self.is_print: print('att7', x.size())
            x = self.att8(skip2 + self.up3(x))
            if self.is_print: print('att8', x.size())
            x = self.att9(skip1 + self.up2(x))
            if self.is_print: print('att9', x.size())
            x = self.conv1(self.up1(x))
            weight = self.soft(x)
            ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
            x = (1 + weight) * residual
        else:
            ## attention block: top-down
            x = self.att6(self.quant_func.add(skip4, self.up5(x)))
            if self.is_print: print('att6', x.size())
            x = self.att7(self.quant_func.add(skip3, self.up4(x)))
            if self.is_print: print('att7', x.size())
            x = self.att8(self.quant_func.add(skip2, self.up3(x)))
            if self.is_print: print('att8', x.size())
            x = self.att9(self.quant_func.add(skip1, self.up2(x)))
            if self.is_print: print('att9', x.size())
            x = self.conv1(self.up1(x))
            weight = self.soft(x)
            x = self.quant_func.add(residual, self.quant_func.mul(weight, residual))

        if self.is_print: print('before block1', x.size())

        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn4(self.mp1(x))
        if self.is_print: print('block1', x.size())

        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn7(self.mp2(x))
        if self.is_print: print('block2', x.size())

        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn10(self.mp3(x))
        if self.is_print: print('block3', x.size())

        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn13(self.mp4(x))
        if self.is_print: print('block4', x.size())

        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn16(self.mp5(x))
        if self.is_print: print('block5', x.size())

        ### classifier
        # x = x.view(-1, self.flat_feats)
        x = x.reshape(x.shape[0], self.flat_feats)
        x = self.ln1(x)
        residual = x

        if not self.static_quant:
            x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        else:
            # Modify original BatchNorm1d to BatchNorm2d for Quantization support
            x = self.ln3(self.re8(
                self.bn8(
                    self.ln2(self.re7(self.bn7(
                        x.unsqueeze(-1).unsqueeze(-1)
                    ).squeeze(-1).squeeze(-1))).unsqueeze(-1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)
            ))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        ###
        residual = x

        if not self.static_quant:
            x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        else:
            # Modify original BatchNorm1d to BatchNorm2d for Quantization support
            x = self.ln5(self.re10(
                self.bn10(
                    self.ln4(self.re9(self.bn9(
                        x.unsqueeze(-1).unsqueeze(-1)
                    ).squeeze(-1).squeeze(-1))).unsqueeze(-1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)
            ))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        if not self.static_quant:
            out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        else:
            # Modify original BatchNorm1d to BatchNorm2d for Quantization support
            out = self.sigmoid(self.ln6(self.re11(
                self.bn11(
                    x.unsqueeze(-1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)
            )))

        if self.is_print: print('out', out.size())

        ### For static quantization only
        if self.static_quant:
            out = self.dequant(out)
            weight = self.dequant(weight)

        return out, weight


# Attention-ResNet4 (utterance-based) works only for input (257, x), x = 256, 128
class AttenResNet4Deform(nn.Module):
    def __init__(self, atten_activation, atten_channel=16, size1=(257, 1091), size2=(249, 1075), size3=(233, 1043),
                 size4=(201, 979), size5=(137, 851), is_print=False):

        super(AttenResNet4Deform, self).__init__()

        self.is_print = is_print

        self.original_size = (257, 1091)
        self.original_dilation = (4, 8)
        self.ratio = self.original_size[1] // (size1[1] + 1) or 1              # 1091 // 64 = 17, but 1091 // 64 = 16

        self.pre = nn.Sequential(  # channel expansion
            nn.Conv2d(1, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((4, 8))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((8, 16))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((16, 32))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((32, 64))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        self.att5 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((64, 128))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down
        # self.up5 = nn.UpsamplingBilinear2d(size=size5)
        self.up5 = self._update_size(size1, order=5)
        self.att6 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up4 = nn.UpsamplingBilinear2d(size=size4)
        self.up4 = self._update_size(size1, order=4)
        self.att7 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up3 = nn.UpsamplingBilinear2d(size=size3)
        self.up3 = self._update_size(size1, order=3)
        self.att8 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up2 = nn.UpsamplingBilinear2d(size=size2)
        self.up2 = self._update_size(size1, order=2)
        self.att9 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up1 = nn.UpsamplingBilinear2d(size=size1)
        self.up1 = self._update_size(size1, order=1)

        if atten_channel == 1:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else:
            self.conv1 = nn.Sequential(  # channel compression
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel // 4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel // 4, 1, kernel_size=1, stride=1)
            )

        if atten_activation == 'tanh':
            self.soft = nn.Tanh()
        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        ## block 1
        self.bn1 = nn.BatchNorm2d(16)
        self.re1 = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(16)
        self.re2 = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))

        self.mp1 = nn.MaxPool2d(kernel_size=(1, 2))
        # self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3, 3), dilation=self._update_dilation((2, 2)))  # no padding
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3, 3), dilation=(2, 2))  # no padding
        ## block 2
        self.bn3 = nn.BatchNorm2d(32)
        self.re3 = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(32)
        self.re4 = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=(4, 4))  # no padding
        ## block 3
        self.bn5 = nn.BatchNorm2d(32)
        self.re5 = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(32)
        self.re6 = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn10 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=(4, 4))  # no padding
        ## block 4
        self.bn12 = nn.BatchNorm2d(32)
        self.re12 = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn13 = nn.BatchNorm2d(32)
        self.re13 = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=(4, 4))  # no padding
        ## block 5
        self.bn14 = nn.BatchNorm2d(32)
        self.re14 = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn15 = nn.BatchNorm2d(32)
        self.re15 = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp5 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=(2, 2))  # no padding

        # (N x 32 x 8 x 11) to (N x 32*8*11)
        factor = 4 if self.ratio <= 4 else 3
        self.flat_feats = 32 * factor ** 2

        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10 = nn.BatchNorm1d(32)
        self.re10 = nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11 = nn.BatchNorm1d(32)
        self.re11 = nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    @staticmethod
    def _update_size(size, order=1):
        first = 0
        for i in range(3, order + 2):
            first += (2 ** i)

        second = 0
        for i in range(4, order + 3):
            second += (2 ** i)
        return size[0] - first, size[1] - second

    def _update_dilation(self, dilation):
        return (dilation[0], dilation[1] // self.ratio)

    def forward(self, x):
        """input size should be (N x C x H x W), where
        N is batch size
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        if self.is_print: print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        if self.is_print: print('size2', x.size())
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        if self.is_print: print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        if self.is_print: print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        if self.is_print: print('size5', x.size())
        x = self.att5(self.down5(x))
        if self.is_print: print('att5', x.size())
        ## attention block: top-down
        # x = self.att6(skip4 + F.interpolate(x, size=self.up5, mode='bilinear', align_corners=True))
        # print('att6', x.size())
        # x = self.att7(skip3 + F.interpolate(x, size=self.up4, mode='bilinear', align_corners=True))
        # print('att7', x.size())
        # x = self.att8(skip2 + F.interpolate(x, size=self.up3, mode='bilinear', align_corners=True))
        # print('att8', x.size())
        # x = self.att9(skip1 + F.interpolate(x, size=self.up2, mode='bilinear', align_corners=True))
        # print('att9', x.size())
        # x = self.conv1(F.interpolate(x, size=self.up1, mode='bilinear', align_corners=True))

        x = self.att6(skip4 + F.interpolate(x, size=skip4.shape[-2:], mode='bilinear', align_corners=True))
        if self.is_print: print('att6', x.size())
        x = self.att7(skip3 + F.interpolate(x, size=skip3.shape[-2:], mode='bilinear', align_corners=True))
        if self.is_print: print('att7', x.size())
        x = self.att8(skip2 + F.interpolate(x, size=skip2.shape[-2:], mode='bilinear', align_corners=True))
        if self.is_print: print('att8', x.size())
        x = self.att9(skip1 + F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=True))
        if self.is_print: print('att9', x.size())
        x = self.conv1(F.interpolate(x, size=residual.shape[-2:], mode='bilinear', align_corners=True))

        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual
        if self.is_print: print('before block1', x.size())

        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual
        # x = self.cnn4(self.mp1(x))
        x = self.cnn4(F.adaptive_avg_pool2d(x, output_size=tuple([min(x.shape[2:])] * 2)))
        if self.is_print: print('block1', x.size())

        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual
        x = self.cnn7(self.mp2(x))
        if self.is_print: print('block2', x.size())

        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual
        x = self.cnn10(self.mp3(x))
        if self.is_print: print('block3', x.size())

        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))
        x += residual
        if self.ratio <= 4:
            x = self.mp4(x)
        # x = self.mp4(x)
        x = self.cnn13(x)
        if self.is_print: print('block4', x.size())

        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))
        x += residual
        if self.ratio <= 4:
            x = self.mp5(x)
            x = self.cnn16(x)
        else:
            x = self.cnn16(x)
            x = self.mp5(x)
        if self.is_print: print('block5', x.size())

        ### classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x += residual
        ###
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x += residual
        ###
        out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        if self.is_print: print('out', out.size())

        return out, weight


# the modified network for input with dimension (257,512)
class AttenResNet4Deform_512(nn.Module):
    def __init__(self, atten_activation, atten_channel=16, size1=(257, 1091), size2=(249, 1075), size3=(233, 1043),
                 size4=(201, 979), size5=(137, 851), is_print=False):

        super(AttenResNet4Deform_512, self).__init__()

        self.is_print = is_print

        self.original_size = (257, 1091)
        self.original_dilation = (4, 8)
        self.ratio = self.original_size[1] // (size1[1] + 1) or 1              # 1091 // 64 = 17, but 1091 // 65 = 16
        print('ratio:', self.ratio)

        self.pre = nn.Sequential(  # channel expansion
            nn.Conv2d(1, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((4, 8))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((8, 16))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((16, 32))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((32, 64))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        self.att5 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((64, 128))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down
        # self.up5 = nn.UpsamplingBilinear2d(size=size5)
        self.up5 = self._update_size(size1, order=5)
        self.att6 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up4 = nn.UpsamplingBilinear2d(size=size4)
        self.up4 = self._update_size(size1, order=4)
        self.att7 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up3 = nn.UpsamplingBilinear2d(size=size3)
        self.up3 = self._update_size(size1, order=3)
        self.att8 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up2 = nn.UpsamplingBilinear2d(size=size2)
        self.up2 = self._update_size(size1, order=2)
        self.att9 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up1 = nn.UpsamplingBilinear2d(size=size1)
        self.up1 = self._update_size(size1, order=1)

        if atten_channel == 1:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else:
            self.conv1 = nn.Sequential(  # channel compression
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel // 4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel // 4, 1, kernel_size=1, stride=1)
            )

        if atten_activation == 'tanh':
            self.soft = nn.Tanh()
        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        ## block 1
        self.bn1 = nn.BatchNorm2d(16)
        self.re1 = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(16)
        self.re2 = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))

        self.mp1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3, 3), dilation=self._update_dilation((2, 2)))  # no padding
        ## block 2
        self.bn3 = nn.BatchNorm2d(32)
        self.re3 = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(32)
        self.re4 = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((4, 4)))  # no padding
        ## block 3
        self.bn5 = nn.BatchNorm2d(32)
        self.re5 = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(32)
        self.re6 = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp3 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn10 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((4, 4)))  # no padding
        ## block 4
        self.bn12 = nn.BatchNorm2d(32)
        self.re12 = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn13 = nn.BatchNorm2d(32)
        self.re13 = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp4 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((8, 8)))  # no padding
        ## block 5
        self.bn14 = nn.BatchNorm2d(32)
        self.re14 = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn15 = nn.BatchNorm2d(32)
        self.re15 = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp5 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((8, 8)))  # no padding

        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 32 * 4 * (6 // (self.ratio + 1) or 1)

        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10 = nn.BatchNorm1d(32)
        self.re10 = nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11 = nn.BatchNorm1d(32)
        self.re11 = nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    @staticmethod
    def _update_size(size, order=1):
        first = 0
        for i in range(3, order + 2):
            first += (2 ** i)

        second = 0
        for i in range(4, order + 3):
            second += (2 ** i)
        return size[0] - first, size[1] - second

    def _update_dilation(self, dilation):
        return (dilation[0], max(dilation[1] // self.ratio, 1))

    def forward(self, x):
        """input size should be (N x C x H x W), where
        N is batch size
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        if self.is_print: print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        if self.is_print: print('size2', x.size())
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        if self.is_print: print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        if self.is_print: print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        if self.is_print: print('size5', x.size())
        x = self.att5(self.down5(x))
        if self.is_print: print('att5', x.size())
        ## attention block: top-down
        # x = self.att6(skip4 + F.interpolate(x, size=self.up5, mode='bilinear', align_corners=True))
        # print('att6', x.size())
        # x = self.att7(skip3 + F.interpolate(x, size=self.up4, mode='bilinear', align_corners=True))
        # print('att7', x.size())
        # x = self.att8(skip2 + F.interpolate(x, size=self.up3, mode='bilinear', align_corners=True))
        # print('att8', x.size())
        # x = self.att9(skip1 + F.interpolate(x, size=self.up2, mode='bilinear', align_corners=True))
        # print('att9', x.size())
        # x = self.conv1(F.interpolate(x, size=self.up1, mode='bilinear', align_corners=True))

        x = self.att6(skip4 + F.interpolate(x, size=skip4.shape[-2:], mode='bilinear', align_corners=True))
        if self.is_print: print('att6', x.size())
        x = self.att7(skip3 + F.interpolate(x, size=skip3.shape[-2:], mode='bilinear', align_corners=True))
        if self.is_print: print('att7', x.size())
        x = self.att8(skip2 + F.interpolate(x, size=skip2.shape[-2:], mode='bilinear', align_corners=True))
        if self.is_print: print('att8', x.size())
        x = self.att9(skip1 + F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=True))
        if self.is_print: print('att9', x.size())
        x = self.conv1(F.interpolate(x, size=residual.shape[-2:], mode='bilinear', align_corners=True))

        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual
        if self.is_print: print('before block1', x.size())

        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual
        x = self.cnn4(self.mp1(x))
        if self.is_print: print('block1', x.size())

        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual
        x = self.cnn7(self.mp2(x))
        if self.is_print: print('block2', x.size())

        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual
        x = self.cnn10(self.mp3(x))
        if self.is_print: print('block3', x.size())

        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))
        x += residual
        x = self.cnn13(self.mp4(x))
        if self.is_print: print('block4', x.size())

        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))
        x += residual
        x = self.cnn16(self.mp5(x))
        if self.is_print: print('block5', x.size())

        ### classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x += residual
        ###
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x += residual
        ###
        out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        if self.is_print: print('out', out.size())

        return out, weight


# support 257 x 512, 256, 128, 64
class AttenResNet4DeformAll(nn.Module):
    def __init__(self, atten_activation, atten_channel=16, size1=(257, 1091), size2=(249, 1075), size3=(233, 1043),
                 size4=(201, 979), size5=(137, 851), is_print=False, static_quant=False):

        super(AttenResNet4DeformAll, self).__init__()

        self.is_print = is_print
        self.static_quant = static_quant

        if self.static_quant:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            self.quant_func = nn.quantized.FloatFunctional()

        self.original_size = (257, 1091)
        self.original_dilation = (4, 8)
        self.ratio = self.original_size[1] // (size1[1] + 1) or 1              # 1091 // 64 = 17, but 1091 // 65 = 16
        if self.is_print: print('ratio:', self.ratio)

        self.pre = nn.Sequential(  # channel expansion
            nn.Conv2d(1, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((4, 8))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((8, 16))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((16, 32))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((32, 64))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        self.att5 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((64, 128))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down
        # self.up5 = nn.UpsamplingBilinear2d(size=size5)
        self.up5 = self._update_size(size1, order=5)
        self.att6 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up4 = nn.UpsamplingBilinear2d(size=size4)
        self.up4 = self._update_size(size1, order=4)
        self.att7 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up3 = nn.UpsamplingBilinear2d(size=size3)
        self.up3 = self._update_size(size1, order=3)
        self.att8 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up2 = nn.UpsamplingBilinear2d(size=size2)
        self.up2 = self._update_size(size1, order=2)
        self.att9 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up1 = nn.UpsamplingBilinear2d(size=size1)
        self.up1 = self._update_size(size1, order=1)

        if atten_channel == 1:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else:
            self.conv1 = nn.Sequential(  # channel compression
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel // 4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel // 4, 1, kernel_size=1, stride=1)
            )

        if atten_activation == 'tanh':
            self.soft = nn.Tanh()
        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        ## block 1
        self.bn1 = nn.BatchNorm2d(16)
        self.re1 = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(16)
        self.re2 = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))

        self.mp1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3, 3), dilation=self._update_dilation((2, 2)))  # no padding
        ## block 2
        self.bn3 = nn.BatchNorm2d(32)
        self.re3 = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(32)
        self.re4 = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((4, 4)))  # no padding
        ## block 3
        self.bn5 = nn.BatchNorm2d(32)
        self.re5 = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(32)
        self.re6 = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        mp3_shape = (2, 2)
        if math.log2(self.ratio) >= 3:
            mp3_shape = (2, 1)
        self.mp3 = nn.MaxPool2d(kernel_size=mp3_shape, stride=mp3_shape)
        self.cnn10 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((4, 4)))  # no padding
        ## block 4
        self.bn12 = nn.BatchNorm2d(32)
        self.re12 = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn13 = nn.BatchNorm2d(32)
        self.re13 = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        mp4_shape = (2, 2)
        if math.log2(self.ratio) == 4:
            mp4_shape = (2, 1)
        self.mp4 = nn.MaxPool2d(kernel_size=mp4_shape, stride=mp4_shape)
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((8, 8)))  # no padding
        ## block 5
        self.bn14 = nn.BatchNorm2d(32)
        self.re14 = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn15 = nn.BatchNorm2d(32)
        self.re15 = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp5 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((8, 8)))  # no padding

        # {ratio: value} = {1: 6, 2: 2, 4: 1, 8: 3, 16: 2}
        flat_dim = [6, 2, 1, 3, 2]
        self.flat_feats = 32 * 4 * flat_dim[int(math.log2(self.ratio))]

        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10 = nn.BatchNorm1d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re10 = nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11 = nn.BatchNorm1d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re11 = nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    @staticmethod
    def _update_size(size, order=1):
        first = 0
        for i in range(3, order + 2):
            first += (2 ** i)

        second = 0
        for i in range(4, order + 3):
            second += (2 ** i)
        return size[0] - first, size[1] - second

    def _update_dilation(self, dilation):
        return (dilation[0], max(dilation[1] // self.ratio, 1))

    def fuse_model(self):
        torch.quantization.fuse_modules(self.pre, [['0', '1', '2'], ['3', '4', '5']], inplace=True)
        for i in range(1, 10):
            torch.quantization.fuse_modules(getattr(self, 'att' + str(i)), ['0', '1', '2'], inplace=True)

    def forward(self, x):
        """input size should be (N x C x H x W), where
        N is batch size
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        if self.is_print: print('size1', x.size())

        ### For static quantization only
        if self.static_quant: x = self.quant(x)

        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        if self.is_print: print('size2', x.size())
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        if self.is_print: print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        if self.is_print: print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        if self.is_print: print('size5', x.size())
        x = self.att5(self.down5(x))
        if self.is_print: print('att5', x.size())
        ## attention block: top-down
        # x = self.att6(skip4 + F.interpolate(x, size=self.up5, mode='bilinear', align_corners=True))
        # print('att6', x.size())
        # x = self.att7(skip3 + F.interpolate(x, size=self.up4, mode='bilinear', align_corners=True))
        # print('att7', x.size())
        # x = self.att8(skip2 + F.interpolate(x, size=self.up3, mode='bilinear', align_corners=True))
        # print('att8', x.size())
        # x = self.att9(skip1 + F.interpolate(x, size=self.up2, mode='bilinear', align_corners=True))
        # print('att9', x.size())
        # x = self.conv1(F.interpolate(x, size=self.up1, mode='bilinear', align_corners=True))

        if not self.static_quant:
            x = self.att6(skip4 + F.interpolate(x, size=skip4.shape[-2:], mode='bilinear', align_corners=True))
            if self.is_print: print('att6', x.size())
            x = self.att7(skip3 + F.interpolate(x, size=skip3.shape[-2:], mode='bilinear', align_corners=True))
            if self.is_print: print('att7', x.size())
            x = self.att8(skip2 + F.interpolate(x, size=skip2.shape[-2:], mode='bilinear', align_corners=True))
            if self.is_print: print('att8', x.size())
            x = self.att9(skip1 + F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=True))
            if self.is_print: print('att9', x.size())
            x = self.conv1(F.interpolate(x, size=residual.shape[-2:], mode='bilinear', align_corners=True))

            weight = self.soft(x)
            ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
            x = (1 + weight) * residual
        else:
            x = self.att6(self.quant_func.add(skip4, F.interpolate(x, size=skip4.shape[-2:], mode='bilinear', align_corners=True)))
            if self.is_print: print('att6', x.size())
            x = self.att7(self.quant_func.add(skip3, F.interpolate(x, size=skip3.shape[-2:], mode='bilinear', align_corners=True)))
            if self.is_print: print('att7', x.size())
            x = self.att8(self.quant_func.add(skip2, F.interpolate(x, size=skip2.shape[-2:], mode='bilinear', align_corners=True)))
            if self.is_print: print('att8', x.size())
            x = self.att9(self.quant_func.add(skip1, F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=True)))
            if self.is_print: print('att9', x.size())
            x = self.conv1(F.interpolate(x, size=residual.shape[-2:], mode='bilinear', align_corners=True))

            weight = self.soft(x)
            ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
            x = self.quant_func.add(residual, self.quant_func.mul(weight, residual))

        if self.is_print: print('before block1', x.size())

        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn4(self.mp1(x))
        if self.is_print: print('block1', x.size())

        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn7(self.mp2(x))
        if self.is_print: print('block2', x.size())

        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn10(self.mp3(x))
        if self.is_print: print('block3', x.size())

        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn13(self.mp4(x))
        if self.is_print: print('block4', x.size())

        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn16(self.mp5(x))
        if self.is_print: print('block5', x.size())

        ### classifier
        x = x.reshape(x.shape[0], self.flat_feats)
        x = self.ln1(x)
        residual = x

        if not self.static_quant:
            x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        else:
            # Modify original BatchNorm1d to BatchNorm2d for Quantization support
            x = self.ln3(self.re8(
                self.bn8(
                    self.ln2(self.re7(self.bn7(
                        x.unsqueeze(-1).unsqueeze(-1)
                    ).squeeze(-1).squeeze(-1))).unsqueeze(-1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)
            ))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        ###
        residual = x

        if not self.static_quant:
            x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        else:
            # Modify original BatchNorm1d to BatchNorm2d for Quantization support
            x = self.ln5(self.re10(
                self.bn10(
                    self.ln4(self.re9(self.bn9(
                        x.unsqueeze(-1).unsqueeze(-1)
                    ).squeeze(-1).squeeze(-1))).unsqueeze(-1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)
            ))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        ###
        if not self.static_quant:
            out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        else:
            # Modify original BatchNorm1d to BatchNorm2d for Quantization support
            out = self.sigmoid(self.ln6(self.re11(
                self.bn11(
                    x.unsqueeze(-1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1)
            )))

        if self.is_print: print('out', out.size())

        ### For static quantization only
        if self.static_quant:
            out = self.dequant(out)
            weight = self.dequant(weight)

        return out, weight

# support [257, 128, 64] x [1091, 512, 256, 128, 64]
class AttenResNet4DeformBoth(nn.Module):
    def __init__(self, atten_activation, atten_channel=16, size1=(257, 1091), size2=(249, 1075), size3=(233, 1043),
                 size4=(201, 979), size5=(137, 851), is_print=False, static_quant=False):

        super(AttenResNet4DeformBoth, self).__init__()

        self.is_print = is_print
        self.static_quant = static_quant

        if self.static_quant:
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
            self.quant_func = nn.quantized.FloatFunctional()

        self.original_size = (257, 1091)
        self.original_dilation = (4, 8)
        self.ratio = self.original_size[0] // size1[0], self.original_size[1] // (size1[1] + 1) or 1              # 1091 // 64 = 17, but 1091 // 65 = 16
        if self.is_print: print('ratio:', self.ratio)

        self.pre = nn.Sequential(  # channel expansion
            nn.Conv2d(1, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((4, 8))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((8, 16))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((16, 32))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.att4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((32, 64))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1, 2))
        self.att5 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1), dilation=self._update_dilation((64, 128))),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down
        # self.up5 = nn.UpsamplingBilinear2d(size=size5)
        self.up5 = self._update_size(size1, order=5)
        self.att6 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up4 = nn.UpsamplingBilinear2d(size=size4)
        self.up4 = self._update_size(size1, order=4)
        self.att7 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up3 = nn.UpsamplingBilinear2d(size=size3)
        self.up3 = self._update_size(size1, order=3)
        self.att8 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up2 = nn.UpsamplingBilinear2d(size=size2)
        self.up2 = self._update_size(size1, order=2)
        self.att9 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        # self.up1 = nn.UpsamplingBilinear2d(size=size1)
        self.up1 = self._update_size(size1, order=1)

        if atten_channel == 1:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else:
            self.conv1 = nn.Sequential(  # channel compression
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel // 4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel // 4, 1, kernel_size=1, stride=1)
            )

        if atten_activation == 'tanh':
            self.soft = nn.Tanh()
        if atten_activation == 'softmax2':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'softmax3':
            self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=(1, 1))
        ## block 1
        self.bn1 = nn.BatchNorm2d(16)
        self.re1 = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(16)
        self.re2 = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1))

        self.mp1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3, 3), dilation=self._update_dilation((2, 2)))  # no padding
        ## block 2
        self.bn3 = nn.BatchNorm2d(32)
        self.re3 = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(32)
        self.re4 = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((4, 4)))  # no padding
        ## block 3
        self.bn5 = nn.BatchNorm2d(32)
        self.re5 = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(32)
        self.re6 = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        mp3_shape = (2, 2)
        if math.log2(self.ratio[1]) >= 3:
            mp3_shape = (2, 1)
        self.mp3 = nn.MaxPool2d(kernel_size=mp3_shape, stride=mp3_shape)
        self.cnn10 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((4, 4)))  # no padding
        ## block 4
        self.bn12 = nn.BatchNorm2d(32)
        self.re12 = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn13 = nn.BatchNorm2d(32)
        self.re13 = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        mp4_shape = (2, 2)
        if math.log2(self.ratio[1]) == 4:
            mp4_shape = (2, 1)
        self.mp4 = nn.MaxPool2d(kernel_size=mp4_shape, stride=mp4_shape)
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((8, 8)))  # no padding
        ## block 5
        self.bn14 = nn.BatchNorm2d(32)
        self.re14 = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.bn15 = nn.BatchNorm2d(32)
        self.re15 = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))

        self.mp5 = nn.MaxPool2d(kernel_size=(2, 2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3, 3), dilation=self._update_dilation((8, 8)))  # no padding

        # {ratio: value} = {1: 6, 2: 2, 4: 1, 8: 3, 16: 2}
        flat_dim = [6, 2, 1, 3, 2]
        self.flat_feats = 32 * (4 // self.ratio[0]) * flat_dim[int(math.log2(self.ratio[1]))]

        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm2d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm2d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm2d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10 = nn.BatchNorm2d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re10 = nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11 = nn.BatchNorm2d(32)           # modify original BatchNorm1d to BatchNorm2d for Quantization support
        self.re11 = nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    @staticmethod
    def _update_size(size, order=1):
        first = 0
        for i in range(3, order + 2):
            first += (2 ** i)

        second = 0
        for i in range(4, order + 3):
            second += (2 ** i)
        return size[0] - first, size[1] - second

    def _update_dilation(self, dilation):
        return (max(dilation[0] // self.ratio[0], 1), max(dilation[1] // self.ratio[1], 1))

    def fuse_model(self):
        torch.quantization.fuse_modules(self.pre, [['0', '1', '2'], ['3', '4', '5']], inplace=True)
        for i in range(1, 10):
            torch.quantization.fuse_modules(getattr(self, 'att' + str(i)), ['0', '1', '2'], inplace=True)

    def forward(self, x):
        """input size should be (N x C x H x W), where
        N is batch size
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        if self.is_print: print('size1', x.size())

        ### For static quantization only
        if self.static_quant: x = self.quant(x)

        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        if self.is_print: print('size2', x.size())
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        if self.is_print: print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        if self.is_print: print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        if self.is_print: print('size5', x.size())
        # x = self.att5(self.down5(x))

        x5 = self.down5(x)
        if self.is_print: print('down5', x5.size())
        x = self.att5(x5)

        if self.is_print: print('att5', x.size())
        ## attention block: top-down
        # x = self.att6(skip4 + F.interpolate(x, size=self.up5, mode='bilinear', align_corners=True))
        # print('att6', x.size())
        # x = self.att7(skip3 + F.interpolate(x, size=self.up4, mode='bilinear', align_corners=True))
        # print('att7', x.size())
        # x = self.att8(skip2 + F.interpolate(x, size=self.up3, mode='bilinear', align_corners=True))
        # print('att8', x.size())
        # x = self.att9(skip1 + F.interpolate(x, size=self.up2, mode='bilinear', align_corners=True))
        # print('att9', x.size())
        # x = self.conv1(F.interpolate(x, size=self.up1, mode='bilinear', align_corners=True))

        if not self.static_quant:
            x = self.att6(skip4 + F.interpolate(x, size=skip4.shape[-2:], mode='bilinear', align_corners=True))
            if self.is_print: print('att6', x.size())
            x = self.att7(skip3 + F.interpolate(x, size=skip3.shape[-2:], mode='bilinear', align_corners=True))
            if self.is_print: print('att7', x.size())
            x = self.att8(skip2 + F.interpolate(x, size=skip2.shape[-2:], mode='bilinear', align_corners=True))
            if self.is_print: print('att8', x.size())
            x = self.att9(skip1 + F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=True))
            if self.is_print: print('att9', x.size())
            x = self.conv1(F.interpolate(x, size=residual.shape[-2:], mode='bilinear', align_corners=True))

            weight = self.soft(x)
            ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
            x = (1 + weight) * residual
        else:
            x = self.att6(self.quant_func.add(skip4, F.interpolate(x, size=skip4.shape[-2:], mode='bilinear', align_corners=True)))
            if self.is_print: print('att6', x.size())
            x = self.att7(self.quant_func.add(skip3, F.interpolate(x, size=skip3.shape[-2:], mode='bilinear', align_corners=True)))
            if self.is_print: print('att7', x.size())
            x = self.att8(self.quant_func.add(skip2, F.interpolate(x, size=skip2.shape[-2:], mode='bilinear', align_corners=True)))
            if self.is_print: print('att8', x.size())
            x = self.att9(self.quant_func.add(skip1, F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=True)))
            if self.is_print: print('att9', x.size())
            x = self.conv1(F.interpolate(x, size=residual.shape[-2:], mode='bilinear', align_corners=True))

            weight = self.soft(x)
            ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
            x = self.quant_func.add(residual, self.quant_func.mul(weight, residual))

        if self.is_print: print('before block1', x.size())

        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn4(self.mp1(x))
        if self.is_print: print('block1', x.size())

        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn7(self.mp2(x))
        if self.is_print: print('block2', x.size())

        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn10(self.mp3(x))
        if self.is_print: print('block3', x.size())

        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn13(self.mp4(x))
        if self.is_print: print('block4', x.size())

        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        x = self.cnn16(self.mp5(x))
        if self.is_print: print('block5', x.size())

        ### classifier
        x = x.reshape(x.shape[0], self.flat_feats)
        x = self.ln1(x)
        residual = x

        # Modify original BatchNorm1d to BatchNorm2d for Quantization support
        x = self.ln3(self.re8(
            self.bn8(
                self.ln2(self.re7(self.bn7(
                    x.unsqueeze(-1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1))).unsqueeze(-1).unsqueeze(-1)
            ).squeeze(-1).squeeze(-1)
        ))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        ###
        residual = x

        # Modify original BatchNorm1d to BatchNorm2d for Quantization support
        x = self.ln5(self.re10(
            self.bn10(
                self.ln4(self.re9(self.bn9(
                    x.unsqueeze(-1).unsqueeze(-1)
                ).squeeze(-1).squeeze(-1))).unsqueeze(-1).unsqueeze(-1)
            ).squeeze(-1).squeeze(-1)
        ))

        if not self.static_quant:
            x += residual
        else:
            x = self.quant_func.add(x, residual)

        # Modify original BatchNorm1d to BatchNorm2d for Quantization support
        out = self.sigmoid(self.ln6(self.re11(
            self.bn11(
                x.unsqueeze(-1).unsqueeze(-1)
            ).squeeze(-1).squeeze(-1)
        )))

        if self.is_print: print('out', out.size())

        ### For static quantization only
        if self.static_quant:
            out = self.dequant(out)
            weight = self.dequant(weight)

        return out, weight

# Attention-ResNet3 (utterance-based) 
class AttenResNet3(nn.Module):
    def __init__(self, atten_activation, atten_channel=16, size1=(257,1091), size2=(249,1075), size3=(233,1043), size4=(201,979), size5=(137,851)):

        super(AttenResNet3, self).__init__()

        self.pre = nn.Sequential( # channel expansion
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = PCB(atten_channel, dilation=(4,8))
        self.skip1 = PCB(atten_channel)

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = PCB(atten_channel, dilation=(8,16))
        self.skip2 = PCB(atten_channel)

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = PCB(atten_channel, dilation=(16,32))
        self.skip3 = PCB(atten_channel)

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = PCB(atten_channel, dilation=(32,64))
        self.skip4 = PCB(atten_channel)

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = PCB(atten_channel, dilation=(64,128))

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = PCB(atten_channel)

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = PCB(atten_channel)

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = PCB(atten_channel)

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = PCB(atten_channel)

        self.up1   = nn.UpsamplingBilinear2d(size=size1)

        if atten_channel == 1:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else:
            self.conv1 = nn.Sequential( # channel compression
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )

        if atten_activation == 'softmax':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))

        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(2,2)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding


        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 32*4*6

        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        #print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        #print('size2', x.size())
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        #print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        #print('size5', x.size())
        x = self.att5(self.down5(x))
        #print(x.size())
        ## attention block: top-down 
        x = self.att6(skip4 + self.up5(x))
        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x = self.conv1(self.up1(x))
        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual
        #print(x.size())

        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual
        x = self.cnn4(self.mp1(x))
        ##print(x.size())
        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual
        x = self.cnn7(self.mp2(x))
        ##print(x.size())
        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual
        x = self.cnn10(self.mp3(x))
        ##print(x.size())
        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))
        x += residual
        x = self.cnn13(self.mp4(x))
        ##print(x.size())
        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))
        x += residual
        x = self.cnn16(self.mp5(x))
        ##print(x.size())
        ### classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x += residual
        ###
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x += residual
        ###
        out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        ##print(out.size()) 

        return out

# Attention-ResNet2 (utterance-based)
class AttenResNet2(nn.Module):
    def __init__(self, atten_activation, atten_channel=16, size1=(257,1091), size2=(249,1075), size3=(233,1043), size4=(201,979), size5=(137,851)):

        super(AttenResNet2, self).__init__()

        self.pre = nn.Sequential( # channel expansion
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(4,8)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(8,16)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(16,32)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(32,64)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(64,128)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)

        if atten_channel == 1:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else:
            self.conv1 = nn.Sequential( # channel compression
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )

        if atten_activation == 'softmax':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))

        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(2,2)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding


        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 32*4*6

        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        #print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        #print('size2', x.size())
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        #print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        #print('size5', x.size())
        x = self.att5(self.down5(x))
        #print(x.size())
        ## attention block: top-down 
        x = self.att6(skip4 + self.up5(x))
        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x = self.conv1(self.up1(x))
        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        x = (1 + weight) * residual
        #print(x.size())

        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual
        x = self.cnn4(self.mp1(x))
        ##print(x.size())
        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual
        x = self.cnn7(self.mp2(x))
        ##print(x.size())
        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual
        x = self.cnn10(self.mp3(x))
        ##print(x.size())
        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))
        x += residual
        x = self.cnn13(self.mp4(x))
        ##print(x.size())
        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))
        x += residual
        x = self.cnn16(self.mp5(x))
        ##print(x.size())
        ### classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x += residual
        ###
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x += residual
        ###
        out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        ##print(out.size()) 

        return out, weight

# Pretrained-Attention-ResNet (utterance-based)
class PreAttenResNet(nn.Module):
    def __init__(self, pretrain_resnet, atten_activation, atten_channel=16, size1=(257,1091), size2=(249,1075), size3=(233,1043), size4=(201,979), size5=(137,851)):

        super(PreAttenResNet, self).__init__()

        self.resnet = pretrain_resnet # load pretrained resnet

        self.channel_expansion = nn.Sequential( # channel expansion
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(4,8)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(8,16)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(16,32)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(32,64)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(64,128)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)

        if atten_channel == 1:
            self.channel_compression = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else:
            self.channel_compression = nn.Sequential( # channel compression
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )

        if atten_activation == 'softmax':
            self.soft = nn.Softmax(dim=2)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        #print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.channel_expansion(x)
        x = self.att1(self.down1(x))
        skip1 = self.skip1(x)
        #print('size2', x.size())
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        #print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        #print('size5', x.size())
        x = self.att5(self.down5(x))
        #print(x.size())
        ## attention block: top-down 
        x = self.att6(skip4 + self.up5(x))
        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x = self.channel_compression(self.up1(x))
        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1 for softmax 
        #x = (1 + weight) * residual
        x = weight * residual
        #print(x.size())

        out = self.resnet(x) # pre-trained resnet

        return out


# Attention-ResNet (utterance-based) 
class AttenResNet(nn.Module):
    def __init__(self, atten_activation, atten_channel=16, size1=(257,1091), size2=(249,1075), size3=(233,1043), size4=(201,979), size5=(137,851)):

        super(AttenResNet, self).__init__()

        self.pre = nn.Sequential( # channel expansion
            nn.Conv2d(1, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        ## attention branch: bottom-up
        self.down1 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att1  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(4,8)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip1 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att2  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(8,16)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down3 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att3  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(16,32)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down4 = nn.MaxPool2d(kernel_size=3, stride=(1,1))
        self.att4  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(32,64)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )
        self.skip4 = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.down5 = nn.MaxPool2d(kernel_size=3, stride=(1,2))
        self.att5  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1), dilation=(64,128)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        ## attention branch: top-down 
        self.up5   = nn.UpsamplingBilinear2d(size=size5)
        self.att6  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up4   = nn.UpsamplingBilinear2d(size=size4)
        self.att7  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up3   = nn.UpsamplingBilinear2d(size=size3)
        self.att8  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up2   = nn.UpsamplingBilinear2d(size=size2)
        self.att9  = nn.Sequential(
            nn.Conv2d(atten_channel, atten_channel, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(atten_channel),
            nn.ReLU(inplace=True)
        )

        self.up1   = nn.UpsamplingBilinear2d(size=size1)

        if atten_channel == 1:
            self.conv1 = nn.Sequential(
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, 1, kernel_size=1, stride=1)
            )
        else:
            self.conv1 = nn.Sequential( # channel compression
                nn.BatchNorm2d(atten_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel, atten_channel//4, kernel_size=1, stride=1),
                nn.BatchNorm2d(atten_channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(atten_channel//4, 1, kernel_size=1, stride=1)
            )

        if atten_activation == 'softmax':
            self.soft = nn.Softmax(dim=2)
            #self.soft = nn.Softmax(dim=3)
        if atten_activation == 'sigmoid':
            self.soft  = nn.Sigmoid()

        self.cnn1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=(1,1))
        ## block 1
        self.bn1  = nn.BatchNorm2d(16)
        self.re1  = nn.ReLU(inplace=True)
        self.cnn2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2  = nn.BatchNorm2d(16)
        self.re2  = nn.ReLU(inplace=True)
        self.cnn3 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=(1,1))

        self.mp1  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn4 = nn.Conv2d(16, 32, kernel_size=(3,3), dilation=(2,2)) # no padding
        ## block 2
        self.bn3  = nn.BatchNorm2d(32)
        self.re3  = nn.ReLU(inplace=True)
        self.cnn5 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn4  = nn.BatchNorm2d(32)
        self.re4  = nn.ReLU(inplace=True)
        self.cnn6 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp2  = nn.MaxPool2d(kernel_size=(1,2))
        self.cnn7 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 3
        self.bn5  = nn.BatchNorm2d(32)
        self.re5  = nn.ReLU(inplace=True)
        self.cnn8 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn6  = nn.BatchNorm2d(32)
        self.re6  = nn.ReLU(inplace=True)
        self.cnn9 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp3  = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn10= nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(4,4)) # no padding
        ## block 4
        self.bn12  = nn.BatchNorm2d(32)
        self.re12  = nn.ReLU(inplace=True)
        self.cnn11 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn13  = nn.BatchNorm2d(32)
        self.re13  = nn.ReLU(inplace=True)
        self.cnn12 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp4   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn13 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding
        ## block 5
        self.bn14  = nn.BatchNorm2d(32)
        self.re14  = nn.ReLU(inplace=True)
        self.cnn14 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))
        self.bn15  = nn.BatchNorm2d(32)
        self.re15  = nn.ReLU(inplace=True)
        self.cnn15 = nn.Conv2d(32, 32, kernel_size=(3,3), padding=(1,1))

        self.mp5   = nn.MaxPool2d(kernel_size=(2,2))
        self.cnn16 = nn.Conv2d(32, 32, kernel_size=(3,3), dilation=(8,8)) # no padding


        # (N x 32 x 8 x 11) to (N x 32*8*11)
        self.flat_feats = 32*4*6

        # fc
        self.ln1 = nn.Linear(self.flat_feats, 32)
        self.bn7 = nn.BatchNorm1d(32)
        self.re7 = nn.ReLU(inplace=True)
        self.ln2 = nn.Linear(32, 32)
        self.bn8 = nn.BatchNorm1d(32)
        self.re8 = nn.ReLU(inplace=True)
        self.ln3 = nn.Linear(32, 32)
        ####
        self.bn9 = nn.BatchNorm1d(32)
        self.re9 = nn.ReLU(inplace=True)
        self.ln4 = nn.Linear(32, 32)
        self.bn10= nn.BatchNorm1d(32)
        self.re10= nn.ReLU(inplace=True)
        self.ln5 = nn.Linear(32, 32)
        ###
        self.bn11= nn.BatchNorm1d(32)
        self.re11= nn.ReLU(inplace=True)
        self.ln6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        ## Weights initialization
        def _weights_init(m):
            if isinstance(m, nn.Conv2d or nn.Linear):
                xavier_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.apply(_weights_init)

    def forward(self, x):
        """input size should be (N x C x H x W), where 
        N is batch size 
        C is channel (usually 1 unless static, velocity and acceleration are used)
        H is feature dim (e.g. 40 for fbank)
        W is time frames (e.g. 2*(M + 1))
        """
        ### feature
        #print('size1', x.size())
        residual = x
        ## attention block: bottom-up
        x = self.att1(self.down1(self.pre(x)))
        skip1 = self.skip1(x)
        #print('size2', x.size())
        x = self.att2(self.down2(x))
        skip2 = self.skip2(x)
        #print('size3', x.size())
        x = self.att3(self.down3(x))
        skip3 = self.skip3(x)
        #print('size4', x.size())
        x = self.att4(self.down4(x))
        skip4 = self.skip4(x)
        #print('size5', x.size())
        x = self.att5(self.down5(x))
        #print(x.size())
        ## attention block: top-down 
        x = self.att6(skip4 + self.up5(x))
        x = self.att7(skip3 + self.up4(x))
        x = self.att8(skip2 + self.up3(x))
        x = self.att9(skip1 + self.up2(x))
        x = self.conv1(self.up1(x))
        weight = self.soft(x)
        ##print(torch.sum(weight,dim=2)) # attention weight sum to 1
        #x = (1 + weight) * residual
        x = weight * residual
        #print(x.size())

        ## block 1
        x = self.cnn1(x)
        residual = x
        x = self.cnn3(self.re2(self.bn2(self.cnn2(self.re1(self.bn1(x))))))
        x += residual
        x = self.cnn4(self.mp1(x))
        ##print(x.size())
        ## block 2
        residual = x
        x = self.cnn6(self.re4(self.bn4(self.cnn5(self.re3(self.bn3(x))))))
        x += residual
        x = self.cnn7(self.mp2(x))
        ##print(x.size())
        ## block 3
        residual = x
        x = self.cnn9(self.re6(self.bn6(self.cnn8(self.re5(self.bn5(x))))))
        x += residual
        x = self.cnn10(self.mp3(x))
        ##print(x.size())
        ## block 4
        residual = x
        x = self.cnn12(self.re13(self.bn13(self.cnn11(self.re12(self.bn12(x))))))
        x += residual
        x = self.cnn13(self.mp4(x))
        ##print(x.size())
        ## block 5
        residual = x
        x = self.cnn15(self.re15(self.bn15(self.cnn14(self.re14(self.bn14(x))))))
        x += residual
        x = self.cnn16(self.mp5(x))
        ##print(x.size())
        ### classifier
        x = x.view(-1, self.flat_feats)
        x = self.ln1(x)
        residual = x
        x = self.ln3(self.re8(self.bn8(self.ln2(self.re7(self.bn7(x))))))
        x += residual
        ###
        residual = x
        x = self.ln5(self.re10(self.bn10(self.ln4(self.re9(self.bn9(x))))))
        x += residual
        ###
        out = self.sigmoid(self.ln6(self.re11(self.bn11(x))))
        ##print(out.size()) 

        return out, weight

