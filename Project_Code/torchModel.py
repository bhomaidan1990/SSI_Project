
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models

# from ..utils import initialize_weights
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes, input_channel):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(input_channel, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear', align_corners=True)], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear', align_corners=True)], 1))
        final = self.final(dec1)
        return F.interpolate(final, x.size()[2:], mode='bilinear', align_corners=True)

# ================================================================
# the following code is for the customed Unet with mobileNet as backbone
# ================================================================

def get_encoder(model):
    featList = list(model.features.children())
    layer_cnt = 0
    concatenate_layer = []
    for i in featList:
        if isinstance(i, models.mobilenet.InvertedResidual):
            stride = i.conv[0][0].stride
            for j in i.conv:
                if isinstance(j, models.mobilenet.ConvBNReLU):
                    if j[0].stride == (2,2):
                        concatenate_layer.append(layer_cnt)
        elif isinstance(i, models.mobilenet.ConvBNReLU):
            if i[0].stride == (2,2):
                concatenate_layer.append(layer_cnt)
        layer_cnt = layer_cnt + 1
            # print(i.conv[0][0].out_channels)
    encoder_list = []
    for i in range(len(concatenate_layer)):
        if i != len(concatenate_layer)-1:
            start = concatenate_layer[i]
            last = concatenate_layer[i+1]
            temp = nn.Sequential(*featList[start:last])
            encoder_list.append(temp)
        else:
            start = concatenate_layer[i]
            temp = nn.Sequential(*featList[start:])
            encoder_list.append(temp)
    return encoder_list
    
class convUpSample(nn.Module):
    def __init__(self, ipc, opc, uks=[2,2], cks=[3,3], cpadding=[1,1],ustride=[2,2], se_module=False):
        # ipc after concatenation
        super(convUpSample, self).__init__()
        # self.conv = nn.Conv2d(in_channels=ipc,out_channels=opc,kernel_size=cks,padding=cpadding)
        self.block = resBlcok(ipc, opc, ks=cks, pd=cpadding, se_module=se_module)
        self.bn1 = nn.BatchNorm2d(opc, track_running_stats=True, momentum=0.1, eps=1e-5)
        self.relu = nn.ReLU6(inplace=True)
        self.upsample = nn.ConvTranspose2d(in_channels=opc, out_channels=opc, kernel_size=uks, stride=[2,2])
    def forward(self, x):
        x = self.block(x)
        x = self.upsample(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class mobileUnet(nn.Module):
    def __init__(self, pretraind_model):
        super(mobileUnet, self).__init__()
        self.encoder_list = nn.Sequential(*get_encoder(pretraind_model))
        # self.bn_encoder = nn.BatchNorm2d(1280)
        self.center = nn.Conv2d(1280,320, [3,3], padding=[1,1])
        self.center_bn = nn.BatchNorm2d(320, track_running_stats=True, momentum=0.1, eps=1e-5)
        self.decoder_list = []
        self.up1 = convUpSample(320, 96)
        self.decoder_list.append(convUpSample(96+96, 48, se_module=True))
        self.decoder_list.append(convUpSample(32+48, 36, se_module=True))
        self.decoder_list.append(convUpSample(24+36, 24, se_module=True))
        self.decoder_list.append(convUpSample(16+24, 32, se_module=True))
        self.decoder_list = nn.Sequential(*self.decoder_list)
        self.final = nn.Conv2d(32, 4, [3,3], padding=[1,1])
        self.bn = nn.BatchNorm2d(4, track_running_stats=True, momentum=0.1, eps=1e-5)
        
    def freeze_encoder(self):
        for param in self.encoder_list.parameters():
            param.requires_grad = False

    def initialize_decoder(self):
        initialize_weights(self.decoder_list)

    def forward(self, x):
        temp_list = []
        for i in self.encoder_list:
            x = i(x)
            temp_list.append(x)
        
        x = self.center(x)
        x = self.center_bn(x)
        x = F.relu6(x)
        x = self.up1(x)
        for i in range(len(self.decoder_list)):
            idx = i+2
            temp = temp_list[-idx]
            x = torch.cat((x,temp), dim=1)
            x = self.decoder_list[i](x)
        x = self.final(x)
        # x = self.bn(x)
        # x = F.relu6(x)
        return x

# ======================================================
# blocks for general usage
# ======================================================
class CBRBlock(nn.Module):
    def __init__(self, ipc, opc, ks=(3,3), pd=(1,1), sd=(1,1), se_module=False):
        super(CBRBlock, self).__init__()
        self.conv = nn.Conv2d(ipc, opc, kernel_size=ks, padding=pd, stride=sd)
        self.bn = nn.BatchNorm2d(opc, track_running_stats=True, momentum=0.1, eps=1e-5)
        self.relu6 = nn.ReLU6(inplace=True)
        self.se_module = se_module
        if self.se_module:
            self.se = SELayer(opc)
        else:
            self.se = None

    
    def forward(self,x):
        x = self.conv(x)
        if self.se is not None:
            x = self.se(x)
        x = self.bn(x)
        x = self.relu6(x)
        return x



class resBlcok(nn.Module):
    def __init__(self, ipc, opc, ks=(3,3), pd=(1,1), sd=(1,1), se_module=False):
        super(resBlcok, self).__init__()
        self.block1 = CBRBlock(ipc, opc, ks=ks, pd=pd, sd=sd, se_module=se_module)
        self.block2 = CBRBlock(opc, opc, se_module=se_module)
        self.block3 = CBRBlock(opc, opc, se_module=se_module)

    def forward(self, x):
        x1 = self.block1(x)
        x = self.block2(x1)
        x = self.block3(x)
        x = F.relu6(x + x1)
        return x


# ======================================================
# attention layer
# ======================================================




# ======================================================
# SE module
# ======================================================
# cite from: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

