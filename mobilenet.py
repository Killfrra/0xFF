import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
from torchsummary import summary
from pytorch_lightning import LightningModule

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(LightningModule):

    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    


def _squeezenet(pretrained, progress, **kwargs):
    model = SqueezeNet(**kwargs)
    if pretrained:
        url = 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth'
        state_dict = load_state_dict_from_url(url, progress=progress)
        model.load_state_dict(state_dict)
    return model

model = SqueezeNet(42)

summary(model, (3, 31, 31), -1, 'cpu')

#print(model)
"""
conv(conv(conv(
    conv(conv(conv(
        conv(conv(conv(
            conv(conv(conv(
                conv(
                    conv(conv(conv(
                        conv(conv(conv(
                            conv(
                                conv(conv(conv(
                                    conv(conv(conv(
                                        conv(
                                            conv(x,3,2)
                                        ,3,2)
                                    ,1,1),1,1),3,1,1)
                                ,1,1),1,1),3,1,1)
                            ,3,2)
                        ,1,1),1,1),3,1,1)
                    ,1,1),1,1),3,1,1)
                ,3,2)
            ,1,1),1,1),3,1,1)
        ,1,1),1,1),3,1,1)
    ,1,1),1,1),3,1,1)
,1,1),1,1),3,1,1) = 1
"""