import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self, enable_decoder=False):
        super(Net, self).__init__()
        self.enable_decoder = enable_decoder

        ndf = 16
        self.encoder = nn.Sequential(
            nn.Conv2d( 1, ndf, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2),
            nn.Conv2d(ndf, 2*ndf, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(2*ndf),
            nn.MaxPool2d(2),
        )

        if enable_decoder:

            self.decoder = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(128, 64, 3, 1, 1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(64,  1, 3, 1, 1, padding_mode='reflect'),
                nn.Sigmoid()
            )

        else:

            self.classifier = nn.Sequential(
                nn.Conv2d(2*ndf, 4*ndf, 3, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(4*ndf),
                nn.Conv2d(4*ndf, 4*ndf, 3, 1),
                nn.ReLU(True),
                nn.BatchNorm2d(4*ndf)
            )

            self.fc = nn.Sequential(
                nn.Linear(1*4*ndf, 5)
            )

    def forward(self, x):

        x = self.encoder(x)
        if self.enable_decoder:
            x = self.decoder(x)
        else:
            x = self.classifier(x)
            #x = torch.flatten(x, 1)
            x = x.mean(dim=2)
            x = x.mean(dim=2)
            x = self.fc(x)
    
        return x

if __name__ == '__main__':
    image_side = 56
    model = Net(False)
    summary(model, (1, image_side, image_side), 128, 'cpu')