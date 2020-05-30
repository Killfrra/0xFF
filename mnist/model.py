import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self, enable_decoder=False):
        super(Net, self).__init__()
        self.enable_decoder = enable_decoder

        ndf = 16
        n_classes = 5

        self.classifier = nn.Sequential(
            
            nn.Conv2d( 1, ndf, 3, 2),
            nn.ReLU(True),
            #nn.Identity(),
            nn.BatchNorm2d(ndf),

            nn.Conv2d(ndf, 2*ndf, 3, 2),
            nn.ReLU(True),
            #nn.Identity(),
            nn.BatchNorm2d(2*ndf),
            
            nn.Conv2d(2*ndf, 4*ndf, 3, 2),
            nn.ReLU(True),
            #nn.Identity(),
            nn.BatchNorm2d(4*ndf),

            nn.Conv2d(4*ndf, 8*ndf, 3, 2),
            nn.ReLU(True),
            #nn.Identity(),
            nn.BatchNorm2d(8*ndf),

            nn.Conv2d(8*ndf, 16*ndf, 3, 2),
            nn.ReLU(True),
            #nn.Identity(),
            #nn.BatchNorm2d(16*ndf),

            #nn.AdaptiveAvgPool2d(1)
            nn.AdaptiveMaxPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(16*ndf, n_classes)
        )

    def forward(self, x):

        x = self.classifier(x)
        #print(x.size())
        x = torch.flatten(x, 1)
        #x, _ = x.max(dim=2)
        #x, _ = x.max(dim=2)
        x = self.fc(x)
    
        return x

if __name__ == '__main__':
    model = Net(False)
    summary(model, (1, 63, 63), 128, 'cpu')