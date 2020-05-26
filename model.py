import torch
import torch.nn as nn
from encoder.model import Autoencoder

class DeepFont(nn.Module):
    def __init__(self, encoder, ngpu):
        super(DeepFont, self).__init__()
        self.ngpu = ngpu
        self.lru = nn.LeakyReLU(0.2, inplace=True)
        self.encoder = encoder
        
        def conv2d(in_channels=64, out_channels=64):
            return [
                nn.Conv2d(in_channels, out_channels, 3, 1, 0),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(out_channels),
            ]

        self.filters = nn.Sequential(
            * conv2d(64, 32),
            * conv2d(32, 16),
            * conv2d(16, 8),
            #nn.MaxPool2d(2)
            #nn.AdaptiveAvgPool2d((1, 1))
        )
        """
        def fc(in_features, out_features):
            return [
                nn.Linear(in_features, out_features),
                nn.LeakyReLU(0.2, True),
                nn.Dropout()
            ]
        """
        self.classifier = nn.Sequential(
            # * fc(9*9*64, 64),
            # * fc(64, 64),
            nn.Linear(8, 5)
        )

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(x)

        x = self.filters(x)
            
            #x.reshape(x.size(0), -1)
        #x = torch.flatten(x, start_dim=1)
        x = x.mean(dim=2)
        x = x.mean(dim=2)
        
        x = self.classifier(x)
        #x = self.softmax(x)
        
        return x

if __name__ == '__main__':
    from torchsummary import summary
    encoder = Autoencoder(0, enable_decoder=False)
    model = DeepFont(encoder , 0)
    summary(model, (1, 96, 96), 128, 'cpu')