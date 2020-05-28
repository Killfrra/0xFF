import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self, size, ndf, n_encoder_layers, enable_decoder=False, n_classifier_layers = 2, n_fc_layers = 3, n_classes = 5):
        super(Net, self).__init__()
        self.enable_decoder = enable_decoder

        #assert(n_encoder_layers > 1)
        print('ENCODER')
        print('conv&pool', 1, '->' , ndf, size, '->', size // 2)
        encoder_layers = [
            nn.Conv2d(1, ndf, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        ]
        size //= 2
        for _ in range(n_encoder_layers - 1):
            print('conv&pool', ndf, '->' , 2*ndf, size, '->', size // 2)
            encoder_layers += [ 
                nn.Conv2d(ndf, 2*ndf, 3, 1, 1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.MaxPool2d(2)
            ]

            ndf *= 2
            size //= 2

        self.encoder = nn.Sequential(*encoder_layers)

        if enable_decoder:

            print('DECODER')
            decoder_layers = []
            for _ in range(n_encoder_layers - 1):
                print('upsample&conv', ndf, '->' , ndf // 2, size, '->', size * 2)
                decoder_layers += [
                    nn.UpsamplingNearest2d(scale_factor=2),
                    nn.Conv2d(ndf, ndf // 2, 3, 1, 1, padding_mode='reflect'),
                    nn.ReLU(True)
                ]

                ndf //= 2
                size *= 2

            print('upsample&conv', ndf, '->' , 1, size, '->', size * 2)
            decoder_layers += [
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(ndf, 1, 3, 1, 1, padding_mode='reflect')
            ]

            self.decoder = nn.Sequential(*decoder_layers)

        else:

            #assert(n_classifier_layers > 1)
            print('CLASSIFICATOR')
            classifier_layers = []
            for _ in range(n_classifier_layers):
                print('conv&pool', ndf, '->' , ndf, size, '->', size // 2)
                classifier_layers += [ 
                    nn.Conv2d(ndf, ndf, 3, 1, 1),
                    nn.ReLU(True),
                    nn.MaxPool2d(2)
                ]
                size = size // 2

            self.classifier = nn.Sequential(*classifier_layers)

            #assert(n_fc_layers > 2)
            print('fc', round(size), '*', round(size), '*', ndf, '->' , 2*ndf)
            fc_layers = [
                nn.Linear((round(size)**2) * ndf, 2*ndf),
            ]
            ndf *= 2

            for _ in range(n_fc_layers - 2):
                print('fc', ndf, '->' , ndf)
                fc_layers.append(
                    nn.Linear(ndf, ndf)
                )

            print('fc', ndf, '->' , n_classes)
            fc_layers.append(
                nn.Linear(ndf, n_classes)
            )

            self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):

        x = self.encoder(x)
        if self.enable_decoder:
            x = self.decoder(x)
        else:
            x = self.classifier(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
    
        return x

if __name__ == '__main__':
    image_side = 64
    model = Net(image_side, 64, 2, False, 3, 3, 5)
    summary(model, (1, image_side, image_side), 128, 'cpu')