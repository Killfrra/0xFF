import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, ngpu):
        super(Autoencoder, self).__init__()
        self.ngpu = ngpu
        self.enable_decoder = True

        def down_conv_block(n_input, n_output, k_size=3, stride=1, padding=1):
            return [
                nn.Conv2d(n_input, n_output, k_size, stride, padding),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(n_output),
                nn.MaxPool2d(2)
            ]
        
        def up_conv_block(n_input, n_output, k_size=3, stride=1, padding=1):
            return [
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(n_input, n_output, k_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2, True), # negative_slope?
                nn.BatchNorm2d(n_output),
            ]
        
        self.encoder = nn.Sequential(
            * down_conv_block(1, 32),
            * down_conv_block(32, 64)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            * up_conv_block(64, 64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.enable_decoder:
            x = self.decoder(x)
        return x
