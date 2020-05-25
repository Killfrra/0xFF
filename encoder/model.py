import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, ngpu, enable_decoder=True):
        super(Autoencoder, self).__init__()
        self.ngpu = ngpu
        self.enable_decoder = enable_decoder

        def down_conv_block(n_input, n_output, k_size=3, stride=1, padding=1):
            return [
                nn.Conv2d(n_input, n_output, k_size, stride, padding, padding_mode='reflect'),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(n_output),
                nn.MaxPool2d(2)
            ]
        
        def up_conv_block(n_input, n_output, k_size=3, stride=1, padding=1):
            return [
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(n_input, n_output, k_size, stride=stride, padding=padding, padding_mode='reflect'),
                nn.LeakyReLU(0.2, True), # negative_slope?
                nn.BatchNorm2d(n_output),
            ]
        
        self.encoder = nn.Sequential(
            # 96
            * down_conv_block(1, 32),
            # 48
            * down_conv_block(32, 64)
            # 24
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            # 24
            * up_conv_block(64, 64),
            # 48
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
            # 96
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.enable_decoder:
            x = self.decoder(x)
        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = Autoencoder(0)
    summary(model, (1, 96, 96), 128, 'cpu')