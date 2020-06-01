from torch import torch, nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule

class Autoencoder(LightningModule):
    def __init__(self, hparams, enable_decoder=True):
        super(Autoencoder, self).__init__()
    
        self.enable_decoder = enable_decoder
        self.hparams = hparams

        def down_conv_block(n_input, n_output, k_size=3, stride=2, padding=1):
            return [
                nn.Conv2d(n_input, n_output, k_size, stride, padding, padding_mode='reflect', bias=False),
                nn.ReLU6(True),
                nn.BatchNorm2d(n_output),
                #nn.MaxPool2d(2)
            ]
        
        def up_conv_block(n_input, n_output, k_size=3, stride=1, padding=1):
            return [
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.Conv2d(n_input, n_output, k_size, stride=stride, padding=padding, padding_mode='reflect', bias=False),
                nn.ReLU6(True),
                nn.BatchNorm2d(n_output),
            ]
        
        self.encoder = nn.Sequential(
            # 64
            * down_conv_block(1, 16, stride=1),
            * down_conv_block(16, 16),
            # 32
            * down_conv_block(16, 32, stride=1),
            * down_conv_block(32, 32),
            # 16
            * down_conv_block(32, 64, stride=1),
            * down_conv_block(64, 64),
            # 8
        )
        self.decoder = nn.Sequential(
            * up_conv_block(64, 64),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU6(True), nn.BatchNorm2d(32),
            
            * up_conv_block(32, 32),
            nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU6(True), nn.BatchNorm2d(16),

            * up_conv_block(16, 16),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
            # 64
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.enable_decoder:
            x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.mse_loss(output, target)

        return { 'loss': loss }

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        tensorboard_logs = { 'avg_train_loss': avg_loss }
        return { 'train_loss': avg_loss, 'log': tensorboard_logs }

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.mse_loss(output, target)

        return { 'val_loss': loss }

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = { 'avg_val_loss': avg_loss }
        return { 'val_loss': avg_loss, 'log': tensorboard_logs }

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
        return [ optimizer ], [ scheduler ]

import argparse
from custom_dataset import CustomDataset

hparams = argparse.Namespace()
hparams.batch_size = 128
hparams.learning_rate = 1

def train():

    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader
    
    model = Autoencoder(hparams)    #.load_from_checkpoint('mnist/saves/epoch=27_v2.ckpt')
    checkpoint_callback = ModelCheckpoint('saves/autoencoder', save_top_k=10)

    kwargs = {'num_workers': 16, 'pin_memory': True}

    train_loader = DataLoader(
        CustomDataset('ram/mini_ru_train/no_label'),
        batch_size=hparams.batch_size, shuffle=True, **kwargs
    )
    val_loader   = DataLoader(
        CustomDataset('ram/mini_ru_test/no_label'),
        batch_size=hparams.batch_size, shuffle=False, **kwargs
    )

    trainer = Trainer(
        gpus=1, #accumulate_grad_batches=32,
        checkpoint_callback=checkpoint_callback,
        #auto_lr_find=True
        #resume_from_checkpoint='mnist/saves/epoch=31_v1.ckpt'
    )
    trainer.fit(model, train_loader, val_loader)

def main():
    from torchsummary import summary
    model = Autoencoder(None)
    summary(model, (1, 64, 64), 128, 'cpu')

def eval():
    from torchvision.transforms.functional import to_pil_image
    model = Autoencoder.load_from_checkpoint('saves/autoencoder/epoch=33.ckpt')
    model.eval()
    dataset = CustomDataset('ram/mini_ru_test/no_label')
    for i, (data, _) in enumerate(dataset):
        data.unsqueeze_(1)
        output = model(data)
        output.squeeze_(1)
        image = to_pil_image(output)
        savedir = 'ram/results'
        image.save('%s/%d.tiff' % (savedir, i))

import sys
if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'train':
            train()
        elif sys.argv[1] == 'eval':
            eval()
        else:
            main()
    else:
        main()
    