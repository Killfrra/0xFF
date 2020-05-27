import torch
import torch.nn as nn
from encoder.model import Autoencoder
from pytorch_lightning import LightningModule, Trainer
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam, SGD
from argparse import ArgumentParser

class DeepFont(LightningModule):
    def __init__(self, encoder, hparams):
        super(DeepFont, self).__init__()

        self.hparams = hparams

        self.data_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        self.lru = nn.LeakyReLU(0.2, inplace=True)
        self.encoder = encoder
        
        def conv2d(in_channels=64, out_channels=64):
            return [
                nn.Conv2d(in_channels, out_channels, 3, 1, 0),
                nn.LeakyReLU(0.2, True),
                nn.BatchNorm2d(out_channels),
            ]

        self.filters = nn.Sequential(
            * conv2d(64, 64),
            * conv2d(64, 64),
            * conv2d(64, 64),
            nn.MaxPool2d(2)
            #nn.AdaptiveAvgPool2d((1, 1))
        )
        
        def fc(in_features, out_features):
            return [
                nn.Linear(in_features, out_features),
                nn.LeakyReLU(0.2, True),
                #nn.Dropout()
            ]
        
        self.classifier = nn.Sequential(
            * fc(9*9*64, 64),
            * fc(64, 64),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        self.encoder.eval()
        with torch.no_grad():
            x = self.encoder(x)

        x = self.filters(x)
            
            #x.reshape(x.size(0), -1)
        x = torch.flatten(x, start_dim=1)
        #x = x.mean(dim=2)
        #x = x.mean(dim=2)
        
        x = self.classifier(x)
        #x = self.softmax(x)
        
        return x

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # Adam(self.parameters(), lr=self.hparams.lr)
        return SGD(self.parameters(), self.hparams.lr, self.hparams.momentum)

    def train_dataloader(self):
        # REQUIRED
        train_dataset = 'datasets/mini_ru_synth_train_preprocessed'
        return DataLoader(datasets.ImageFolder(train_dataset, self.data_transform), batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.workers)

    def val_dataloader(self):
        # OPTIONAL
        eval_dataset  = 'datasets/mini_ru_synth_test_preprocessed'
        return DataLoader(datasets.ImageFolder(eval_dataset, self.data_transform), batch_size=self.hparams.batch_size, num_workers=self.hparams.workers)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        images, labels = batch
        outputs = self.forward(images)
        loss = F.cross_entropy(outputs, labels)

        tensorboard_logs = { 'train_loss': loss }

        return { 'loss': loss, 'log': tensorboard_logs }

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        images, labels = batch
        outputs = self.forward(images)
        return { 'val_loss': F.cross_entropy(outputs, labels) }

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-l', '--lr', default=0.002, type=float)
        parser.add_argument('-m', '--momentum', default=0.9, type=float)
        parser.add_argument('-b', '--batch-size', default=128, type=int)
        parser.add_argument('-w', '--workers', default=6, type=int)

        return parser


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser = DeepFont.add_model_specific_args(parser)

    args = parser.parse_args()

    encoder = Autoencoder(1, False)
    model = DeepFont(encoder, args)
    trainer = Trainer(gpus=1)

    # Run learning rate finder
    lr_finder = trainer.lr_find(model)

    # Results can be found in
    print('results', lr_finder.results)

    # Plot with
    fig = lr_finder.plot(suggest=True, show=True)

    # Pick point based on plot, or get suggestion
    print('suggestion', lr_finder.suggestion())

    """
    from torchsummary import summary
    encoder = Autoencoder(0, enable_decoder=False)
    model = DeepFont(encoder , 0)
    summary(model, (1, 96, 96), 128, 'cpu')
    """