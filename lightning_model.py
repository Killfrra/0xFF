import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from math import floor

class Net(LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()

        self.hparams = hparams

        ndf = 64
        n_classes = 5

        self.classifier = nn.Sequential(
            nn.Conv2d(    1,   ndf, 3, 2), nn.ReLU(True), nn.BatchNorm2d(ndf),
            nn.Conv2d(  ndf, 2*ndf, 3, 2), nn.ReLU(True), nn.BatchNorm2d(2*ndf),
            nn.Conv2d(2*ndf, 4*ndf, 3, 2), nn.ReLU(True), nn.BatchNorm2d(4*ndf),
            nn.AdaptiveMaxPool2d(7)
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*4*ndf, n_classes)
        )

    def forward(self, x):
        x = self.classifier(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)

        return { 'loss': loss }

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        num_correct = pred.eq(target.view_as(pred)).sum()
        return { 'val_loss': loss, 'num_correct': num_correct }

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['num_correct'] for x in outputs]).sum().float()
        avg_accuracy /= (len(outputs) * self.hparams.batch_size)

        tensorboard_logs = {'avg_val_loss': avg_loss, 'avg_val_accuracy': avg_accuracy}
        return {'val_loss': avg_loss, 'val_acc': avg_accuracy, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
        return [ optimizer ], [ scheduler ]

if __name__ == '__main__':
    from torchsummary import summary
    model = Net(None)
    summary(model, (1, 63, 63), 128, 'cpu')