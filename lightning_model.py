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

        self.conv1 = nn.Sequential(
            nn.Conv2d(    1,   ndf, 3, 1, 1), nn.ReLU6(True), nn.BatchNorm2d(ndf)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(  ndf, 2*ndf, 3, 1, 1), nn.ReLU6(True), nn.BatchNorm2d(2*ndf)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2*ndf, 4*ndf, 3, 1, 1), nn.ReLU6(True), nn.BatchNorm2d(4*ndf)
        )
        self.fc = nn.Sequential(
            nn.Linear(8*8*4*ndf, n_classes),
        )

    def forward(self, x):
        w, h = x.size()[2:]
        cw = (8 / w)**(1/3)
        ch = (8 / h)**(1/3)

        x = self.conv1(x)
        x = F.adaptive_max_pool2d(x, (floor(w * cw), floor(h * ch)))
        x = self.conv2(x)
        x = F.adaptive_max_pool2d(x, (floor(w * cw * cw), floor(h * ch * ch)))
        x = self.conv3(x)
        x = F.adaptive_max_pool2d(x, (floor(w * cw * cw * cw), floor(h * ch * ch * ch)))
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
    summary(model, (1, 64, 64), 128, 'cpu')