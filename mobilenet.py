from torchvision.models import MobileNetV2
from torchsummary import summary
from torch import nn, torch
from torch.nn import functional as F
from pytorch_lightning import LightningModule

# от MobileNetV2 тут только одно название и Residual-блок

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Net(LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()

        self.hparams = hparams

        self.features = nn.Sequential(
            ConvBNReLU(1, 32),
            InvertedResidual(32, 64, 2, 1),
            InvertedResidual(64, 96, 2, 6),
            InvertedResidual(96, 128, 2, 6),
            InvertedResidual(128, 160, 2, 6),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(160, 5)
        )

    def forward(self, x):

        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
    
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)
        return [ optimizer ], [ scheduler ]

#model = Net()
#summary(model, (1, 63, 63), -1, 'cpu')