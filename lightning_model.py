import torch
from torch import nn
from torch.nn import init
from pytorch_lightning import LightningModule

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, bias=False)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand_planes // 2, kernel_size=1, bias=False)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(expand_planes // 2)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand_planes // 2, kernel_size=3, padding=1, bias=False)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(expand_planes // 2)

    def forward(self, x):
        x = self.bn1(self.squeeze_activation(self.squeeze(x)))
        return torch.cat([
            self.bn2(self.expand1x1_activation(self.expand1x1(x))),
            self.bn3(self.expand3x3_activation(self.expand3x3(x)))
        ], 1)

class SqueezeNet(LightningModule):

    def __init__(self, num_classes):
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 128),
            Fire(128, 16, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 256),
            Fire(256, 32, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 384),
            Fire(384, 48, 384),
            Fire(384, 64, 512),
            Fire(512, 64, 512),
        )

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            #nn.Dropout(p=0.5),
            nn.Identity(),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters())

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)
        return { 'loss': loss }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([ x['loss'] for x in outputs ]).mean()
        logs = { 'train_loss': avg_loss }
        return { 'log': logs }

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = nn.functional.cross_entropy(outputs, labels)
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == labels.data).item()
        samples = inputs.size(0)
        return { 'loss': loss, 'corrects': corrects, 'samples': samples }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([ x['loss'] for x in outputs ]).mean()
        avg_acc = sum([ x['corrects'] for x in outputs ]) / sum([ x['samples'] for x in outputs ])
        logs = { 'val_loss': avg_loss, 'val_acc': avg_acc }
        return { 'val_loss': avg_loss, 'val_acc': avg_acc, 'log': logs }
