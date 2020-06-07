import torch
import torch.nn as nn
import torch.nn.init as init
from torch.hub import load_state_dict_from_url
from torchsummary import summary
from pytorch_lightning import LightningModule
import torch.nn.functional as F

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        #self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand_planes // 2, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand_planes // 2, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(LightningModule):

    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
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
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
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

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = F.cross_entropy(output, target)

        return { 'loss': loss }

    def training_epoch_end(self, outputs):
        self.avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()

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

        print(f'Train loss: {self.avg_train_loss} Val loss: {avg_loss} acc: {avg_accuracy}')
        
        return {'val_loss': avg_loss, 'val_acc': avg_accuracy }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
        return [ optimizer ]#, [ scheduler ]


def _squeezenet(pretrained, progress, **kwargs):
    model = SqueezeNet(**kwargs)
    if pretrained:
        url = 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth'
        state_dict = load_state_dict_from_url(url, progress=progress)
        model.load_state_dict(state_dict)
    return model

summary(model, (3, 31, 31), -1, 'cpu')

#print(model)
"""
conv(conv(conv(
    conv(conv(conv(
        conv(conv(conv(
            conv(conv(conv(
                conv(
                    conv(conv(conv(
                        conv(conv(conv(
                            conv(
                                conv(conv(conv(
                                    conv(conv(conv(
                                        conv(
                                            conv(x,3,2)
                                        ,3,2)
                                    ,1,1),1,1),3,1,1)
                                ,1,1),1,1),3,1,1)
                            ,3,2)
                        ,1,1),1,1),3,1,1)
                    ,1,1),1,1),3,1,1)
                ,3,2)
            ,1,1),1,1),3,1,1)
        ,1,1),1,1),3,1,1)
    ,1,1),1,1),3,1,1)
,1,1),1,1),3,1,1) = 1
"""

import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from pytorch_lightning import Trainer

hparams = argparse.Namespace()
hparams.batch_size = 1024
hparams.learning_rate = 1

model = _squeezenet(
    pretrained = True,
    progress = True,
    num_classes = 42
)
checkpoint_callback = ModelCheckpoint(
    'saves',
    monitor='val_acc',
    save_top_k=10
)

kwargs = {'num_workers': 6, 'pin_memory': True}

train_loader = DataLoader(
    CustomDataset('../datasets/datasets_254.hdf5'),
    batch_size=hparams.batch_size, shuffle=True, **kwargs
)
val_loader   = DataLoader(
    CustomDataset('../datasets/datasets_381.hdf5'),
    batch_size=hparams.batch_size, shuffle=False, **kwargs
)

trainer = Trainer(
    gpus=1,
    checkpoint_callback=checkpoint_callback,
    #resume_from_checkpoint='saves/main/epoch=45.ckpt'
)
trainer.fit(model, train_loader, val_loader)