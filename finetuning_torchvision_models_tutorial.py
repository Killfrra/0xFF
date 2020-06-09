# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline


from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import torch.nn.init as init

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
import h5py
import torch

from torch.utils.data import BatchSampler


class CustomDataset(Dataset):

    def __init__(self, filepath='datasets.hdf5', group='train'):
        self.file = h5py.File(filepath, 'r')
        self.class_num = self.file.attrs['class_num']
        self.group = self.file[group]
        self.length = len(self.group['data'])
        self.current = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.group['data'][idx]
        label = self.group['labels'][idx]
        return (to_tensor(data), label.astype('i8'))

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.length:
            item = self.__getitem__(self.current)
            self.current += 1
            return item
        raise StopIteration

    def close(self):
        self.file.close()

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        #self.inplanes = inplanes
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

class SqueezeNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
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
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
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

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
data_dir = "datasets/mini_ru_"

# Number of classes in the dataset
num_classes = 42

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for 
num_epochs = 32

input_size = 63


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f'squeezenet_{round(best_acc.item() * 100)}acc')

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# Initialize the model for this run
model = SqueezeNet(num_classes)
state_dict = torch.load('squeezenet_72acc')
del state_dict['classifier.1.weight']
del state_dict['classifier.1.bias']
#print('\n\n', type(state_dict['classifier.1.weight']), type(state_dict['classifier.1.bias']), '\n\n')
#state_dict['classifier.1.weight'] = nn.functional.pad(state_dict['classifier.1.weight'], (0,0,0,0,0,0,0,10))
#state_dict['classifier.1.bias'] = nn.functional.pad(state_dict['classifier.1.bias'], (0, 10))
#print('\n\n', state_dict['classifier.1.weight'].size(), state_dict['classifier.1.bias'].size(), '\n\n')
model.load_state_dict(state_dict, strict=False)

# Print the model we just instantiated
#print(model)

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
image_datasets = {
    'train': CustomDataset('datasets/miniru2_381.hdf5', 'train'),
    #datasets.ImageFolder(data_dir + 'train', data_transforms['train']),
    'val': CustomDataset('datasets/miniru2_508.hdf5', 'train'),
    #datasets.ImageFolder(data_dir + 'test', data_transforms['val'])
}

kwargs = { 'drop_last':True, 'pin_memory': True } #'num_workers': 6, 'pin_memory': True}
# Create training and validation dataloaders
dataloaders_dict = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True, **kwargs),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32, shuffle=False, **kwargs)
}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Send the model to GPU
model = model.to(device)

# Gather the parameters to be optimized/updated in this run
params_to_update = model.parameters()
print("Params to learn:")
for name,param in model.named_parameters():
    if param.requires_grad == True:
        print("\t",name)

# Observe that all parameters are being optimized
optimizer = optim.Adadelta(params_to_update)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)
