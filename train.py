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

from model import SqueezeNet

# Number of classes in the dataset
num_classes = 42

# Number of epochs to train for 
num_epochs = 32

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


# Initialize the model for this run
model = SqueezeNet(num_classes)
state_dict = torch.load('squeezenet_72acc')
del state_dict['classifier.1.weight']
del state_dict['classifier.1.bias']
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
