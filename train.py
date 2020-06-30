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
from model import SqueezeNet
from custom_dataset import CustomSampler, CustomDataset

# Number of classes in the dataset
num_classes = 47

# Number of epochs to train for 
num_epochs = 2

batch_size = 8

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []
    
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in range(2):
            if phase:
                model.eval()    # Set model to evaluate mode
            else:
                model.train()   # Set model to training mode

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
                with torch.set_grad_enabled(bool(1 - phase)):
                    # Get model outputs and calculate loss

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if not phase:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase:
                #if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f'saves/squeezenet_{num_classes}c_ep{epoch}_{round(best_acc.item() * 100)}acc')
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Initialize the model for this run
model = SqueezeNet(num_classes)
state_dict = torch.load('saves/squeezenet_42c_86acc')
del state_dict['classifier.1.weight']
del state_dict['classifier.1.bias']
model.load_state_dict(state_dict, strict=False)

# Print the model we just instantiated
#print(model)

print("Initializing Datasets and Dataloaders...")

with h5py.File('datasets/train.hdf5', 'r') as train_dataset_file, \
     h5py.File('datasets/test.hdf5', 'r')  as val_dataset_file:
    
    samplers = [
        CustomSampler(train_dataset_file, batch_size, shuffle=True, drop_last=True),
        CustomSampler(train_dataset_file, batch_size, shuffle=False, drop_last=True)
    ]
    
    # Create training and validation datasets
    datasets = [
        CustomDataset(train_dataset_file, samplers[0]),
        CustomDataset(val_dataset_file, samplers[1])
    ]

    kwargs = {
    #    'num_workers': 6,
        'pin_memory': True
    }
    # Create training and validation dataloaders
    dataloaders = [
        torch.utils.data.DataLoader(datasets[0], batch_size=None, sampler=samplers[0], **kwargs),
        torch.utils.data.DataLoader(datasets[1], batch_size=None, sampler=samplers[1], **kwargs)
    ]

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run
    params_to_update = model.parameters()

    # Observe that all parameters are being optimized
    optimizer = optim.Adadelta(params_to_update)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs)
