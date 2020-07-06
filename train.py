import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adadelta
import time
#import copy
import h5py
from model import SqueezeNet
from custom_dataset import CustomBatchSampler, CustomDataset
from torch.utils.data import DataLoader

# Number of classes in the dataset
num_classes = 47

# Number of epochs to train for 
num_epochs = 1

mem = 127 * 127 * 128

def print_stat(phase, running_loss, running_corrects, num_samples):
    epoch_loss = running_loss / num_samples
    epoch_acc = running_corrects / num_samples
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('val' if phase else 'train', epoch_loss, epoch_acc))
    return epoch_loss, epoch_acc

def save_model(model, epoch, best_acc, interruped=False):
    torch.save(model.state_dict(), f'saves/squeezenet_{num_classes}c_ep{epoch}_{round(best_acc * 100)}acc{ "_int" if interruped else "" }')

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []
    
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    try:

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
                num_samples = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

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
                    running_corrects += torch.sum(preds == labels.data).item()
                    num_samples += inputs.size(0)

                _, epoch_acc = print_stat(phase, running_loss, running_corrects.double(), num_samples)

                if phase:
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                    save_model(model, epoch, epoch_acc)
                    val_acc_history.append(epoch_acc)

    except KeyboardInterrupt:
        _, epoch_acc = print_stat(phase, running_loss, running_corrects, num_samples)
        save_model(model, epoch, epoch_acc, interruped=True)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Initialize the model for this run
model = SqueezeNet(num_classes)
state_dict = torch.load('saves/squeezenet_47c_ep1_56acc')
#del state_dict['classifier.1.weight']
#del state_dict['classifier.1.bias']
model.load_state_dict(state_dict, strict=False)

# Print the model we just instantiated
#print(model)

print("Initializing Datasets and Dataloaders...")

with h5py.File('datasets/train.hdf5', 'r') as train_dataset_file, \
     h5py.File('datasets/test.hdf5', 'r')  as val_dataset_file:
    
    batch_samplers = [
        CustomBatchSampler(train_dataset_file, mem, shuffle=True, drop_last=True),
        CustomBatchSampler(val_dataset_file, mem, shuffle=False, drop_last=True)
    ]

    # Create training and validation datasets
    datasets = [
        CustomDataset(train_dataset_file),
        CustomDataset(val_dataset_file)
    ]

    kwargs = {
        #'num_workers': 6,
        'pin_memory': True
    }
    # Create training and validation dataloaders
    dataloaders = [
        DataLoader(datasets[0], batch_sampler=batch_samplers[0], **kwargs),
        DataLoader(datasets[1], batch_sampler=batch_samplers[1], **kwargs)
    ]

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run
    params_to_update = model.parameters()

    # Observe that all parameters are being optimized
    optimizer = Adadelta(params_to_update)

    # Setup the loss fxn
    criterion = CrossEntropyLoss()

    # Train and evaluate
    model, hist = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs)
