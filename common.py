import os
import torch
import torchvision.transforms as transforms
from PIL.ImageOps import autocontrast

def load_checkpoint(savefile, model, optimizer=None, strict=True):
    epoch = 0
    total_loss = 0
    if os.path.isfile(savefile):
        checkpoint = torch.load(savefile)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'], strict)
        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_loss = checkpoint['loss']
        #model.train()
    return (epoch, total_loss)

def save_checkpoint(savefile, model, optimizer, epoch, total_loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss
    }, savefile)


def transform(image_size):
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.Lambda(autocontrast),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,)),
    ])