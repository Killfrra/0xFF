from lightning_model import Net
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import torch

hparams = argparse.Namespace()
hparams.batch_size = 1024
hparams.learning_rate = 1

model = Net(hparams) #.load_from_checkpoint('saves/epoch=27_v2.ckpt')
#checkpoint = torch.load('mnist/saves/backup/anyx63_81_acc/mnist_cnn_epoch_12.pt')
#model.load_state_dict(checkpoint['model'], strict=False)
checkpoint_callback = ModelCheckpoint('saves/main', monitor='val_acc', save_top_k=10)

kwargs = {'num_workers': 6, 'pin_memory': True}

seq = iaa.Sequential([
    iaa.Sometimes(0.5, [
        iaa.PerspectiveTransform(keep_size=False, cval=ia.ALL, mode=ia.ALL),
        #iaa.PiecewiseAffine(cval=ia.ALL, mode=ia.ALL),
        #iaa.ElasticTransformation(alpha=(0, 0.25), sigma=(0, 0.05)),
        iaa.Affine(scale={'x': (1.0, 1.1)}, rotate=(-10, 10), shear=(-15, 15), order=ia.ALL, cval=ia.ALL, mode=ia.ALL, fit_output=True)
    ]),
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0, 2.0)),
        iaa.MotionBlur(k=3),
        #iaa.imgcorruptlike.GlassBlur(severity=1),
        #iaa.imgcorruptlike.DefocusBlur(severity=1),
        #iaa.imgcorruptlike.Pixelate(severity=1)
    ]),
    iaa.OneOf([
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255)),
        #iaa.imgcorruptlike.ShotNoise(severity=1)
    ]),
    iaa.JpegCompression(compression=(0, 75)),
    iaa.Crop(percent=0.01),
    iaa.Resize({"height": 63, "width": "keep-aspect-ratio"}, interpolation=ia.ALL),
    iaa.CropToFixedSize(63, 252),
    iaa.PadToFixedSize(63, 63, pad_mode=ia.ALL, pad_cval=(0, 255))
])

def unnamed(image):
    image = np.array(image)
    return seq(image=image)

transform = transforms.Compose([
    transforms.Grayscale(),
    #transforms.Lambda(lambda img: transforms.functional.resize(img, 63) if min(img.size[0], img.size[1]) < 63 else img),
    #transforms.Resize(63),
    #transforms.RandomCrop((63, 252), pad_if_needed=True, padding_mode='reflect'),
    transforms.Lambda(unnamed),
    transforms.ToTensor()
])

train_loader = DataLoader(
    datasets.ImageFolder('ram/mini_ru_train', transform),
    batch_size=hparams.batch_size, shuffle=True, **kwargs
)
val_loader   = DataLoader(
    datasets.ImageFolder('ram/mini_ru_test', transform),
    batch_size=hparams.batch_size, shuffle=False, **kwargs
)

trainer = Trainer(
    gpus=1, #accumulate_grad_batches=32,
    checkpoint_callback=checkpoint_callback,
    #auto_lr_find=True,
    resume_from_checkpoint='saves/main/epoch=45.ckpt'
)
trainer.fit(model, train_loader, val_loader)
