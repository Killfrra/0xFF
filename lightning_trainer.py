from lightning_model import Net
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

hparams = argparse.Namespace()
hparams.batch_size = 128
hparams.learning_rate = 0.5754399373371567 #1.0964781961431852e-07

model = Net(hparams) #.load_from_checkpoint('mnist/saves/epoch=27_v2.ckpt')
checkpoint_callback = ModelCheckpoint('mnist/saves', monitor='val_acc', save_top_k=10)

kwargs = {'num_workers': 16, 'pin_memory': True}

transform = transforms.Compose([
    transforms.Grayscale(),
    #transforms.Lambda(lambda img: transforms.functional.resize(img, 63) if min(img.size[0], img.size[1]) < 63 else img),
    transforms.Resize(64),
    transforms.RandomCrop(64),
    transforms.ToTensor()
])

train_loader = DataLoader(
    datasets.ImageFolder('mnist/ram/mini_ru_train', transform),
    batch_size=hparams.batch_size, shuffle=True, **kwargs
)
val_loader   = DataLoader(
    datasets.ImageFolder('mnist/ram/mini_ru_test', transform),
    batch_size=hparams.batch_size, shuffle=False, **kwargs
)

trainer = Trainer(
    gpus=1, accumulate_grad_batches=2048 if hparams.batch_size == 1 else 1,
    checkpoint_callback=checkpoint_callback,
    auto_lr_find=True
    #resume_from_checkpoint='mnist/saves/epoch=31_v1.ckpt'
)
trainer.fit(model, train_loader, val_loader)