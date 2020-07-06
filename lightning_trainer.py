from argparse import ArgumentParser
from pytorch_lightning import Trainer
import h5py
from custom_dataset import CustomBatchSampler, CustomDataset
from torch.utils.data import DataLoader
from lightning_model import SqueezeNet as Model
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

#if __name__ == '__main__':

parser = ArgumentParser()
parser.add_argument('-g', '--gpus', type=int, default=1)
parser.add_argument('-n', '--num_workers', type=int, default=1)
parser.add_argument('-e', '--epochs', type=int, default=1)
parser.add_argument('-m', '--mem', type=int, default=127*127*128)
args = parser.parse_args()

#if self.use_tpu:
#    sampler_kwargs = dict(num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
#else:
sampler_kwargs = dict(num_replicas=1, rank=0)
dset_kwargs = dict(
    num_workers=args.num_workers,
    pin_memory=True
)

def get_dataloader(dataset_file):
    batch_sampler = CustomBatchSampler(dataset_file, args.mem, **sampler_kwargs)
    dataset = CustomDataset(dataset_file)
    return DataLoader(dataset, batch_sampler=batch_sampler, **dset_kwargs)

with h5py.File('datasets/train.hdf5', 'r') as train_dataset_file, \
     h5py.File('datasets/test.hdf5', 'r') as val_dataset_file:

    train_dataloader = get_dataloader(train_dataset_file)
    val_dataloader = get_dataloader(val_dataset_file)

    num_classes = train_dataset_file.attrs['class_num']

    model = Model(num_classes)
    model.load_state_dict(torch.load('saves/squeezenet_47c_ep1_56acc'))

    model_prefix = f'saves/squeezenet_{num_classes}c'
    checkpoint_callback = ModelCheckpoint(
        filepath=model_prefix + '_ep{epoch}_{val_acc:.2f}acc.ckpt',
        monitor='val_loss',
        verbose=True,
        save_last=True,
        save_top_k=-1
    )

    trainer = Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        #precision=16,
        #resume_from_checkpoint='saves/squeezenet_47c_ep0_interrupted.ckpt'
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    
    #TODO:
    #if trainer.interrupted:
    #    trainer.save_checkpoint(f'{model_prefix}_ep{trainer.current_epoch}_interrupted.ckpt')
