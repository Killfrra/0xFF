from argparse import ArgumentParser
from pytorch_lightning import Trainer
import h5py
from custom_dataset import CustomBatchSampler, CustomDataset
from torch.utils.data import DataLoader
from lightning_model import SqueezeNet as Model
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

class InterruptedCheckpoint(Callback):

    def __init__(self, model_prefix):
        self.model_prefix = model_prefix
    
    @rank_zero_only
    def on_keyboard_interrupt(self, trainer, pl_module):
        trainer.save_checkpoint(f'{self.model_prefix}_ep{trainer.current_epoch}_interrupted.ckpt')
        print('Interrupted. Model saved')

import torch

#if __name__ == '__main__':

parser = ArgumentParser()
parser.add_argument('-g', '--gpus', type=int, default=1)
parser.add_argument('-n', '--num_workers', type=int, default=1)
parser.add_argument('-e', '--epochs', type=int, default=1)
parser.add_argument('-m', '--mem', type=int, default=127*127*256)
parser.add_argument('-f', '--fast', type=bool, default=False)
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
    #model.load_state_dict(torch.load('saves/squeezenet_47c_ep1_56acc'))

    model_prefix = f'saves/squeezenet_{num_classes}c'

    trainer = Trainer(
        checkpoint_callback=ModelCheckpoint(
            filepath=model_prefix + '_ep{epoch}_{val_acc:.2f}acc',
            monitor='val_loss',
            verbose=True,
            save_last=True,
            save_top_k=-1
        ),
        callbacks=[ InterruptedCheckpoint(model_prefix) ],
        gpus=args.gpus,
        fast_dev_run=args.fast,
        max_epochs=args.epochs,
        precision=16,
        resume_from_checkpoint='saves/squeezenet_47c_epepoch=1_val_acc=0.59acc.ckpt',
        profiler=True
    )

    #try:
    trainer.fit(model, train_dataloader, val_dataloader)
    #except KeyError:
    #    print(train_dataloader.batch_sampler.indices)
    #    print(train_dataloader.batch_sampler.batches)
"""
    try:
        trainer.fit(model, train_dataloader, val_dataloader)
    except Exception as e:
        if not trainer.interrupted:
            trainer.save_checkpoint(f'{model_prefix}_ep{trainer.current_epoch}_interrupted.ckpt')
            print('Exception occurred. Model saved')
        raise e
"""