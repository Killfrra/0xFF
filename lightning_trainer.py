from encoder.model import Autoencoder
from lightning_model import DeepFont
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser
import torch
import os
import glob

def get_last_checkpoint(path):
    files = sorted(glob.glob(path + '/checkpoint*.pth'), key=os.path.getmtime, reverse=True)
    return files[0] if len(files) > 0 else None

def main(hparams):
    encoder = Autoencoder(1, False).to(torch.device("cuda:0"))

    output_dir = 'output/autoencoder'
    last_checkpoint = get_last_checkpoint(output_dir)
    checkpoint = torch.load(last_checkpoint)
    encoder.load_state_dict(checkpoint, strict=False)
    encoder.enable_decoder = False
    print('LOADED', last_checkpoint, '!')

    model = DeepFont(encoder, hparams)

    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name='no_dropout',
        #version=5
    )

    trainer = Trainer(logger, gpus=hparams.gpus)

    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    parser = DeepFont.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)