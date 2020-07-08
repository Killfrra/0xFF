import h5py
import torch
from torch.utils.data import DataLoader
from custom_dataset import CustomBatchSampler, CustomDataset
from argparse import ArgumentParser
from lightning_model import SqueezeNet as Model
from tqdm import tqdm
import os

parser = ArgumentParser()
parser.add_argument('-n', '--num_workers', type=int, default=1)
parser.add_argument('-m', '--mem', type=int, default=127*127*64)
parser.add_argument('-s', '--startfolder', type=str, default='datasets/google_test/127')
args = parser.parse_args()

classes = sorted(os.listdir(args.startfolder))
num_classes = len(classes)

with h5py.File('datasets/google_train.hdf5', 'r') as dataset_file:
    
    batch_sampler = CustomBatchSampler(dataset_file, args.mem, num_replicas=1, rank=0)
    dataset = CustomDataset(dataset_file)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(num_classes)
    state_dict = torch.load('saves/squeezenet_115c_epepoch=10_val_acc=0.51acc.ckpt')['state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    samples = 0
    corrects = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        preds = torch.argmax(outputs, 1)
        corrects += torch.sum(preds == labels.data).item()
        samples += inputs.size(0)

    print(corrects, '/', samples)

"""
    mistaken = [ [ 0 ] * num_classes ] * num_classes
    accuracy = [ 0 ] * num_classes
    num_samples = 0

    try:
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            labels = labels.cpu().tolist()

            for label, pred in zip(labels, preds):
                if label == pred:
                    accuracy[label] += 1
                else:
                    mistaken[label][pred] += 1

            num_samples += len(labels)

    except Exception as e:
        print(e)

    print(classes)
    print(num_samples)
    print(accuracy)
    print(mistaken)
"""