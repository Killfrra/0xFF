import h5py
import numpy as np
from custom_dataset import CustomSampler, CustomDataset
from tqdm import tqdm
from math import sqrt

with h5py.File('datasets/train.hdf5', 'r') as file:
    sampler = CustomSampler(file, 127*127*1024)
    dataset = CustomDataset(file, sampler, std=1, convert_to_tensor=False)
    nimages = 0
    mean = 0.0
    var = 0.0
    for idx in tqdm(list(sampler)):
        batch, _ = dataset[idx]
        nimages += batch.shape[0]
        mean += batch.mean()
        var += batch.var()

mean /= nimages
var /= nimages

print('mean:', mean, 'var:', var, 'std:', sqrt(var))

