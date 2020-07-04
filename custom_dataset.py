from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms.functional import to_tensor
import random as rnd
import torch
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
import numpy as np

class CustomBatchSampler(Sampler):
    
    def __init__(self, file, mem, num_replicas=1, rank=0, shuffle=True, drop_last=False):
        #self.mem, self.shuffle, self.drop_last = (mem, shuffle, drop_last)
        self.epoch = 0
        self.indices = {}
        self.slices = [] # all batches
        self.num_batches = 0
        self.num_samples = 0
        global_mem = mem * num_replicas
        for key in file:
            data_grp = file[key]['data']
            key_len = data_grp.shape[0]
            global_batch_size = min(global_mem // np.prod(data_grp[0].shape).item(), key_len)
            batch_size = global_batch_size // num_replicas               # сейчас они отбрасываются
            if batch_size == 0:
                continue
            modulo = key_len % global_batch_size
            key_batches = key_len // global_batch_size + (not drop_last and modulo > 0)
            self.num_batches += key_batches
            if drop_last:
                key_len -= modulo
            #TODO: что делать, если остаётся меньше samples, чем num_replica?
            self.num_samples += (key_len // num_replicas) * num_replicas # сейчас они отбрасываются
            self.indices[key] = range(key_len)
            for i in range(key_batches):
                batch_start = i * global_batch_size + batch_size * rank
                batch_end = min(batch_start + batch_size, key_len)
                self.slices.append((key, slice(batch_start, batch_end)))
        

    def __iter__(self):
        g = torch.random.manual_seed(self.epoch)
        for key, value in self.indices.items():
            self.indices[key] = torch.randperm(len(value), generator=g).tolist()
        batch_indices = torch.randperm(self.num_batches, generator=g).tolist()
        for batch in batch_indices:
            key, _slice = self.slices[batch]
            yield [ (key, index) for index in self.indices[key][_slice] ]

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch


def normalize(x):
    x_min = x.min()
    return (x - x_min) / (x.max() - x_min)


class CustomDataset(Dataset):

    def __init__(self, file, sampler, mean=0, std=255, convert_to_tensor=True):
        self.file = file
        self.num_samples = sampler.num_samples
        self.convert_to_tensor = convert_to_tensor
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        group, index = idx
        grp = self.file[group]
        data = grp['data'][index] 
        label = grp['labels'][index]
        if self.convert_to_tensor:
            #data = torch.from_numpy(data).float().sub_(self.mean).div_(self.std) #.div_(255).unsqueeze_(1)
            data = to_tensor(data)
        else:
            data = data.astype(np.float32)
            np.subtract(data, self.mean, out=data)
            np.true_divide(data, self.std, out=data)
            
        return (data, label)

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    file = {
        '127': {
            'data': np.array([[[0]], [[1]], [[2]]], dtype=np.uint8),
            'labels': np.array([7, 8, 9], dtype='i8')
        },
        '254': {
            'data': np.array([[[3, 0]], [[4, 0]], [[5, 0]], [[6, 0]]], dtype=np.uint8),
            'labels': np.array([10, 11, 12, 13], dtype='i8')
        }
    }
    
    batch_sampler = CustomBatchSampler(file, 1, num_replicas=3, rank=0)
    print(batch_sampler.num_samples)
    for batch in batch_sampler:
        print(batch)

    exit()