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
        for key in file:
            data_grp = file[key]['data']
            key_len = data_grp.shape[0]
            batch_size = mem // np.prod(data_grp[0].shape).item()
            if batch_size == 0:
                continue # недостаточно памяти для того, чтобы вместить хотя бы один элемент

            global_batch_size = batch_size * num_replicas
            global_batch_count = key_len // global_batch_size
            for i in range(global_batch_count):
                batch_start = i * global_batch_size + rank * batch_size
                batch_end = batch_start + batch_size
                self.slices.append((key, slice(batch_start, batch_end), 0))
            
            self.num_batches += global_batch_count

            remaining_samples = key_len - global_batch_count * global_batch_size
            if not drop_last and remaining_samples > 0:
                # 1 < remaining_batch_size < batch_size
                # remaining_batch_size * num_repicas >= key_len
                remaining_batch_size = min(max(remaining_samples // num_replicas + (remaining_samples % num_replicas > 0), 1), batch_size)
                batch_start = global_batch_count * global_batch_size + rank * remaining_batch_size
                batch_end = batch_start + remaining_batch_size
                
                if batch_start < key_len and batch_end < key_len:
                    self.slices.append((key, slice(batch_start, batch_end), 0))
                elif batch_start >= key_len:
                    self.slices.append((key, slice(0), remaining_batch_size))
                elif batch_end >= key_len:
                    self.slices.append((key, slice(batch_start, key_len), batch_end - key_len))
                
                self.num_batches += 1

            if drop_last:
                key_len = global_batch_count * global_batch_size
            self.indices[key] = range(key_len)
            self.num_samples += key_len        

    def __iter__(self):
        g = torch.random.manual_seed(self.epoch)
        for key, value in self.indices.items():
            self.indices[key] = torch.randperm(len(value), generator=g).tolist()
        batch_indices = torch.randperm(self.num_batches, generator=g).tolist()
        for batch in batch_indices:
            key, _slice, additional_count = self.slices[batch]
            additional = torch.randint(0, len(self.indices[key]), (additional_count, ), generator=g).tolist()
            yield [ (key, index) for index in self.indices[key][_slice] ] + [ (key, index) for index in additional ]

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
    
    num_replicas = 4
    for rank in range(num_replicas):
        print('rank', rank)
        batch_sampler = CustomBatchSampler(file, 4, num_replicas, rank)
        #print(batch_sampler.slices)
        for batch in batch_sampler:
            print(batch)