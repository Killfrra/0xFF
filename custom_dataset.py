from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms.functional import to_tensor
import random as rnd
import torch
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler
import numpy as np

class CustomSampler(Sampler):

    def __init__(self, file, mem, shuffle=False, drop_last=False):
        #print('sampler init called')
        self.file = file
        self.shuffle = shuffle
        if shuffle:
            self.sampler = RandomSampler
        else:
            self.sampler = SequentialSampler
        self.mem = mem
        self.drop_last = drop_last
        #self.__iter__()

    def __iter__(self):
        #print('iter called')
        batches = []
        for key in self.file:
            data_grp = self.file[key]['data']
            batch_size = min(self.mem // np.prod(data_grp[0].shape).item(), data_grp.shape[0])
            group_batches = list(BatchSampler(self.sampler(data_grp), batch_size, self.drop_last))
            batches.extend(zip([key] * len(group_batches), group_batches))
        if self.shuffle:
            rnd.shuffle(batches)
        self.length = len(batches)
        return iter(batches)

    def __len__(self):
        return self.length


class CustomBatchSampler(Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        return ([ (group, index) for index in indexes ] for group, indexes in self.sampler)

    def __len__(self):
        return len(self.sampler)


def normalize(x):
    x_min = x.min()
    return (x - x_min) / (x.max() - x_min)


class CustomDataset(Dataset):

    def __init__(self, file, sampler, mean=0, std=255, convert_to_tensor=True):
        #print('init called')
        self.file = file
        self.num_samples = sum([ len(indexes) for _, indexes in list(sampler) ])
        self.convert_to_tensor = convert_to_tensor
        self.mean = mean
        self.std = std

    def __getitem__(self, idx):
        #print('getitem', idx)
        group, index = idx
        grp = self.file[group]
        data_grp = grp['data']
        labels_grp = grp['labels']
        data = data_grp[index] 
        label = labels_grp[index]
        if self.convert_to_tensor:
            #data = torch.from_numpy(data).float().sub_(self.mean).div_(self.std) #.div_(255).unsqueeze_(1)
            data = to_tensor(data)
        else:
            data = data.astype(np.float32)
            np.subtract(data, self.mean, out=data)
            np.true_divide(data, self.std, out=data)
            
        return (data, label)

    def __len__(self):
        #print('len called')
        return self.num_samples

if __name__ == '__main__':
    import numpy as np
    """
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
    """
    import h5py
    #mean = 0.29518359668870897
    #var = 10.006859636120366
    #std = 3.163362077935494
    with h5py.File('datasets/train.hdf5', 'r') as file:
        sampler = CustomSampler(file, mem=127*127*64, shuffle=False, drop_last=True)
        batch_sampler = CustomBatchSampler(sampler)
        dataset = CustomDataset(file, sampler)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
        i = 0
        for data, labels in dataloader:
            if i == 1: break
            i += 1
            print(data, labels)
        #print(dataloader.dataset.num_samples, len(dataloader.dataset))
