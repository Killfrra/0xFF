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
            batch_size = self.mem // np.prod(data_grp[0].shape).item()
            group_batches = list(BatchSampler(self.sampler(data_grp), batch_size, self.drop_last))
            batches.extend(zip([key] * len(group_batches), group_batches))
        rnd.shuffle(batches)
        self.length = len(batches)
        return iter(batches)

    def __len__(self):
        return self.length


class CustomDataset(Dataset):

    def __init__(self, file, sampler):
        #print('init called')
        self.file = file
        self.num_samples = sum([ len(indexes) for _, indexes in list(sampler) ])
        self.length = len(sampler)

    def __getitem__(self, idx):
        #print('getitem', idx)
        group, indexes = idx
        grp = self.file[group]
        data_grp = grp['data']
        labels_grp = grp['labels']
        shape = data_grp[0].shape
        batch_size = len(indexes)
        data = np.empty((batch_size, shape[0], shape[1]), dtype=np.uint8)
        labels = np.empty((batch_size,), dtype='i8')
        for i, j in enumerate(indexes):
            data[i] = data_grp[j]
            labels[i] = labels_grp[j]
        data = torch.from_numpy(data).float().div(255).unsqueeze_(1)
        return (data, labels)

    def __len__(self):
        #print('len called')
        return self.length

if __name__ == '__main__':
    import numpy as np
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
    import h5py
    with h5py.File('ram/train.hdf5', 'r') as file:
        sampler = CustomSampler(file, mem=127*127*64, shuffle=True, drop_last=True)
        dataset = CustomDataset(file, sampler)
        dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
        #for data, labels in dataloader:
        #    print(data.size(), labels)
        print(dataloader.dataset.num_samples, len(dataloader.dataset))
