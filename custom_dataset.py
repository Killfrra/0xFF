from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler
from torchvision.transforms.functional import to_tensor, to_pil_image
import random as rnd
import torch
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler

class CustomSampler(Sampler):

    def __init__(self, file, batch_size=1, shuffle=False, drop_last=False):
        #print('sampler init called')
        self.file = file
        self.shuffle = shuffle
        if shuffle:
            self.sampler = RandomSampler
        else:
            self.sampler = SequentialSampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.__iter__()

    def __iter__(self):
        #print('iter called')
        batches = []
        for key in self.file:
            group_batches = list(BatchSampler(self.sampler(self.file[key]['data']), self.batch_size, self.drop_last))
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
        self.length = len(sampler)

    def __getitem__(self, idx):
        #print('getitem', idx)
        group, indexes = idx
        grp = self.file[group]
        data = grp['data'][indexes]
        labels = grp['labels'][indexes]
        return (data, labels)

    def __len__(self):
        #print('len called')
        return self.length

if __name__ == '__main__':
    import numpy as np
    file = {
        '127': {
            'data': np.array([0, 1, 2]),
            'labels': np.array(['a', 'b', 'c'])
        },
        '254': {
            'data': np.array([3, 4, 5, 6]),
            'labels': np.array(['d', 'e', 'f', 'g'])
        }
    }
    sampler = CustomSampler(file, batch_size=2)
    dataset = CustomDataset(file, sampler)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    for data, labels in dataloader:
        print(data, labels)