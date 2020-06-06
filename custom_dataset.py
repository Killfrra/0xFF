from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image
import h5py
import torch

class CustomDataset(Dataset):

    def __init__(self, filepath='datasets.hdf5', group='train'):
        self.file = h5py.File(filepath, 'r')
        self.class_num = self.file.attrs['class_num']
        self.group = self.file[group]
        self.length = len(self.group['data'])
        self.current = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.group['data'][idx]
        label = self.group['labels'][idx]
        labels = torch.zeros(self.class_num) # int64?
        labels[label] = 1

        return (to_tensor(data), labels)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.length:
            item = self.__getitem__(self.current)
            self.current += 1
            return item
        raise StopIteration

    def close(self):
        self.file.close()

if __name__ == '__main__':
    dataset = CustomDataset()
    for i, (data, labels) in enumerate(dataset):
        print(i)
        #image = to_pil_image(data)
        #print(labels.argmax(dim=0))
        #savedir = 'datasets/test'
        #image.save('%s/%d.tiff' % (savedir, i))
    dataset.close()