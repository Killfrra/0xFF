from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.transforms.functional import to_tensor
import torch
import numpy as np

class CustomBatchSampler(Sampler):
    """
    Идея в том, чтобы при инициализации поделить места обитания индексов сэмплов на батчи,
    а потом просто перетасовывать индексы и вырезать из них батчи

    Индексы:    | 1 2 3 4 | 5 6 7 8 | 9 . . . |
    Диапазоны:  0    -    4    -    8 - 9 (+ 3) 
    батчей      |  batch  |  batch  |  batch  |
    """
    def __init__(self, file, mem, num_replicas=1, rank=0, shuffle=True, drop_last=False):
        self.epoch = 0
        self.indices = {}
        self.batches = []
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
                self.batches.append((key, slice(batch_start, batch_end), 0))
            
            self.num_batches += global_batch_count

            remaining_samples = key_len - global_batch_count * global_batch_size
            remaining_batch_size = 0
            if remaining_samples > 0 and (not drop_last or remaining_samples >= num_replicas):
                # 1 < remaining_batch_size < batch_size
                # remaining_batch_size * num_repicas >= key_len
                remaining_batch_size = remaining_samples // num_replicas + (not drop_last and remaining_samples % num_replicas > 0)
                batch_start = global_batch_count * global_batch_size + rank * remaining_batch_size
                batch_end = batch_start + remaining_batch_size
                
                if batch_start < key_len and batch_end < key_len:
                    self.batches.append((key, slice(batch_start, batch_end), 0))
                elif batch_start >= key_len:
                    self.batches.append((key, slice(0), remaining_batch_size))
                elif batch_end >= key_len:
                    self.batches.append((key, slice(batch_start, key_len), batch_end - key_len))
                
                self.num_batches += 1

            if drop_last:
                key_len = global_batch_count * global_batch_size + num_replicas * remaining_batch_size
            self.indices[key] = range(key_len)
            self.num_samples += key_len        

    def __iter__(self):
        g = torch.random.manual_seed(self.epoch)
        for key, value in self.indices.items():
            self.indices[key] = torch.randperm(len(value), generator=g).tolist()
        batch_indices = torch.randperm(self.num_batches, generator=g).tolist()
        for batch in batch_indices:
            key, _slice, additional_count = self.batches[batch]
            res = [ (key, index) for index in self.indices[key][_slice] ]
            if additional_count:
                res += [ (key, index) for index in torch.randint(0, len(self.indices[key]), (additional_count, ), generator=g).tolist() ]
            yield res

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

    for mem in range(0, 5):
        for num_replicas in range(1, 5):
            for drop_last in [True, False]:
                indices = {}
                batches = []
                for rank in range(num_replicas):
                    #print('mem:', mem, 'num_replicas:', num_replicas, 'rank:', rank, 'drop_last:', drop_last)
                    batch_sampler = CustomBatchSampler(file, mem, num_replicas, rank, drop_last=drop_last)
                    batches.append(list(batch_sampler))
                    for batch in batches[rank]:
                        #print(batch)
                        for key, value in batch:
                            if not key in indices:
                                indices[key] = [ value ]
                            elif not value in indices[key]:
                                indices[key].append(value)
                #print(indices)
                for key in file:
                    key_len = file[key]['data'].shape[0]
                    if key not in indices:
                        # нет ни одного элемента с этим ключом,
                        # если элементы file нельзя распределить по replicas без добавления случайных сэмплов
                        # или памяти не хватает даже для одного элемента
                        if (drop_last and key_len < num_replicas) or mem < np.prod(file[key]['data'][0].shape).item():
                            continue
                        else:
                            raise ValueError(f'Нет ни одного элемента группы {key}')
                    else:
                        diff = key_len - len(indices[key])  # количество неохваченных элементов
                        if not((not drop_last and diff == 0) or (drop_last and diff < num_replicas)):
                            raise ValueError('Не охвачены все элементы группы')
                        elif not((0 <= min(indices[key]) < key_len) and (0 <= max(indices[key]) < key_len)):
                            raise ValueError('Индексы выходят за рамки группы')

                def list_of_one_element(l):
                    return l == [ l[0] ] * len(l)
                
                if not list_of_one_element([ len(batches[rank]) for rank in range(num_replicas) ]):
                    raise ValueError('Не все батчи одинакового размера')

                """ #TODO: все сэмплы в батчах кроме случайных должны разниться между репликами
                diff = {}
                for i in range(len(batches[0])):
                    diff[key] += list_of_one_element(
                        [ batches[rank][i] for rank in range(num_replicas) ]
                    )
                for key, value in dict.items():
                    pass
                """