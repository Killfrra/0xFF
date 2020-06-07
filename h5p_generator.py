import os
import sys
import h5py
import numpy as np
from PIL import Image

basepath = sys.argv[1]
outpath = sys.argv[2]
width = int(sys.argv[3])
group = sys.argv[4]

folders = os.listdir(basepath)
files = []
min_files = -1

for i, folder in enumerate(folders, 0):
    files.append(os.listdir('%s/%s' % (basepath, folder)))
    count = len(files[i])
    if min_files == -1 or count < min_files:
        min_files = count

print('min_files', min_files)

dataset_len = len(folders) * min_files
fmt = (127, width)

with h5py.File(outpath, 'w') as f:
    f.attrs['class_num'] = len(folders)
    grp = f.create_group(group)
    data = grp.create_dataset('data', shape=(dataset_len, fmt[0], fmt[1]), dtype=np.uint8)
    labels = grp.create_dataset('labels', shape=(dataset_len,), dtype='u4')
    for folder_id, folder in enumerate(folders):
        #labels[folder_id * min_files: folder_id * (min_files + 1) - 1] = [folder_id] * min_files
        for file_id, file_name in enumerate(files[folder_id]):
            if file_id == min_files:
                break
            img = Image.open('%s/%s/%s' % (basepath, folder, file_name))
            img = np.array(img, dtype=np.uint8)
            i = folder_id * min_files + file_id
            data[i] = img
            labels[i] = folder_id
        print(folder_id, folder, 'ready')
        