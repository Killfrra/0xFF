import os
import sys
import h5py
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Synthetic data generator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-b', '--basepath', default='datasets/test', type=str)
parser.add_argument('-o', '--outpath', default='datasets/test.hdf5', type=str)
parser.add_argument('-s', '--startfolder', default='127', type=str)
args = parser.parse_args()

lvl1_size_folders = sorted(os.listdir(args.basepath), reverse=True)
lvl2_font_folders = sorted(os.listdir('%s/%s' % (args.basepath, args.startfolder)))
fonts = len(lvl2_font_folders)
min_files = [ 0 ] * len(lvl1_size_folders)
files     = [    ]

for i, size in enumerate(lvl1_size_folders):
    if len(os.listdir('%s/%s' % (args.basepath, size))) != fonts:
        continue
    for j, font in enumerate(lvl2_font_folders):
        images = sorted(os.listdir('%s/%s/%s' % (args.basepath, size, font)))
        #files[i * fonts + j] = images
        files.append(images)
        count = len(images)
        if min_files[i] == 0 or count < min_files[i]:
            min_files[i] = count

lvl1_size_folders = [ size for i, size in enumerate(lvl1_size_folders) if min_files[i] != 0 ]
min_files = [ count for count in min_files if count != 0 ]
sizes = len(lvl1_size_folders)
print(sum(min_files) * fonts, 'files total')

with h5py.File(args.outpath, 'w') as f:
    f.attrs['class_num'] = fonts
    for i, size in enumerate(lvl1_size_folders):
        if min_files[i] == 0:
            continue
        grp = f.create_group(size)
        data_len = min_files[i] * fonts
        fmt = (127, int(size))
        data = grp.create_dataset('data', shape=(data_len, fmt[0], fmt[1]), dtype=np.uint8)
        labels = grp.create_dataset('labels', shape=(data_len,), dtype='i8')
        for j, font in enumerate(lvl2_font_folders):
            images = files[i * fonts + j]
            for k in range(min_files[i]):
                img = Image.open('%s/%s/%s/%s' % (args.basepath, size, font, images[k]))
                img = np.array(img, dtype=np.uint8)
                n = j * min_files[i] + k
                data[n] = img
                labels[n] = j
        print(i, size, 'ready')