import os
import sys
import random as rnd

batch_size = int(sys.argv[1])
basepath = sys.argv[2]

folders = os.listdir(basepath)
files = []
min_files = 0

for i, folder in enumerate(folders, 0):
    files.append(os.listdir('%s/%s' % (basepath, folder)))
    count = len(files[i])
    if min_files == 0 or count < min_files:
        min_files = count

if len(sys.argv) > 3:
    min_files = int(sys.argv[3])

addit = ((min_files * len(folders)) % batch_size) // len(folders)
for i, folder in enumerate(folders, 0):
    for _ in range(len(files[i]) - min_files + addit):
        deleted = rnd.choice(files[i])
        os.remove('%s/%s/%s' % (basepath, folder, deleted))
        files[i].remove(deleted)