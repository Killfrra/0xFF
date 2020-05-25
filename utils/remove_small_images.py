 
import os
from PIL import Image

basepath = '/media/m/UBUNTU/home/mak/Downloads/WTF/manually_selected'
min_size = 64

for image_name in os.listdir(basepath):
    try:
        image_path = '%s/%s' % (basepath, image_name)
        image = Image.open(image_path)
        if image.size[0] < min_size or image.size[1] < min_size:
            print(image_name)
            os.remove(image_path)
        image.close()
    except ValueError:
        print('ERROR', image_name)
