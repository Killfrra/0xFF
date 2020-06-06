import h5py
import os
from PIL import Image, ImageFont
import numpy as np
import random as rnd

fonts_dir = 'fonts'
fonts_list = [ f'font_{x}' for x in range(3) ] #sorted(os.listdir(fonts_dir))
fonts_count = len(fonts_list)

alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
formats = [(63, 63), (63, 126)]
imgs_per_fnt = 10
min_font_size = 11

dataset_len = fonts_count * imgs_per_fnt

with h5py.File('datasets.hdf5', 'w') as f:

    grp = f.create_group('train')
    for fmt in formats:
        data = grp.create_dataset(f'{fmt[0]}x{fmt[1]}', shape=(dataset_len, fmt[0], fmt[1]), dtype=np.uint8)
        for class_id, font in enumerate(fonts_list):
            for image_id in range(imgs_per_fnt):

                img = np.array(generate_image(fmt, font))

                i = class_id * imgs_per_fnt + image_id
                data[i] = img

def generate_image(size, font_name):
    width, height = size
    background = image_background(size)

    font_size = rnd.randint(min_font_size, height)
    font = ImageFont.truetype(os.path.join(fonts_dir, font_name), size=font_size)

    text = rnd.choice(alphabet)
    text_width = font.getsize(text)[0]
    while text_width < width:
        char = rnd.choice(alphabet)
        text += char
        text_width += font.getsize(char)[0]

    return background

def image_background(size):
    return Image.new('L', size, 255)