import os
import sys
import math
import random as rnd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse
import PIL

parser = argparse.ArgumentParser(description='Synthetic data generator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--font-folder', default='data/fonts/top60_ru', type=str)
parser.add_argument('-o', '--output-dir', default='datasets/top60_ru_synth', type=str)
parser.add_argument('-i', '--images', help='number of images generated per font', default=100, type=int)
parser.add_argument('-u', '--unlabeled', default=False, type=bool)
parser.add_argument('-r', '--regular', default=False, type=bool)
parser.add_argument("-b", "--image_dir", type=str, default='datasets/bg_img')
args = parser.parse_args()

alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
square_side = 64

font_list = os.listdir(args.font_folder)
background_images = os.listdir(args.image_dir)

def drop_random_shadow(txt_alpha, from_on_to_offset, txt_background):
    box = (rnd.randint(-10, 10), rnd.randint(-10, 10))
    radius = rnd.randint(0, 10)
    color = rnd.randint(0, 255)
    transparency = rnd.randint(0, 255)
    return drop_shadow(txt_alpha, from_on_to_offset, txt_background, box, radius, color, transparency)

def drop_shadow(from_alpha, from_on_to_offset, to, shadow_offset, blur_radius, color, transparency):
    shadow = Image.new('LA', (
        from_alpha.size[0] + 2*blur_radius,
        from_alpha.size[1] + 2*blur_radius,
    ))
    shadow_alpha = Image.new('L', shadow.size, 0)

    shadow_alpha.paste(from_alpha, (
        blur_radius,
        blur_radius
    ))
    shadow_alpha = shadow_alpha.filter(ImageFilter.BoxBlur(blur_radius))
    shadow.paste((color, 255), shadow_alpha.point(lambda pix: round(pix * transparency)))
    to.paste(shadow, (
        from_on_to_offset[0] + shadow_offset[0] - blur_radius,
        from_on_to_offset[1] + shadow_offset[1] - blur_radius
    ), shadow_alpha)

def create_random_txt(text, font, font_size):
    fill_color = rnd.randint(0, 255)
    stroke_width = rnd.randint(0, round(font_size * 0.3))
    stroke_fill_color = rnd.randint(0, 255)

    diff = fill_color - stroke_fill_color
    if abs(diff) < 30:
        fill_color += int(math.copysign(10, diff))
        stroke_fill_color -= int(math.copysign(10, diff))
 
    return create_txt(text, (fill_color, 255), font, stroke_width=stroke_width, stroke_fill=(stroke_fill_color, 255))

def create_txt(c, fill, font, stroke_width, stroke_fill, fit=True):
    text_width, text_height = font.getsize(c)
    text_width += stroke_width * 2
    text_height += stroke_width * 2

    txt_img = Image.new("LA", (text_width, text_height), (0, 0))
    txt_img_draw = ImageDraw.Draw(txt_img)

    txt_img_draw.text((0, 0), c, fill, font, stroke_width=stroke_width, stroke_fill=stroke_fill)
    txt_img_draw.text((stroke_width, stroke_width), c, fill, font)

    return (txt_img.crop(txt_img.getbbox()) if fit else txt_img, stroke_width)


def image_background(size): # from trdg.background_generator
    
    width, height = size
    
    while len(background_images) > 0:
        try:
            pic_name = background_images[rnd.randint(0, len(background_images) - 1)]
            pic_path = '%s/%s' % (args.image_dir, pic_name)
            pic = Image.open(pic_path).convert('L')
            break
        except Exception as e:
            print(e)
            os.remove(pic_path)
            background_images.remove(pic_name)

    if pic.size[0] < width:
        pic = pic.resize(
            [width, int(pic.size[1] * (width / pic.size[0]))], Image.ANTIALIAS
        )
    if pic.size[1] < height:
        pic = pic.resize(
            [int(pic.size[0] * (height / pic.size[1])), height], Image.ANTIALIAS
        )

    if pic.size[0] == width:
        x = 0
    else:
        x = rnd.randint(0, pic.size[0] - width)
    if pic.size[1] == height:
        y = 0
    else:
        y = rnd.randint(0, pic.size[1] - height)

    return pic.crop((x, y, x + width, y + height))

i = 0
font_num = 0
for font_name in font_list:
    
    if args.regular and not 'regular' in font_name.lower():
        continue

    if font_num == 5: break
    font_num += 1
    #if font_num < 3: continue
    
    savedir = '%s/%s' % (args.output_dir, 'no_label' if args.unlabeled else font_name[:-4])
    os.makedirs(savedir, exist_ok=True)
    font_path = os.path.join(args.font_folder, font_name)

    for _ in range(args.images):
        for letter in alphabet:
        
            font_size = rnd.randint(32, 64)
            font = ImageFont.truetype(font_path, size=font_size)

            txt, stroke_width = create_random_txt(letter, font, font_size)

            txt_alpha = txt.split()[-1]

            text_width, text_height = txt.size
            text_width -= stroke_width * 2
            text_height -= stroke_width * 2

            border = (square_side - text_width, square_side - text_height)
            bg_size = (text_width + 2*border[0], text_height + 2*border[1])
            txt_background = image_background(bg_size)

            txt_offset = ((bg_size[0] - txt.size[0]) // 2, (bg_size[1] - txt.size[1]) // 2)

            for _ in range(rnd.randint(0, 5)):
                drop_random_shadow(txt_alpha, txt_offset, txt_background)
            
            txt_background.paste(txt, txt_offset, txt_alpha)

            try:
                txt_background.save('%s/%d.tiff' % (savedir, i))
            except TypeError:
                pass

            """
            Traceback (most recent call last):
            File "generator.py", line 149, in <module>
                txt_background.save('%s/%d.tiff' % (savedir, i))
            File "/home/m/.local/lib/python3.6/site-packages/PIL/Image.py", line 2134, in save
                save_handler(self, fp, filename)
            File "/home/m/.local/lib/python3.6/site-packages/PIL/TiffImagePlugin.py", line 1616, in _save
                e = Image._getencoder(im.mode, "libtiff", a, im.encoderconfig)
            File "/home/m/.local/lib/python3.6/site-packages/PIL/Image.py", line 461, in _getencoder
                return encoder(mode, *args + extra)
            TypeError: argument 3 must be str, not int
            """

            i += 1

    
