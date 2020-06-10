import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse

parser = argparse.ArgumentParser(description='Synthetic data generator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--font-folder', default='data/fonts/top60_ru', type=str)
parser.add_argument('-o', '--output-dir', default='docs/static/previews', type=str)
parser.add_argument('-r', '--regular', default=False, type=bool)
args = parser.parse_args()

font_list = os.listdir(args.font_folder)

phrase = [
    'возможность',
    'менять текст',
    'на превью обяза-',
    'тельно добавим,',
    'но чуть позже'
]

os.makedirs(args.output_dir, exist_ok=True)

font_num = 0
for font_name in font_list:
    
    if args.regular and not 'regular' in font_name.lower():
        continue
    
    for i, line in enumerate(phrase):
    
        font = ImageFont.truetype(os.path.join(args.font_folder, font_name), size=72)

        text_width, text_height = font.getsize(line)
        txt_img = Image.new("L", (text_width, text_height), 255)
        txt_img_draw = ImageDraw.Draw(txt_img)
        txt_img_draw.text((0, 0), line, 0, font)

        txt_img = txt_img.crop(txt_img.getbbox())

        background = Image.new("L", (text_width + 16, 72), 255)
        background.paste(txt_img, (
            (background.size[0] - txt_img.size[0]) // 2,
            (background.size[1] - txt_img.size[1]) // 2
        ))

        txt_img.save(f"{args.output_dir}/{font_name[:-4].replace(' ', '').replace('_', ''). replace('-', '')}_{i}.jpg")

    
