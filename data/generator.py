import os
import sys
import math
import random as rnd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse

parser = argparse.ArgumentParser(description='Synthetic data generator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--font-folder', default='data/fonts/top60_ru', type=str)
parser.add_argument('-o', '--output-dir', default='datasets/top60_ru_synth', type=str)
parser.add_argument('-i', '--images', help='number of images generated per font', default=100, type=int)
parser.add_argument('-u', '--unlabeled', default=False, type=bool)
# https://github.com/hingston/russian
parser.add_argument('-d', '--dict', default='10000-russian-words-cyrillic-only.txt', type=str)
parser.add_argument('-r', '--regular', default=False, type=bool)
args = parser.parse_args()

font_list = os.listdir(args.font_folder)

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
    #shadow_alpha.paste(0, (0, 0, shadow_alpha.size[0], shadow_alpha.size[1])) # reset shadow_alpha
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

def create_random_txt(text, font):
    fill_color = rnd.randint(0, 255)
    stroke_width = rnd.randint(0, 20)
    stroke_fill_color = rnd.randint(0, 255)
    character_spacing = rnd.randint(0, 20)

    diff = fill_color - stroke_fill_color
    if abs(diff) < 20:
        fill_color += int(math.copysign(10, diff))
        stroke_fill_color -= int(math.copysign(10, diff))
 
    return create_txt(text, character_spacing, (fill_color, 255), font, stroke_width=stroke_width, stroke_fill=(stroke_fill_color, 255))

def create_txt(text, character_spacing, fill, font, stroke_width, stroke_fill, fit=True):
    piece_widths = [
        font.getsize(c)[0] for c in text
    ]
    text_width = sum(piece_widths) + ((character_spacing) * (len(text) - 1)) + stroke_width * 2
    text_height = max([font.getsize(c)[1] + (stroke_width * 2) for c in text])
    txt_img = Image.new("LA", (text_width, text_height), (0, 0))
    txt_img_draw = ImageDraw.Draw(txt_img)

    cx = 0
    i = 0
    for c in text:
        txt_img_draw.text((cx, 0), c, fill, font, stroke_width=stroke_width, stroke_fill=stroke_fill)
        cx += piece_widths[i] + character_spacing
        i += 1

    cx = stroke_width
    i = 0
    for c in text:
        txt_img_draw.text((cx, stroke_width), c, fill, font)
        cx += piece_widths[i] + character_spacing
        i += 1
    
    if fit:
        return txt_img.crop(txt_img.getbbox())
    else:
        return txt_img

def np_img(pil_img):
    return np.array(pil_img)

def pil_img(np_img):
    return Image.fromarray(np_img)

lang_dict = []
if os.path.isfile(args.dict):
    with open(args.dict, "r", encoding="utf8", errors="ignore") as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 0]
else:
    sys.exit("Cannot open dict")
dict_len = len(lang_dict)

def random_string_from_dict(length = 3, allow_variable=True):
    current_string = ""
    for _ in range(0, rnd.randint(1, length) if allow_variable else length):
        current_string += lang_dict[rnd.randrange(dict_len)]
    return current_string

i = 0
font_num = 0
for font_name in font_list:
    
    if args.regular:
        if not(('regular' in font_name) or ('Regular' in font_name)):
            continue

    if font_num == 5: break
    font_num += 1

    savedir = '%s/%s' % (args.output_dir, 'no_label' if args.unlabeled else font_name[:-4])

                   #len(os.listdir(savedir))
    for _ in range(0, args.images):
    
        font = ImageFont.truetype(os.path.join(args.font_folder, font_name), size=rnd.randint(40, 80))

        str = random_string_from_dict(1)
        txt = create_random_txt(str, font)

        txt_alpha = txt.split()[-1]

        border = 10
        txt_offset = (border, border)
        txt_background = Image.new('LA', (txt.size[0] + 2*border, txt.size[1] + 2*border), (255, 255))

        shadow_count = rnd.randint(0, 5)
        while shadow_count > 0:
            drop_random_shadow(txt_alpha, txt_offset, txt_background)
            shadow_count -= 1

        #print('1', txt_background.size)
        txt_background.paste(txt, txt_offset, txt_alpha)
        txt_background.resize((
            round(txt_background.size[0] * rnd.uniform(0.5, 1.5)),
            round(txt_background.size[1] * rnd.uniform(0.5, 1.5))
        ))
        #print('2', txt_background.size)

        os.makedirs(savedir, exist_ok=True)
        txt_background.save('%s/%d.tiff' % (savedir, i))

        i += 1

    