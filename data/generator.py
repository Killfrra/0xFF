import os
import sys
import math
import random as rnd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse
#import imgaug as ia
#import imgaug.augmenters as iaa

parser = argparse.ArgumentParser(description='Synthetic data generator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--font-folder', default='data/fonts/top60_ru', type=str)
parser.add_argument('-o', '--output-dir', default='datasets/top60_ru_synth', type=str)
parser.add_argument('-i', '--images', help='number of images generated per font', default=100, type=int)
parser.add_argument('-u', '--unlabeled', default=False, type=bool)
# https://github.com/hingston/russian
parser.add_argument('-d', '--dict', default='10000-russian-words-cyrillic-only.txt', type=str)
parser.add_argument('-r', '--regular', default=False, type=bool)
parser.add_argument("-b", "--image_dir", type=str, default='datasets/bg_img')
parser.add_argument("-m", "--mask", type=bool, default=False)
args = parser.parse_args()

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

def create_random_txt(text, font, font_size):
    fill_color = rnd.randint(0, 255)
    stroke_width = rnd.randint(0, round(font_size * 0.3)) * rnd.choice([0,1])
    stroke_fill_color = rnd.randint(0, 255)
    character_spacing = rnd.randint(0, 20)

    diff = fill_color - stroke_fill_color
    if abs(diff) < 30:
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

    if args.mask:
        mask = Image.new('L', txt_img.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        cx = stroke_width
        i = 0
        for c in text:
            mask_draw.text((cx, stroke_width), c, 255, font)
            cx += piece_widths[i] + character_spacing
            i += 1
    
    if args.mask:
        if fit:
            bbox = txt_img.getbbox()
            return (txt_img.crop(bbox), mask.crop(bbox))
        else:
            return txt_img, mask
    else:
        if fit:
            bbox = txt_img.getbbox()
            return txt_img.crop(bbox), None
        else:
            return txt_img, None

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

import PIL

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

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import SegmentationMapsOnImage
from math import sqrt

preprocess = iaa.Sequential([
    iaa.PerspectiveTransform(keep_size=False, fit_output=False),
    iaa.PiecewiseAffine(),
    #iaa.ElasticTransformation(alpha=(0, 0.25), sigma=(0, 0.05)),
    iaa.Affine(scale={'x': (1.0, 1.1), 'y':(1.0, 1.1)}, rotate=(-10, 10), shear=(-10, 10), fit_output=False),
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0, 2.0)),
        iaa.Sometimes(0.5, [iaa.MotionBlur(k=3)])
    ]),
    #iaa.imgcorruptlike.GlassBlur(severity=1),
    #iaa.imgcorruptlike.DefocusBlur(severity=1),
    #iaa.imgcorruptlike.Pixelate(severity=1)
    #iaa.imgcorruptlike.ShotNoise(severity=1)
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255)),
    iaa.JpegCompression(compression=(0, 75)),
    iaa.Crop(percent=0.01),
])

postprocess = iaa.Sequential([
    iaa.Resize({"height": 63, "width": "keep-aspect-ratio"}),
    iaa.CropToFixedSize(63, 252),
    iaa.PadToFixedSize(63, 63, pad_mode=ia.ALL, pad_cval=(0, 255))
])

i = 0
font_num = 0
for font_name in font_list:
    
    if args.regular and not 'regular' in font_name.lower():
        continue

    if font_num == 1: break
    font_num += 1
    #if font_num < 5: continue
    
    #savedir = '%s/%s' % (args.output_dir, 'no_label' if args.unlabeled else font_name[:-4])
    #os.makedirs(savedir, exist_ok=True)

                   #len(os.listdir(savedir))
    for _ in range(0, args.images):
    
        font_size = rnd.randint(22, 72)
        font = ImageFont.truetype(os.path.join(args.font_folder, font_name), size=font_size)

        str = random_string_from_dict(1)
        txt, mask = create_random_txt(str, font, font_size)

        txt_alpha = txt.split()[-1]

        border = 16 #rnd.randint(0, round(font_size * 0.3))
        #txt_offset = (border, border)
        max_text_side = round(max(txt.size[0], txt.size[1]) * 1.1)
        min_text_side = round(min(txt.size[0], txt.size[1]) * 1.1)
        bg_side = max_text_side + min_text_side // 2 + border
        bg_size = [ bg_side ] * 2 #(txt.size[0] + 2*border, txt.size[1] + 2*border)
        txt_background = image_background(bg_size)

        txt_offset = (
            (txt_background.size[0] - txt.size[0]) // 2,
            (txt_background.size[1] - txt.size[1]) // 2
        )
        shadow_count = rnd.randint(0, 5)
        while shadow_count > 0:
            drop_random_shadow(txt_alpha, txt_offset, txt_background)
            shadow_count -= 1

        txt_background.paste(txt, txt_offset, txt_alpha)

        if args.mask:
            txt_mask = Image.new('L', bg_size, 0)
            txt_mask.paste(mask, txt_offset)
            #txt_mask = txt_mask.resize((txt_mask.size[0] // 8, txt_mask.size[1] // 8))
            #txt_mask.save('%s/%d_mask.tiff' % (savedir, i))

        mask = SegmentationMapsOnImage(np_img(txt_mask), txt_mask.size)
        background, mask = preprocess(image=np_img(txt_background), segmentation_maps=mask)
        mask = pil_img(mask.get_arr())
        bbox = mask.getbbox()
        bbox = (max(0, bbox[0] - border),
                max(0, bbox[1] - border),
                min(bbox[2] + border, mask.size[0]),
                min(bbox[3] + border, mask.size[0]))
        background = pil_img(background).crop(bbox)

        #mask.save(f'{savedir}/{i}_mask.tiff')

        square_side = 127

        background = background.resize((
            round(square_side * background.size[0] / background.size[1]),
            square_side
        ))

        nearest_width = (background.size[0] // square_side) * square_side
        
        if nearest_width > 0:
            diff = background.size[0] - nearest_width 
            background = background.crop((
                0 + diff // 2,
                0,
                nearest_width + diff // 2,
                square_side
            ))
        else:
            continue

        savedir = f'{args.output_dir}/{nearest_width}/{font_name[:-4]}'
        os.makedirs(savedir, exist_ok=True)

        background.save(f'{savedir}/{i}.tiff')

        """
        save_path = '%s/%d.tiff' % (savedir, i)
        try:
            txt_background.save(save_path)
        except TypeError:
            try:
                os.remove(save_path)
                i -= 1
            except:
                pass
        """

        i += 1

    
