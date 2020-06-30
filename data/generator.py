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
#parser.add_argument('-u', '--unlabeled', default=False, type=bool)
# https://github.com/hingston/russian
parser.add_argument('-d', '--dict', default='10000-russian-words-cyrillic-only.txt', type=str)
parser.add_argument('-r', '--regular', default=False, type=bool)
parser.add_argument("-b", "--image-dir", type=str, default='datasets/bg_img')
#parser.add_argument("-m", "--mask", type=bool, default=False)
parser.add_argument('-t', '--thread-count', type=int, default=1)
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
    min_color_diff = 30
    if abs(diff) < min_color_diff:
        fill_color += int(math.copysign(min_color_diff // 2, diff))
        stroke_fill_color -= int(math.copysign(min_color_diff // 2, diff))
 
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

    #if args.mask:
    mask = Image.new('1', txt_img.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    cx = stroke_width
    i = 0
    for c in text:
        mask_draw.text((cx, stroke_width), c, 255, font)
        cx += piece_widths[i] + character_spacing
        i += 1
    
    #if args.mask:
    if fit:
        bbox = txt_img.getbbox()
        return (txt_img.crop(bbox), mask.crop(bbox))
    else:
        return txt_img, mask
    #else:
    #    if fit:
    #        bbox = txt_img.getbbox()
    #        return txt_img.crop(bbox), None
    #    else:
    #        return txt_img, None

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

def random_string_from_dict(): #length = 3, allow_variable=True):
    #current_string = ""
    #for _ in range(0, rnd.randint(1, length) if allow_variable else length):
    #    current_string += lang_dict[rnd.randrange(dict_len)]
    return lang_dict[rnd.randrange(dict_len)] #current_string

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

preprocess = [
    iaa.Identity(),
    iaa.Sequential([
        iaa.Affine(scale={'x': (1, 1.1), 'y': (1, 1.1)}, rotate=(-10, 10), fit_output=False),
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),
        #iaa.JpegCompression(compression=(75, 100)),
    ]),
    iaa.Sequential([
        iaa.Affine(scale={'x': (1, 1.1), 'y': (1, 1.1)}, rotate=(-10, 10), shear=(-10, 10), fit_output=False),
        iaa.PiecewiseAffine(),
        iaa.PerspectiveTransform(keep_size=False, fit_output=False),
        #iaa.ElasticTransformation(alpha=(0, 0.25), sigma=(0, 0.05)),
        #iaa.OneOf([
        iaa.GaussianBlur(sigma=(0, 2.0)),
        #    iaa.Sometimes(0.5, [iaa.MotionBlur(k=3)])
        #]),
        #iaa.imgcorruptlike.GlassBlur(severity=1),
        #iaa.imgcorruptlike.DefocusBlur(severity=1),
        #iaa.imgcorruptlike.Pixelate(severity=1)
        #iaa.imgcorruptlike.ShotNoise(severity=1)
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),
        #iaa.JpegCompression(compression=(75, 99)),
        #iaa.Crop(percent=0.01),
    ])
]

def generate_from_tuple(tuple):
    return generate(*tuple)

def generate(i, font_folder, font_name, string, output_dir):
    font_size = rnd.randint(11, 72)
    font = ImageFont.truetype(os.path.join(font_folder, font_name), size=font_size)

    txt, mask = create_random_txt(string, font, font_size)

    txt_alpha = txt.split()[-1]

    max_affine_scale = 1.1
    max_text_side = round(max(txt.size[0], txt.size[1]) * max_affine_scale)
    min_text_side = round(min(txt.size[0], txt.size[1]) * max_affine_scale)
    border = 16 # вместо того, чтобы париться с тригонометрией, имперически выведеден отступ
    bg_side = max_text_side + min_text_side // 2 + border
    bg_size = [ bg_side ] * 2
    background = image_background(bg_size)

    txt_offset = (
        (background.size[0] - txt.size[0]) // 2,
        (background.size[1] - txt.size[1]) // 2
    )
    max_shadow_count = 5
    for _ in range(rnd.randint(0, max_shadow_count)):
        drop_random_shadow(txt_alpha, txt_offset, background)

    background.paste(txt, txt_offset, txt_alpha)

    #if args.mask:
    txt_mask = Image.new('L', bg_size, 0)
    txt_mask.paste(mask, txt_offset)
    #txt_mask = txt_mask.resize((txt_mask.size[0] // 8, txt_mask.size[1] // 8))
    #txt_mask.save('%s/%d_mask.tiff' % (savedir, i))

    mask = SegmentationMapsOnImage(np_img(txt_mask), txt_mask.size)
    if   font_size <= 16: level = 0
    elif font_size <= 32: level = 1
    else                : level = 2
    background, mask = preprocess[level](image=np_img(background), segmentation_maps=mask)
    mask = pil_img(mask.get_arr())
    bbox = mask.getbbox()

    #min_side = bbox[3] - bbox[1] #min(bbox[2] - bbox[0], bbox[3] - bbox[1])
    #precent_loss = rnd.uniform(0, 10)
    #loss_distribution = rnd.uniform(0, 1)
    max_border = round(90 * ((font_size - 11) / (72 - 11)) + 10) #px
    bbox = ( #TODO: what if minside is not Y?
        max(0, bbox[0] - rnd.randint(0, max_border)),
        max(0, bbox[1] - rnd.randint(0, max_border)),
        #round(-min_side * precent_loss * (1 - loss_distribution))
        min(bbox[2] + rnd.randint(0, max_border), mask.size[0]), #TODO: mask.size[0]?
        min(bbox[3] + rnd.randint(0, max_border), mask.size[0])
    )
    bbox_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    ratio = bbox_size[0] / bbox_size[1]
    square_side = 127
    w_after_resize = square_side * ratio
    nearest_width = round((w_after_resize // square_side) * square_side)
    if nearest_width == 0:
        nearest_width = square_side
    diff = (w_after_resize - nearest_width) * (bbox_size[1] / square_side)
    #print('DEBUG', nearest_width, square_side * ((bbox_size[0] - diff) / bbox_size[1]))
    background = pil_img(background).crop((
        bbox[0] + diff // 2,
        bbox[1],
        bbox[2] - diff // 2,
        bbox[3]
    )).resize((
        nearest_width,
        square_side
    ))

    savedir = f'{output_dir}/{nearest_width}/{font_name[:-4]}'
    os.makedirs(savedir, exist_ok=True)

    background.save(f'{savedir}/{i}.tiff')

from tqdm import tqdm
from multiprocessing import Pool
pool = Pool(args.thread_count)
count = len(font_list) * args.images

for _ in tqdm(
    pool.imap_unordered(
        generate_from_tuple,
        zip(
            [ i for i in range(count) ],
            [ args.font_folder ] * count,
            font_list * args.images,
            [ random_string_from_dict() for _ in range(count) ],
            [ args.output_dir ] * count
        )
    ),
    total=count
): pass

pool.terminate()
