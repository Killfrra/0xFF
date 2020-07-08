import os
import sys
import math
import random as rnd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import argparse
import PIL
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import SegmentationMapsOnImage
from math import sqrt
from tqdm import tqdm
from multiprocessing import Pool

parser = argparse.ArgumentParser(description='Synthetic data generator', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--font-folder', default='data/fonts/google-fonts-cyrillic', type=str)
parser.add_argument('-o', '--output-dir', default='datasets/google', type=str)
parser.add_argument('-i', '--images', help='number of images generated per font', default=100, type=int)
#parser.add_argument('-u', '--unlabeled', default=False, type=bool)
# https://github.com/hingston/russian
parser.add_argument('-d', '--dict', default='10000-russian-words-cyrillic-only.txt', type=str)
#parser.add_argument('-r', '--regular', default=False, type=bool)
parser.add_argument("-b", "--image-dir", type=str, default='datasets/bg_img')
parser.add_argument('-t', '--thread-count', type=int, default=(os.cpu_count() or 1))
args = parser.parse_args()

font_list = {
    font_name: [
        font_ver for font_ver in os.listdir(f'{args.font_folder}/{font_name}')
    ] for font_name in os.listdir(args.font_folder)
}

background_images = os.listdir(args.image_dir)

def create_random_txt(text, font, font_size): #TODO: remove font_size
    fill_color = rnd.randint(0, 255)
    character_spacing = rnd.randint(0, 20)

    return create_txt(text, character_spacing, (fill_color, 255), font)

def create_txt(text, character_spacing, fill, font):
    piece_widths = [
        font.getsize(c)[0] for c in text
    ]
    text_width = sum(piece_widths) + character_spacing * (len(text) - 1)
    text_height = max([ font.getsize(c)[1] for c in text ])
    txt_img = Image.new("LA", (text_width, text_height), (0, 0))
    txt_img_draw = ImageDraw.Draw(txt_img)

    cx = 0
    i = 0
    for c in text:
        txt_img_draw.text((cx, 0), c, fill, font)
        cx += piece_widths[i] + character_spacing
        i += 1

    bbox = txt_img.getbbox()
    txt_img = txt_img.crop(bbox)
    return (txt_img, txt_img.split()[-1])

def np_img(pil_img):
    return np.array(pil_img)

def pil_img(np_img):
    return Image.fromarray(np_img)

lang_dict = []
if os.path.isfile(args.dict):
    with open(args.dict, "r", encoding="utf8", errors="ignore") as d:
        lang_dict = [l for l in d.read().splitlines() if len(l) > 2]
else:
    sys.exit("Cannot open dict")
dict_len = len(lang_dict)

def random_string_from_dict():
    return lang_dict[rnd.randrange(dict_len)]

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

preprocess = iaa.Sequential([
    iaa.Affine(scale={'x': (1, 1.1), 'y': (1, 1.1)}, rotate=(-15, 15), fit_output=False), # shear removed
    #iaa.PiecewiseAffine(),
    #iaa.PerspectiveTransform(keep_size=False, fit_output=False),
    #iaa.ElasticTransformation(alpha=(0, 0.25), sigma=(0, 0.05)),
    #iaa.OneOf([
    #iaa.GaussianBlur(sigma=(0, 2.0)),
    #    iaa.Sometimes(0.5, [iaa.MotionBlur(k=3)])
    #]),
    iaa.AdditiveGaussianNoise(scale=(0, 0.03*255)),
    #iaa.JpegCompression(compression=(75, 99)),
    #iaa.Crop(percent=0.01),
])

def generate_from_tuple(tuple):
    return generate(*tuple)

def generate(i, font_folder, font_name, font_ver, string, output_dir):
    font_size = rnd.randint(11, 72)
    font = ImageFont.truetype(os.path.join(font_folder, font_name, font_ver), size=font_size)

    txt, mask = create_random_txt(string, font, font_size)

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
    background.paste(txt, txt_offset, mask)

    txt_mask = Image.new('L', bg_size, 0)
    txt_mask.paste(mask, txt_offset)
    #txt_mask.save('%s/%d_mask.tiff' % (savedir, i))

    mask = SegmentationMapsOnImage(np_img(txt_mask), txt_mask.size)
    background, mask = preprocess(image=np_img(background), segmentation_maps=mask)
    mask = pil_img(mask.get_arr())
    bbox = mask.getbbox()

    square_side = 127 #TODO: split to height and step

    #TODO: border and loss distribution
    max_border = (square_side - font_size) // 2 #px
    bbox = ( #TODO: what if minside is not Y?
        max(0, bbox[0] - rnd.randint(0, max_border)),
        max(0, bbox[1] - rnd.randint(0, max_border)),
        #round(-min_side * precent_loss * (1 - loss_distribution))
        min(bbox[2] + rnd.randint(0, max_border), mask.size[0]), #TODO: mask.size[0]?
        min(bbox[3] + rnd.randint(0, max_border), mask.size[0])
    )
    bbox_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
    ratio = bbox_size[0] / bbox_size[1]
    
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

    savedir = f'{output_dir}/{nearest_width}/{font_name}'
    os.makedirs(savedir, exist_ok=True)

    background.save(f'{savedir}/{i}.png')


pool = Pool(args.thread_count)
count = len(font_list) * args.images

font_names = []
font_vers = []
for key, value in font_list.items():
    font_names.extend([key]*args.images)
    key_len = len(value)
    font_vers.extend(value * (args.images // key_len))
    font_vers.extend(value[:args.images % key_len])

for _ in tqdm(
    pool.imap_unordered(
        generate_from_tuple,
        zip(
            [ i for i in range(count) ],
            [ args.font_folder ] * count,
            font_names,
            font_vers,
            [ random_string_from_dict() for _ in range(count) ],
            [ args.output_dir ] * count
        )
    ),
    total=count
): pass

pool.terminate()
