import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

font_folder = 'fonts'
font_list = os.listdir(font_folder)

def drop_random_shadow(txt_alpha, from_on_to_offset, txt_background):
    box = (random.randint(-10, 10), random.randint(-10, 10))
    radius = random.randint(0, 10)
    color = random.randint(0, 255)
    transparency = random.randint(0, 255)
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
    fill_color = random.randint(0, 255)
    stroke_width = random.randint(0, 20)
    stroke_fill_color = random.randint(0, 255)
    character_spacing = random.randint(-10, 20)

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

i = 0
for font_name in font_list:
    if i >= 30:
        break

    #try:

    font = ImageFont.truetype(os.path.join(font_folder, font_name), size=80)

    txt = create_random_txt('привет', font)

    txt_alpha = txt.split()[-1]

    border = 30
    txt_offset = (border, border)
    txt_background = Image.new('LA', (txt.size[0] + 2*border, txt.size[1] + 2*border), (0, 0))

    shadow_count = random.randint(0, 5)
    while shadow_count > 0:
        drop_random_shadow(txt_alpha, txt_offset, txt_background)
        shadow_count -= 1

    #print('1', txt_background.size)
    txt_background.paste(txt, txt_offset, txt_alpha)
    txt_background = txt_background.resize((
        txt_background.size[0] - random.randint(0, 30),
        txt_background.size[1] - random.randint(0, 30)
    ))
    #print('2', txt_background.size)

    txt_background.convert('RGBA').save('./output/' + str(i) + '.png')

    i += 1
    #except ValueError as e:
    #    print(font_name, '\n', e)

for img in [txt, txt_alpha, txt_background]:
    img.close()
    