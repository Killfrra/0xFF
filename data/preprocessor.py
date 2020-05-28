 
import os
from PIL import Image
import torchvision.transforms.functional as F
from math import floor
from PIL.ImageOps import autocontrast
import sys

square_side = 56#px
min_shift = 30#px
min_size = 56#px
max_size = square_side * 1 + min_shift
downscale = 0.5

def is_fractional(x):
    return x > floor(x)

def overlap(side):
    count = side / square_side
    if is_fractional(count):     # если не делится нацело
        count = floor(count) + 1 # округляем в большую сторону
    
    if count == 1:
        overlap = 0
    else:
        overlap = round((count * square_side - side) / (count - 1))

        if square_side - overlap < min_shift:
            count -= 1
            if count == 1:
                overlap = 0
            else:
                overlap = round(max(0, (count * square_side - side) / (count - 1)))
        
    discarded = side - (square_side * count - overlap * (count - 1))

    return (int(count), int(overlap), int(discarded))

def generate_crops(image):

    width, height = image.size

    count_x, overlap_x, discarded_x = overlap(width)
    count_y, overlap_y, discarded_y = overlap(height)

    if abs(discarded_x) > 0 or abs(discarded_y) > 0:
        image = image.resize((
            width - max(0, discarded_x),
            height - max(0, discarded_y)
        ))
        width, height = image.size
        background = image.resize((1, 1)).resize((
            width - min(0, discarded_x),
            height - min(0, discarded_y)
        ))
        background.paste(image, (
            round((background.size[0] - width) / 2),
            round((background.size[1] - height) / 2)
        ))
        image = background

    crops = []

    for iy in range(count_y):
        for ix in range(count_x):
            x = (square_side - overlap_x) * ix
            y = (square_side - overlap_y) * iy
            
            crop = image.crop((x, y, x + square_side, y + square_side))
            crop = autocontrast(crop)

            crops.append(crop)

    return crops

def fit_square(image, square_side):
    width, height = image.size
    background = image.resize((1, 1)).resize((square_side, square_side))
    if width > height:
        image = image.resize((square_side, round(height*square_side/width)))
    else:
        image = image.resize((round(width*square_side/height), square_side))
    background.paste(image, (
        round((square_side - image.size[0]) / 2),
        round((square_side - image.size[1]) / 2)
    ))
    return background

def process_image(image):

    image = image.convert('L').crop(image.getbbox())

    width, height = image.size

    if max(width, height) <= square_side:
        image = fit_square(image, square_side)
        autocontrast(image)
        return [image]
        #return generate_crops(image)

    if min(width, height) > max_size:
        image = F.resize(image, max_size)
    
    all_crops = []

    while True:
        
        crops = generate_crops(image)
        all_crops.extend(crops)

        if len(crops) == 1:
            return all_crops
        
        width, height = image.size
        new_width = round(width * downscale)
        new_height = round(height * downscale)
        
        if min(new_width, new_height) < min_size:
            return all_crops

        if max(new_width, new_height) <= square_side:
            if round(min(width, height) * square_side / max(width, height)) >= min_size:
                image = fit_square(image, square_side)
                all_crops.append(image)
            
            return all_crops

        image = image.resize((new_width, new_height))

i = 0
def process_images_in_folder(basepath, output, k = 1):
    global i
    os.makedirs(output, exist_ok=True)
    for image_name in os.listdir(basepath):
        
        image_path = '%s/%s' % (basepath, image_name)
        if os.path.isdir(image_path):
            if k > 0:
                process_images_in_folder(image_path, '%s/%s' % (output, image_name), k - 1)
            continue

        try:
            image = Image.open(image_path)
        except ValueError as e:
            print(image_name, e)
            continue

        crops = process_image(image)
        for crop in crops:
            crop.save('%s/%d.tiff' % (output, i))
            i += 1

if __name__ == '__main__':
    process_images_in_folder(
        basepath=sys.argv[1],
        output=sys.argv[2]
    )
    #print(i, 'crops generated from', j, 'images. ', k, 'crops filtered')