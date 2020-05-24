 
import os
from PIL import Image
import torchvision.transforms.functional as F
from math import floor
from PIL.ImageOps import autocontrast

square_side = 96#px
min_shift = 20#px

def is_fractional(x):
    return x > floor(x)

def overlap(side):
    count = side / square_side
    if is_fractional(count):     # если не делится нацело
        count = floor(count) + 1 # округляем в большую сторону
    
    if count == 1:
        overlap = 0
        discarded = 0
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

i = 0
def generate_crops(image, output):
    global i
    width, height = image.size

    count_x, overlap_x, discarded_x = overlap(width)
    count_y, overlap_y, discarded_y = overlap(height)

    #print(image_name, width, height, count_x, count_y, overlap_x, overlap_y, discarded_x, discarded_y)

    if abs(discarded_x) > 0 or abs(discarded_y) > 0:
        image = image.resize((width - discarded_x, height - discarded_y))
        width, height = image.size
        #count_x, overlap_x, discarded_x = overlap(width)
        #count_y, overlap_y, discarded_y = overlap(height)
        #print(image_name, width, height, count_x, count_y, overlap_x, overlap_y, discarded_x, discarded_y)

    for iy in range(count_y):
        for ix in range(count_x):
            x = (square_side - overlap_x) * ix
            y = (square_side - overlap_y) * iy
            #print(i, 'crop', x, y, x + square_side, y + square_side)
            crop = image.crop((x, y, x + square_side, y + square_side))
            crop = autocontrast(crop)

            crop.save('%s/%d.tiff' % (output, i))
            i += 1

    return image, width, height

downscale = 0.5
def process_images_in_folder(basepath, output, k = 1):
    os.makedirs(output, exist_ok=True)
    for image_name in os.listdir(basepath):
        image_path = '%s/%s' % (basepath, image_name)
        if os.path.isdir(image_path):
            if k > 0:
                process_images_in_folder(image_path, '%s/%s' % (output, image_name), k - 1)
            continue

        try:
            image = Image.open(image_path).convert('L')
        except ValueError as e:
            print(image_name, e)
            continue
        image = image.crop(image.getbbox())
        width, height = image.size
        if min(width, height) < square_side:
            image = F.resize(image, square_side)
            width, height = image.size
            #image.save('%s/%s_%dx%d.tiff' % (output, image_name, width, height))
            generate_crops(image, output)
        else:
            while True:
                #print(image_name, width, height)
                #image.save('%s/%s_%dx%d.tiff' % (output, image_name, width, height))
                generate_crops(image, output)
                new_width = round(width * downscale)
                new_height = round(height * downscale)
                if new_width < square_side or new_height < square_side:
                    break
                image = image.resize((new_width, new_height))
                width, height = image.size

if __name__ == '__main__':
    process_images_in_folder(
        #basepath='datasets/manually_selected_and_cropped_real',
        #output='datasets/preprocessed_unlabeled_real'
        basepath='datasets/VFR_real_test',
        output='datasets/preprocessed_labeled_real'
    )

print(i, 'crops generated')