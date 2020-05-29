import os
import sys
from PIL import Image
from PIL.ImageOps import autocontrast
from torchvision.transforms.functional import resize

side_size = 46#px

def process_image(image):

    image = image.convert('L')
    image = image.crop(image.getbbox())
    image = resize(image, side_size)

    width, height = image.size
    x, y = 0, 0

    crops = []

    while y <= height - side_size:
        while x <= width - side_size:
            crop = image.crop((x, y, x + side_size, y + side_size))
            crop = autocontrast(crop)
            crops.append(crop)

            x += side_size // 2
        y += side_size // 2

    return crops

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