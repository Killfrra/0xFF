import os
from PIL import Image
from text_detection import get_crop_boxes

basepath = '../datasets/manually_selected'
output = '../datasets/cropped'

i = 0

os.makedirs(output, exist_ok=True)
for image_name in os.listdir(basepath):
    if i >= 100:
        break
    
    image_path = '%s/%s' % (basepath, image_name)
    try:
        image = Image.open(image_path)
    except ValueError as e:
        print(image_name, e)
        continue

    crop_boxes = get_crop_boxes(image)
    for box in crop_boxes:
        image.crop(box).save('%s/%d.tiff' % (output, i))
        i += 1
