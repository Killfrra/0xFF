from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, dir):
        self.dir = dir

        files = os.listdir(dir)
        extention = '.tiff'
        mask_postfix = '_mask' + extention
        masks = [ f for f in files if f.endswith(mask_postfix) ]
        self.dataset = []
        for mask in masks:
            image = mask[:-len(mask_postfix)] + extention
            if image in files:
                self.dataset.append((image, mask))

        self.transform = iaa.Sequential([
            iaa.Sometimes(0.5, [
                iaa.PerspectiveTransform(keep_size=False, cval=ia.ALL, mode=ia.ALL),
                iaa.Affine(scale={'x': (1.0, 1.1)}, rotate=(-10, 10), shear=(-15, 15), order=ia.ALL, cval=ia.ALL, mode=ia.ALL, fit_output=True)
            ]),
            iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, 2.0)),
                iaa.MotionBlur(k=3)
            ]),
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255)),
            iaa.JpegCompression(compression=(0, 75)),
            iaa.Crop(percent=0.01),
            iaa.Resize({"height": 64, "width": "keep-aspect-ratio"}),
            iaa.CenterCropToFixedSize(height=64, width=64),
            iaa.PadToFixedSize(width=64, height=64, pad_mode=ia.ALL)
        ])

    
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image, mask = Image.open(self.dir + '/' + image), Image.open(self.dir + '/' + mask)

        image, mask = np.array(image), np.array(mask)
        
        mask = HeatmapsOnImage.from_uint8(mask, shape=image.shape)        
        image, mask = self.transform(image=image, heatmaps=mask)
        
        image, mask = to_tensor(image), to_tensor(mask.get_arr())
        return (image, mask)

if __name__ == '__main__':
    dataset = CustomDataset('datasets/test/no_label')
    for i, (image, mask) in enumerate(dataset):
        image, mask = to_pil_image(image), to_pil_image(mask)
        savedir = 'datasets/test/dataloader'
        mask.save('%s/%d_mask.tiff' % (savedir, i))
        image.save('%s/%d.tiff' % (savedir, i))