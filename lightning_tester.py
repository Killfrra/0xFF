import torch
from lightning_model import Net
from PIL import Image
import sys
from torchvision import datasets, transforms, utils
from torchvision.transforms.functional import to_tensor
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

seq = iaa.Sequential([
    iaa.Sometimes(0.5, [
        iaa.PerspectiveTransform(keep_size=False, cval=ia.ALL, mode=ia.ALL),
        #iaa.PiecewiseAffine(cval=ia.ALL, mode=ia.ALL),
        #iaa.ElasticTransformation(alpha=(0, 0.25), sigma=(0, 0.05)),
        iaa.Affine(scale={'x': (1.0, 1.1)}, rotate=(-10, 10), shear=(-15, 15), order=ia.ALL, cval=ia.ALL, mode=ia.ALL, fit_output=True)
    ]),
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0, 2.0)),
        iaa.MotionBlur(k=3),
        #iaa.imgcorruptlike.GlassBlur(severity=1),
        #iaa.imgcorruptlike.DefocusBlur(severity=1),
        #iaa.imgcorruptlike.Pixelate(severity=1)
    ]),
    iaa.OneOf([
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255)),
        #iaa.imgcorruptlike.ShotNoise(severity=1)
    ]),
    iaa.JpegCompression(compression=(0, 75)),
    iaa.Crop(percent=0.01),
    iaa.Resize({"height": 63, "width": "keep-aspect-ratio"}, interpolation=ia.ALL),
    iaa.CropToFixedSize(63, 252),
    iaa.PadToFixedSize(63, 63, pad_mode=ia.ALL, pad_cval=(0, 255))
])

def unnamed(image):
    image = np.array(image)
    return seq(image=image)

hparams = argparse.Namespace()
hparams.batch_size = 1

model = Net.load_from_checkpoint('saves/main/epoch=61.ckpt')
model.eval()

out_dir = 'mnist/ram/bad'

transform = transforms.Compose([
    transforms.Grayscale(),
    #transforms.Lambda(unnamed),
    transforms.Resize(63),
    #transforms.RandomCrop(63),
    #transforms.Lambda(lambda img: transforms.functional.resize(img, 63) if min(img.size[0], img.size[1]) < 63 else img),
    transforms.ToTensor()
])

def test():
    device = torch.device('cuda')
    model = model.to(device)
    

    test_loader = DataLoader(
        datasets.ImageFolder('ram/mini_ru_test', transform),
        batch_size=hparams.batch_size, shuffle=False, num_workers=6, pin_memory=True
    )
    
    
    test_loss = 0
    correct = 0
    classified = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            classified += data.size(0)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= classified
    accuracy = correct / classified

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, classified,
        100. * accuracy))

    """
    i = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            mask = pred.ne(target.view_as(pred)).view(-1)
            selection = data[mask]
            if selection.size(0) > 0:
                utils.save_image(selection, '%s/%d.tiff' % (out_dir, i), normalize=True)
                i += 1
    """

def classify(image):
    #crops = process_image(image)
    #crops = [image]

    #for i, crop in enumerate(crops):
    #    crop.save('%d.tiff' % i)

    #inputs = torch.cat([ to_tensor(crop) for crop in crops ], dim=0)
    inputs = transform(image)
    inputs.unsqueeze_(1)

    with torch.no_grad():
        output = model.classifier(inputs)
        #utils.save_image(output, 'mnist/ram/eval.tiff', normalize=True)
        output = model(inputs).tolist()
        #output = softmax(model(inputs).sum(dim=0), dim=0).tolist()
        #print(output.size())
        #for i in range(output.size(1)):
        #    utils.save_image(output[0][i], 'mnist/ram/layer_%d.tiff' % i, normalize=True)
        return output

if __name__ == '__main__':
    device = torch.device('cpu')
    model = model.to(device)
    print(classify(Image.open(sys.argv[1])))
    #test()
