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

device = torch.device('cpu')
hparams = argparse.Namespace()
hparams.batch_size = 1

model = Net.load_from_checkpoint('mnist/saves/epoch=9_v4.ckpt').to(device)
model.eval()

out_dir = 'mnist/ram/bad'

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(63),
    #transforms.RandomCrop(63),
    #transforms.Lambda(lambda img: transforms.functional.resize(img, 63) if min(img.size[0], img.size[1]) < 63 else img),
    transforms.ToTensor()
])

def test():

    test_loader = DataLoader(
        datasets.ImageFolder('mnist/ram/mini_ru_test', transform),
        batch_size=hparams.batch_size, shuffle=False, num_workers=6, pin_memory=True
    )
    
    """
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
    #"""

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
        #output = model(inputs).tolist()
        #output = softmax(model(inputs).sum(dim=0), dim=0).tolist()
        print(output.size())
        for i in range(output.size(1)):
            utils.save_image(output[0][i], 'mnist/ram/layer_%d.tiff' % i, normalize=True)

if __name__ == '__main__':
    print(classify(Image.open(sys.argv[1])))
    #test()