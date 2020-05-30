import torch
from model import Net
from PIL import Image
import sys
from torchvision import datasets, transforms, utils
from torchvision.transforms.functional import to_tensor
from preprocessor import process_image
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda')
model = Net(False).to(device)

checkpoint = torch.load('saves/mnist_cnn_epoch_12.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

out_dir = 'ram/bad'

transform = transforms.Compose([
    transforms.Grayscale(),
    #transforms.Resize(63),
    transforms.Lambda(lambda img: transforms.functional.resize(img, 63) if min(img.size[0], img.size[1]) < 63 else img),
    transforms.ToTensor(),
])

def test():

    test_loader = DataLoader(
        datasets.ImageFolder('ram/mini_ru_test', transform),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )
    
    model.eval()
    test_loss = 0
    correct = 0
    classified = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            try:
                output = model(data)
            except RuntimeError as e:
                #print(e)
                continue
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
    model.eval()
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
        #output = model(inputs).tolist()
        output = softmax(model(inputs).sum(dim=0), dim=0).tolist()
        return output

if __name__ == '__main__':
    #print(classify(Image.open(sys.argv[1])))
    test()