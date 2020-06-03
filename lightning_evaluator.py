import torch
from lightning_model import Net
from PIL import Image
import sys
from torchvision import datasets, transforms, utils
from torchvision.transforms.functional import to_tensor
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import torch.nn.functional as F

model = Net.load_from_checkpoint('saves/main/epoch=61.ckpt')
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(63),
    transforms.ToTensor()
])

device = torch.device('cpu')

def classify(image):
    inputs = transform(image).unsqueeze_(1).to(device)
    with torch.no_grad():
        output = F.softmax(model(inputs)).tolist()
        return output[0]

if __name__ == '__main__':
    print(classify(Image.open(sys.argv[1])))
else:
    device = torch.device('cuda')
    model = model.to(device)