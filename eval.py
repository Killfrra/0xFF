import sys
import torch
from model import SqueezeNet
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from PIL.ImageOps import autocontrast

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SqueezeNet(42).to(device)
model.load_state_dict(torch.load('saves/squeezenet_42c_86acc'))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(autocontrast),
    transforms.Resize(127),
    transforms.ToTensor()
])

def classify(image):
    inputs = transform(image).unsqueeze_(1).to(device)
    with torch.no_grad():
        output = model(inputs)
        _, indices = output.topk(k=5)
        return indices.tolist()[0]

if __name__ == '__main__':
    print(classify(Image.open(sys.argv[1])))