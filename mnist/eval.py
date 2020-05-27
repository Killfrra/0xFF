import torch
from model import Net
from PIL import Image
import sys
from torchvision.transforms.functional import to_tensor
from preprocessor import process_image
from torch.nn.functional import softmax

model = Net()

checkpoint = torch.load('saves/mnist_cnn_epoch_14.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

def classify(image):
    crops = process_image(image)
    inputs = torch.cat([ to_tensor(crop) for crop in crops ], dim=0)
    inputs.unsqueeze_(1)

    with torch.no_grad():
        output = softmax(model(inputs).sum(dim=0), dim=0).tolist()
        return output

if __name__ == '__main__':
    image = Image.open(sys.argv[1])
    print(classify(image))