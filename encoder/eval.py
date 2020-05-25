import torch
from PIL import Image
import torchvision.utils as vutils
from ignite.handlers import Checkpoint
from model import Autoencoder
import torchvision.transforms as transforms
import os

cuda = False    #cuda = torch.cuda.is_available()
device = 'cpu'  #device = torch.device("cuda:0" if cuda else "cpu")
ngpu = 0        #ngpu = 1

model = Autoencoder(ngpu) #.to(device)

last_checkpoint = 'output/autoencoder/checkpoint_loss=-0.002959001278966386.pth'

checkpoint = torch.load(last_checkpoint)
Checkpoint.load_objects({ 'model': model }, checkpoint)
print(last_checkpoint, 'loaded')

image_size = 96
data_transform = transforms.Compose([
    transforms.Grayscale(),
    #transforms.Resize(image_size),
    #transforms.CenterCrop(image_size),
    #transforms.Lambda(autocontrast),
    transforms.ToTensor()
])

basepath = 'datasets/top60_ru_synth_unlabeled_preprocessed/no_label'
output = 'output/trash'
for image_name in os.listdir(basepath):
    inputs = data_transform(Image.open('%s/%s' % (basepath, image_name))).unsqueeze_(0) #.to(device)
    model.eval()
    with torch.no_grad():
        #inputs = inputs.to(device)
        outputs = model(inputs)

    vutils.save_image(outputs, '%s/%s' % (output, image_name), normalize=True)
