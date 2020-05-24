import torch
from PIL import Image
import torchvision.utils as vutils
from ignite.handlers import Checkpoint
from model import Autoencoder
import torchvision.transforms as transforms

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")
ngpu = 1

model = Autoencoder(ngpu).to(device)

last_checkpoint = 'output/autoencoder/checkpoint_loss=-0.045623716315243215.pth'

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

inputs = data_transform(Image.open('datasets/preprocessed_labeled_real/ClearfaceGothicLTStd-Roman/22674.tiff')).unsqueeze_(0).to(device)

model.eval()
with torch.no_grad():
    inputs = inputs.to(device)
    outputs = model(inputs)

vutils.save_image(outputs, 'ae_out.tiff', normalize=True)
