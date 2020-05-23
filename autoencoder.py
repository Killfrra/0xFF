import os
import signal
sigint = False
def sigint_handler(number, frame):
    global sigint
    sigint = True
signal.signal(signal.SIGINT, sigint_handler)

import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from torchsummary import summary

from PIL.ImageOps import autocontrast

dataroot = 'datasets/top5_real+synth'
output_folder = 'output'
savefile = output_folder + '/ae.tar'
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

ngpu = 1
niter = 100
batch_size = 256
learning_rate = 1e-2
workers = 1

imageSize = 105

nc = 1
ndf = 64

class Autoencoder(nn.Module):
    def __init__(self, ngpu):
        super(Autoencoder, self).__init__()
        self.ngpu = ngpu
        self.lru = nn.LeakyReLU(0.2, inplace=True)
        self.sig = nn.Sigmoid()
        # Encoder
        # conv2d output size = (W - K + 2P) / S + 1
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=11, stride=2)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=3, stride=1)
        
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=ndf*2, out_channels=ndf, kernel_size=3, stride=1)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=ndf, out_channels=nc, kernel_size=11, stride=2)

    def forward(self, input):
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        """
        x = self.lru(self.conv1(input))
        size = x.size()
        x, indices = self.pool1(x)
        x = self.lru(self.conv2(x))
        x = self.lru(self.deconv1(x))
        x = self.unpool1(x, indices, size)
        x = self.sig(self.deconv2(x))

        return x

model = Autoencoder(ngpu).to(device)

#summary(model, (1, imageSize, imageSize), batch_size)
#exit()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

epoch = 0
total_loss = 0
if os.path.isfile(savefile):
    checkpoint = torch.load(savefile)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #total_loss = checkpoint['loss']
    model.train()

transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(imageSize),
    transforms.CenterCrop(imageSize),
    transforms.Lambda(autocontrast),
    transforms.ToTensor(),
    #transforms.Normalize((0.5,), (0.5,)),
])

dataset = dset.ImageFolder(dataroot, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(workers))

#def train():
while epoch < niter:
    epoch += 1
    if sigint: break

    for i, data in enumerate(dataloader, 0):

        img = data[0].to(device)

        optimizer.zero_grad()

        output = model(img)
        loss = criterion(output, img)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        
        print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, niter, i, len(dataloader), loss.item()))

    vutils.save_image(  img, '%s/input_samples_%03d.png' % (output_folder, epoch), normalize=True)
    vutils.save_image(output, '%s/output_epoch_%03d.png' % (output_folder, epoch), normalize=True)
    
    total_loss = total_loss / float(len(dataloader))
    print('TOTAL LOSS %.4f' % total_loss)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss
    }, savefile)

#if __name__ == '__main__':
#    train()
