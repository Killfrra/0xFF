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
import torchvision.utils as vutils
import torch.nn.functional as F

from torchsummary import summary

import common

dataroot = 'datasets/top5_real+synth'
output_folder = 'output'
savefile = output_folder + '/en.tar'
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

ngpu = 1
niter = 100
batch_size = 128
learning_rate = 1e-2
workers = 5

imageSize = 97
latentDim = 128

class Autoencoder(nn.Module):
    def __init__(self, ngpu):
        super(Autoencoder, self).__init__()
        self.ngpu = ngpu
        self.enable_decoder = True
        # conv2d output size = (W - K + 2P) / S + 1
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            #nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            #nn.MaxPool2d(2)
        )
        self.dence_encoder = nn.Linear(73728, latentDim)
        self.dence_decoder = nn.Linear(latentDim, 73728)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            #nn.MaxUnpool2d(2),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.BatchNorm2d(nc)
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        """

        x = self.encoder(x)
        if self.enable_decoder:
            size = x.size()
            x = x.reshape(size[0], -1)
            x = self.dence_encoder(x)
            x = self.dence_decoder(x)
            x = x.reshape(size)
            x = self.decoder(x)

        return x

model = Autoencoder(ngpu).to(device)
#summary(model, (1, imageSize, imageSize), batch_size); exit()

epoch = total_loss = 0
optimizer = None

def train():
    global epoch, total_loss, optimizer

    criterion = nn.MSELoss()
    model.train()

    dataset = dset.ImageFolder(dataroot, common.transform(imageSize))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=workers)

    while epoch < niter:
        if sigint: break
        epoch += 1

        total_loss = 0.0
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
        print('[%d/%d]      Loss: %.4f' % (epoch, niter, total_loss))

        common.save_checkpoint(savefile, model, optimizer, epoch, total_loss)

if __name__ == '__main__':
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    epoch, total_loss = common.load_checkpoint(savefile, model, optimizer)
    train()
else:
    epoch, total_loss = common.load_checkpoint(savefile, model)
    model.enable_decoder = False
    model.eval()
