import os
import torch
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

dataroot = 'VFR_real_test'
outf = 'output'
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

ngpu = 1
niter = 100
batch_size = 128
learning_rate = 1e-2
workers = 6

imageSize = 64

nc=1
ndf = 64

class Autoencoder(nn.Module):
    def __init__(self, ngpu):
        super(Autoencoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # Encoder
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Decoder
            nn.ConvTranspose2d(in_channels=ndf*2, out_channels=ndf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=ndf, out_channels=nc, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, inplace=True)
            #nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

model = Autoencoder(ngpu).to(device)

"""
img = torch.empty((128, 1, 64, 64), dtype=torch.float)
print(img.size())
output = model(img)
print(output.size())

"""

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

file_prefix = 'ae_epoch_'
file_ext = '.pth'

epoch = 0
for filename in sorted(os.listdir(outf), reverse=True):
    if filename.startswith(file_prefix) and filename.endswith(file_ext):
        epoch = int(filename[len(file_prefix):-len(file_ext)])
        model.load_state_dict(torch.load('%s/%s' % (outf, filename)))
        break

dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Grayscale(),
                                transforms.Resize(imageSize),
                                transforms.CenterCrop(imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                            ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=int(workers))

#def train():
while epoch < niter:
    for i, data in enumerate(dataloader, 0):

        img = data[0]
        img= img.to(device)
        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, niter, i, len(dataloader), loss.item()))

        if i % 100 == 0:
            vutils.save_image(img, '%s/input_samples.png' % outf, normalize=True)
            vutils.save_image(output, '%s/output_epoch_%03d.png' % (outf, epoch), normalize=True)

    torch.save(model.state_dict(), '%s/%s%03d%s' % (outf, file_prefix, epoch, file_ext))

#if __name__ == '__main__':
#    train()