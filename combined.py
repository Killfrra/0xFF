import os
import sys
import signal
sigint = False
def sigint_handler(number, frame):
    global sigint
    sigint = True
signal.signal(signal.SIGINT, sigint_handler)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import common
from PIL import Image

output_folder = 'output'
savefile = output_folder + '/combined.tar'
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

ngpu = 1
niter = 100
batch_size = 128
learning_rate = 1e-2
workers = 5

imageSize = 105

ndf = 64
nc = 1

class DeepFont(nn.Module):
    MODE_AUTOENCODER = 0
    MODE_DEEPFONT = 1
    def __init__(self, ngpu):
        super(DeepFont, self).__init__()
        self.ngpu = ngpu
        self.mode = self.MODE_AUTOENCODER
        self.lru = nn.LeakyReLU(0.2, inplace=True)
        # Encoder
        # conv2d output size = (W - K + 2P) / S + 1
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=11, stride=2)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.pool1 = nn.MaxPool2d(2, return_indices=True)
        self.conv2 = nn.Conv2d(in_channels=ndf, out_channels=2*ndf, kernel_size=1, stride=1)
        # Decoder
        self.deconv1 = nn.ConvTranspose2d(in_channels=ndf*2, out_channels=ndf, kernel_size=1, stride=1)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=ndf, out_channels=nc, kernel_size=11, stride=2)
        # Undefined
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.pool2 = nn.MaxPool2d(2)
        # Classificator
        self.conv3 = nn.Conv2d(in_channels=2*ndf, out_channels=4*ndf, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=4*ndf, out_channels=4*ndf, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(in_channels=4*ndf, out_channels=4*ndf, kernel_size=1, stride=1)

        self.fc1 = nn.Linear(in_features=12 * 12 * 256, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=8)
        self.fc3 = nn.Linear(in_features=8, out_features=5)
        #self.sig = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        """
        x = self.lru(self.conv1(input))
        x = self.bn1(x)
        size = x.size()
        x, indices = self.pool1(x)
        x = self.lru(self.conv2(x))

        if self.mode == self.MODE_AUTOENCODER:
            x = self.lru(self.deconv1(x))
            x = self.unpool1(x, indices, size)
            x = self.lru(self.deconv2(x))
        else:
            x = self.bn2(x)
            x = self.pool2(x)

            x = self.lru(self.conv3(x))
            x = self.lru(self.conv4(x))
            x = self.lru(self.conv5(x))

            x = x.view(-1, 12 * 12 * 256)
            
            x = self.lru(self.fc1(x))
            x = self.lru(self.fc2(x))
            x = self.fc3(x)
        
        return x

model = DeepFont(ngpu).to(device)
"""
img = torch.empty((128, 1, 105, 105), dtype=torch.float)
#print(model)
output = model(img)
print(output.size())
exit()
"""
mode = DeepFont.MODE_AUTOENCODER
training = False
epoch = [0, 0]
optimizer = [
    torch.optim.Adam(model.parameters(), learning_rate),
    optim.SGD([value for (key, value) in model.named_parameters() if not (key.startswith('conv1') or key.startswith('conv2'))], learning_rate, momentum=0.9) #?
]
criterion = None

def prefix(mode):
    if mode:
        return 'df_'
    return 'ae_'

def init_model():
    global mode, epoch, optimizer, criterion
    model.mode = mode

    if os.path.isfile(savefile):
        checkpoint = torch.load(savefile)
        epoch[0] = checkpoint['ae_epoch']
        epoch[1] = checkpoint['df_epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer[0].load_state_dict(checkpoint['ae_optimizer_state_dict'])
        optimizer[1].load_state_dict(checkpoint['df_optimizer_state_dict'])

    if training:
        if mode == DeepFont.MODE_AUTOENCODER:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
            model.conv1.weight.requires_grad = False
            model.conv1.bias.requires_grad = False
            model.conv2.weight.requires_grad = False
            model.conv2.bias.requires_grad = False
        model.train()
    else:
        model.eval()

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(imageSize),
    transforms.CenterCrop(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def loss_func(inputs, outputs, labels):
    global mode, criterion
    if mode == DeepFont.MODE_DEEPFONT:
        return criterion(outputs, labels)
    else:
        return criterion(outputs, inputs)

def train(dataroot):
    global mode, epoch, optimizer, criterion

    dataset = dset.ImageFolder(root=dataroot, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(workers))

    while epoch[mode] < niter:  # loop over the dataset multiple times
        if sigint: break
        epoch[mode] += 1

        total_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)

            if mode == DeepFont.MODE_DEEPFONT:
                labels = data[1].to(device)

            # zero the parameter gradients
            optimizer[mode].zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            if mode == DeepFont.MODE_DEEPFONT:
                #print(outputs.size(), labels.size())
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, inputs)
            loss.backward()
            optimizer[mode].step()

            # print statistics
            total_loss += loss.item()

        torch.save({
            'ae_epoch': epoch[0],
            'df_epoch': epoch[1],
            'model_state_dict': model.state_dict(),
            'ae_optimizer_state_dict': optimizer[0].state_dict(),
            'df_optimizer_state_dict': optimizer[1].state_dict()
        }, savefile)
        
        print('[%d, %2d] loss: %.3f' % (epoch[mode], i + 1, total_loss / len(dataloader)))

        if mode == DeepFont.MODE_AUTOENCODER:
            vutils.save_image(inputs, '%s/input_samples_%03d.png' % (output_folder, epoch[mode]), normalize=True)
            vutils.save_image(outputs, '%s/output_epoch_%03d.png' % (output_folder, epoch[mode]), normalize=True)

def test_on_single_image(path):
    model.eval()
    inputs = preprocess(Image.open(path)).unsqueeze_(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        print(torch.max(outputs.data, 1))

def test():

    dataset = dset.ImageFolder(root='datasets/top5_synth_test', transform=preprocess)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=int(workers))

    model.eval()

    class_count = len(dataset.classes)
    class_correct = list(0. for _ in range(class_count))
    class_total = list(0. for _ in range(class_count))
    with torch.no_grad():
        for data in testloader:

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            i = 0
            for label in labels:
                class_correct[label] += c[i].item()
                class_total[label] += 1
                i += 1

    for i in range(class_count):
        print('Accuracy of %17s : %2d %%' % (dataset.classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    if sys.argv[1] == 'df':
        mode = DeepFont.MODE_DEEPFONT
        training = sys.argv[2] == 'train'
        init_model()
        if training:
            train('datasets/top5_synth_train')
        elif sys.argv[2] == 'test':
            test()
        elif sys.argv[2] == 'image':
            test_on_single_image(sys.argv[3])
        else:
            sys.exit('unknown operation')
    elif sys.argv[1] == 'ae':
        mode = DeepFont.MODE_AUTOENCODER
        training = True
        init_model()
        train('datasets/top5_real+synth')
    else:
        sys.exit('unknown mode')
