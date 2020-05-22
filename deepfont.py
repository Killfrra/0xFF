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

dataroot = 'datasets/top5_synth'
output_folder = 'output'
ae_savefile = output_folder + '/ae.tar'
df_savefile = output_folder + '/df.tar'
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

ngpu = 1
niter = 100
batch_size = 256
learning_rate = 1e-2
workers = 5

imageSize = 105

ndf = 64
nc = 1

class DeepFont(nn.Module):
    def __init__(self, ngpu):
        super(DeepFont, self).__init__()
        self.ngpu = ngpu
        self.lru = nn.LeakyReLU(0.2, inplace=True)
        # Encoder
        # conv2d output size = (W - K + 2P) / S + 1
        self.conv1 = nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=11, stride=2)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=ndf, out_channels=2*ndf, kernel_size=1, stride=1)
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
        x = self.pool1(x)
        x = self.lru(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.lru(self.conv3(x))
        x = self.lru(self.conv4(x))
        x = self.lru(self.conv5(x))

        x = x.view(-1, 12 * 12 * 256)
        
        x = self.lru(self.fc1(x))
        x = self.lru(self.fc2(x))
        x = self.fc3(x)

        #x = self.sig(x)
        #x = self.softmax(x)
        
        return x

model = DeepFont(ngpu).to(device)

"""
img = torch.empty((128, 1, 105, 105), dtype=torch.float)
#print(model)
output = model(img)
print(output.size())
exit()
"""

ae_checkpoint = torch.load(ae_savefile)
missing_keys, unexpected_keys = model.load_state_dict(ae_checkpoint['model_state_dict'], strict=False)
model.conv1.weight.requires_grad = False
model.conv1.bias.requires_grad = False
model.conv2.weight.requires_grad = False
model.conv2.bias.requires_grad = False
#print('missing keys:', missing_keys)
#print('unexpected keys:', unexpected_keys)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda m: m.requires_grad, model.parameters()), learning_rate, momentum=0.9) #?

epoch, total_loss = common.load_checkpoint(df_savefile, model, optimizer)

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(imageSize),
    transforms.CenterCrop(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def train():
    global epoch

    dataset = dset.ImageFolder(root=dataroot, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(workers))
    
    model.train()

    while epoch < niter:  # loop over the dataset multiple times
        epoch += 1

        total_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            total_loss += loss.item()

            if sigint: break
        
        common.save_checkpoint(df_savefile, model, optimizer, epoch, total_loss)
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss / len(dataloader)))

        if sigint: break

def test_on_single_image():
    model.eval()
    inputs = preprocess(Image.open('test.tiff')).unsqueeze_(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        print(torch.max(outputs.data, 1))

def test():

    dataset = dset.ImageFolder(root='datasets/top5_synth_test', transform=preprocess)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=int(workers))

    print(dataset.classes)

    model.eval()

    class_count = len(dataset.classes)
    class_correct = list(0. for i in range(class_count))
    class_total = list(0. for i in range(class_count))
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

    for i in range(5):
        print('Accuracy of %d : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        test()