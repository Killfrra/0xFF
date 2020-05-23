import os
import sys
import signal
sigint = False
def sigint_handler(number, frame):
    global sigint
    sigint = True
    print('\n')
signal.signal(signal.SIGINT, sigint_handler)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import torch.nn.functional as F

import common
import encoder

from torchsummary import summary

output_folder = 'output'
savefile = output_folder + '/main.tar'
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

ngpu = 1
niter = 100
batch_size = 128
learning_rate = 1e-2
workers = 5

imageSize = 97

class DeepFont(nn.Module):
    def __init__(self, encoder, ngpu):
        super(DeepFont, self).__init__()
        self.ngpu = ngpu
        self.lru = nn.LeakyReLU(0.2, inplace=True)
        self.encoder = encoder
        # Classificator
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(in_features=24*24*16, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=8)
        self.fc3 = nn.Linear(in_features=8, out_features=5)

    def forward(self, x):
        """
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        """
        x = self.encoder(x)
        x = self.lru(self.conv3(x))
        x = self.lru(self.conv4(x))
        x = self.lru(self.conv5(x))

        x = x.reshape(x.size(0), -1)
        
        x = self.lru(self.fc1(x))
        x = self.lru(self.fc2(x))
        x = self.fc3(x)

        #x = self.softmax(x)
        
        return x

encoder = encoder.model
model = DeepFont(encoder, ngpu).to(device)
#summary(model, (1, imageSize, imageSize), batch_size); exit()

epoch = total_loss = 0
optimizer = None

preprocess = common.transform(imageSize)

def train():
    global sigint, model, epoch, total_loss, optimizer

    dataset = dset.ImageFolder(root='datasets/top5_synth_train', transform=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(workers))
    
    model.train()

    while epoch < niter:  # loop over the dataset multiple times
        if sigint: break
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
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # print statistics
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, niter, i, len(dataloader), loss.item()))

        total_loss = total_loss / float(len(dataloader))
        print('[%d/%d]      Loss: %.4f' % (epoch, niter, total_loss))

        common.save_checkpoint(savefile, model, optimizer, epoch, total_loss)

def test_on_single_image(image):
    inputs = preprocess(image).unsqueeze_(0).to(device)
    classes = ['ForteMTStd', 'GillSansStd', 'MyriadPro-Regular', 'OndineLTStd', 'ScriptMTStd-Bold' ]
    links = [
        'https://www.myfonts.com/fonts/adobe/forte/regular/',
        'https://fontsgeek.com/fonts/Gill-Sans-Std-Regular',
        'https://fontsgeek.com/fonts/Myriad-Pro-Regular',
        'https://www.myfonts.com/fonts/adobe/ondine/regular/',
        'https://www.azfonts.net/load_font/script-mt-bold.html'
    ]
    with torch.no_grad():
        outputs = model(inputs)
        #print(torch.max(outputs.data, 1))
        outputs = F.softmax(outputs.data, 1).tolist()[0]
        for (i, e) in enumerate(outputs):
            outputs[i] = round(e * 100)
        outputs = list(zip(classes, outputs, links))
        outputs.sort(key=lambda tup: tup[1], reverse=True)
        return outputs

def test():
    dataset = dset.ImageFolder(root='datasets/top5_synth_test', transform=preprocess)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=int(workers))

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
    if sys.argv[1] == 'train':
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.SGD(model.parameters(), learning_rate, momentum=0.9) #?
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        epoch, total_loss = common.load_checkpoint(savefile, model, optimizer)
        train()
    else:
        epoch, total_loss = common.load_checkpoint(savefile, model)
        model.eval()
        if sys.argv[1] == 'test':
            test()
        elif sys.argv[1] == 'image':
            print(test_on_single_image(Image.open(sys.argv[2])))
else:
    epoch, total_loss = common.load_checkpoint(savefile, model)
    model.eval()