import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 28x28x1
        ndf = 11
        self.conv1 = nn.Conv2d(1, ndf, 3, 1, 1)
        self.conv2 = nn.Conv2d(ndf, 2*ndf, 3, 1, 1)
        self.conv3 = nn.Conv2d(2*ndf, 4*ndf, 3, 1, 1)
        #self.conv4 = nn.Conv2d(4*ndf, 8*ndf, 3, 1)
        self.fc1 = nn.Linear(12*12*4*ndf, 16*ndf)
        self.fc2 = nn.Linear(16*ndf, 16*ndf)
        self.fc3 = nn.Linear(16*ndf, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x), True)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x), True)
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x), True)
        #x = F.relu(self.conv4(x), True)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x), True)
        #x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x), True)
        #x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x #F.log_softmax(x, dim=1)

"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [128, 32, 26, 26]             320
            Conv2d-2          [128, 64, 24, 24]          18,496
         Dropout2d-3          [128, 64, 12, 12]               0
            Linear-4                 [128, 128]       1,179,776
         Dropout2d-5                 [128, 128]               0
            Linear-6                  [128, 10]           1,290
================================================================
Total params: 1,199,882
Trainable params: 1,199,882
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.38
Forward/backward pass size (MB): 66.38
Params size (MB): 4.58
Estimated Total Size (MB): 71.34
----------------------------------------------------------------

"""

if __name__ == '__main__':
    model = Net()
    summary(model, (1, 96, 96), 128, 'cpu')