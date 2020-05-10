'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            #32*32*3
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=0),
            #30*30*96
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #15*15*96
            nn.Conv2d(96, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #15*15*256
            nn.MaxPool2d(kernel_size=2, stride=2),
            #7*7*256
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            #7*7*384
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            #7*7*384
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            #7*7*256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride= 2),
            #3*3*256
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc(x)
        return x
