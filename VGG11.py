import torch.nn as nn

class VGG11(nn.Module):

    def __init__(self):
        super(VGG11, self).__init__()
        self.conv = nn.Sequential(
            #64*64*3
            nn.Conv2d(3, 64, kernel_size=2, stride=1, padding=0),
            #63*63*64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #31*31*64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #31*31*128
            nn.MaxPool2d(kernel_size=2, stride=2),
            #15*15*128
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            #15*15*256
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #15*15*256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #7*7*256
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            #7*7*512
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            #7*7*512
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #3*3*512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            #3*3*512
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #1*1*512
        )
        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1*1*512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1*1*512)
        x = self.fc(x)
        return x