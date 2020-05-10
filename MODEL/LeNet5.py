import torch
import torch.nn as nn

class LeNet_5(nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        self.conv_layer1 = nn.Sequential(
            # the first convolutional layer 3 input channels，6 output channels，kernel size 5*5
            # the image is resize to 32-kernel_size+stride = 28*28
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            # activation function
            nn.ReLU(),
            # 2*2 max pooling,the images are resize to 14*14
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.conv_layer2 = nn.Sequential(
            # the second vonvolutional layer, 6 input channels, 16 ouput channels, kernel size 5*5
            # the image is resize to 14-kernel_size+stride = 10*10
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            # 2*2 max pooling,the images are resize to 5*5
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc = nn.Sequential(
            # three fully connected layers
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        #Generating 1D tensor since nn.Linear()takes only 1-D tensor.
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

