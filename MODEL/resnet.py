
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def _make_layer(self, block, channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:  # the first stride
                layers.append(block(self.inchannel, channels, stride))
            else:  # set all stride = 1
                layers.append(block(channels, channels, 1))
                self.inchannel = channels
        return nn.Sequential(*layers)


    def __init__(self, ResBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_1 = self._make_layer(ResBlock, 64, 2, 1)
        self.layer_2 = self._make_layer(ResBlock, 128, 2, 2)
        self.layer_3 = self._make_layer(ResBlock, 256, 2, 2)
        self.layer_4 = self._make_layer(ResBlock, 512, 2, 2)

        self.fc = nn.Linear(512, num_classes)




    def forward(self, x):
        output= self.conv1(x)

        output = self.layer_1(output)
        output = self.layer_2(output)
        output = self.layer_3(output)
        output = self.layer_4(output)

        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def ResNet18():

    return ResNet(ResBlock)