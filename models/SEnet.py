import torch
import torch.nn as nn
import torch.nn.functional as F


# ResNet-50 + SE module
class Bottleneck_fine(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_fine, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        
        # SE Module
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(planes * self.expansion, planes // self.expansion, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(planes // self.expansion, planes * self.expansion, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()
    
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=False)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # SE Module
        out1 = self.global_pool(out)
        out1 = self.conv_down(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up(out1)
        out1 = self.sig(out1)
        
        # out += self.shortcut(x)
        out = out1 * out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet_fine(nn.Module):
    def __init__(self, block, num_blocks, num_classes=38):
        super(ResNet_fine, self).__init__()
        self.in_planes = 64
        
        self.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pool0 = nn.MaxPool2d(3, stride=2)
        
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool0(self.conv0(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SEnet():
    return ResNet_fine(Bottleneck_fine, [3,4,6,3], num_classes=2)