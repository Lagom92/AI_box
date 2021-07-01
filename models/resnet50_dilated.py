import torch.nn as nn
import torch.nn.functional as F

# ResNet-50 + Dilated Conv
# Reference: http://www.mva-org.jp/Proceedings/2017USB/papers/15-11.pdf
class Bottleneck_coarse(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super(Bottleneck_coarse, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_coarse(nn.Module):
    def __init__(self, block, num_blocks, num_classes=38):
        super(ResNet_coarse, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1, dilation=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1, dilation=4)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=1, dilation=8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64*block.expansion, num_classes)
        

    def _make_layer(self, block, planes, num_blocks, stride, dilation=1):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dilation=dilation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

    
# 1차 네트워크
def ResNet_dilated():
    return ResNet_coarse(Bottleneck_coarse, [3,4,6,3], num_classes=38)