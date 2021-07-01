import torch
import torch.nn as nn
import torch.nn.functional as F


# ResNet-50 + non-local module
'''
Use non local layer:  4
Use non local layer:  6
input size:  torch.Size([1, 1, 224, 224])
conv1 size:  torch.Size([1, 64, 112, 112])
pool size:  torch.Size([1, 64, 56, 56])
layer 1:  torch.Size([1, 256, 56, 56])
layer 2:  torch.Size([1, 512, 28, 28])
layer 3:  torch.Size([1, 1024, 28, 28])
layer 4:  torch.Size([1, 2048, 14, 14])
'''

class NLBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(NLBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if bn_layer:
            self.W_z = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    nn.BatchNorm2d(self.in_channels)
                )
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)
        
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )
            
    def forward(self, x):
        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
        
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)
        
        concat = torch.cat([theta_x, phi_x], dim=1)
        f = self.W_f(concat)
        f = f.view(f.size(0), f.size(2), f.size(3))

        N = f.size(-1)
        f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        z = W_y + x

        return z

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResBlock, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):  # [3, 4, 6, 3]
    def __init__(self, block, layers, num_classes, non_local):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=(2, 2), padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)
        
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2, non_local=non_local)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=1, non_local=non_local)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_blocks, out_channels, stride, non_local=False):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_channels*4))

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4

        if not non_local:
            for i in range(num_blocks - 1):
                layers.append(block(self.in_channels, out_channels))

        else:
#             print("Use non local layer: ", num_blocks)
            for i in range(num_blocks - 3):
                layers.append(block(self.in_channels, out_channels))
            layers.append(NLBlock(self.in_channels, out_channels))
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def NLnet(num_classes=2, non_local=True):
    return ResNet(ResBlock, [3, 4, 6, 3], num_classes, non_local=non_local)

