from __future__ import print_function

from torch import nn
from torch.autograd import Variable

import torch
from collections import OrderedDict
import torchvision.models as models

import re

import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNet', 'resnet20', 'resnet56']

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        try:
            init.kaiming_normal_(m.weight)
        except AttributeError:
            init.kaiming_normal(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])

def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))

## network
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # this is LeNet-5 by the way
        self.feature_extractor = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5,5), padding=2)),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2))),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2))),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))
        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        out = self.feature_extractor(img)
        if self.fc is not None:
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        return out


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        self.mean = mean
        self.std = std
    def forward(self, img):
        mean_var = Variable(img.data.new(self.mean).view(1, 3, 1, 1))
        std_var = Variable(img.data.new(self.std).view(1, 3, 1, 1))
        img = (img - mean_var) / std_var
        return img


def load_classifier(static_classifier_name):
    normalizer = None
    if static_classifier_name == 'OurLeNetMNIST':
        model = LeNet()
        state_dict = torch.load('evaluation/mnist_lenet_ours/Classifier.pth')

    elif static_classifier_name == 'OurResNet20CIFAR10':
        model = resnet20()
        bad_state_dict = torch.load('evaluation/cifar_resnet20_ours/Classifier.pth')
        correct_state_dict = {re.sub(r'^.*feature_extractor\.', '', k): v for k, v in
                              bad_state_dict.items()}
        state_dict = correct_state_dict
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalizer = Normalizer(mean, std)

    elif static_classifier_name == 'OurResNet18TinyImagenet':
        model = models.resnet18(False)
        #Finetune Final few layers to adjust for tiny imagenet input
        model.conv1 = nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        model.maxpool = nn.Sequential()
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, 200)

        pretrained_dict = torch.load('evaluation/tinynet_resnet18_ours/Classifier.pth')
        state_dict = {re.sub(r'^.*feature_extractor\.', '', k): v for k, v in
                              pretrained_dict.items()}
        mean = [0.4802, 0.4481, 0.3975]
        std = [0.2302, 0.2265, 0.2262]
        normalizer = Normalizer(mean, std)

    starts_with_module = False
    for key in state_dict.keys():
        if key.startswith('module.'):
            starts_with_module = True
            break
    if starts_with_module:
        correct_classifier_state = {k[7:]: v for k, v in
                                   state_dict.items()}
    else:
        correct_classifier_state = state_dict


    model.load_state_dict(correct_classifier_state)
    if normalizer:
        model = nn.Sequential(normalizer, model)
    model.eval()
    model = model.cuda()
    return model
