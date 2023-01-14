'''
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import torch
import torch.nn as nn
# import math
# from models.quant_layer import QuantConv2d, first_conv, last_fc, QuantReLU
from models.quant_layer import QuantReLU

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Dummy(nn.Module):
    def __init__(self, block):
        super(Dummy, self).__init__()
        self.block = block
        self.idem = False
    def forward(self, x):
        if self.idem:
            return x
        return self.block(x)

class Oneway(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None):
        super(Oneway, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu 
        self.idem = False
    def forward(self, x):
        if self.idem:
            return x
        x = self.conv(x)
        x = self.bn(x) 
        x = self.relu(x)
        return x

class Twoways(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None, downsample=None):
        super(Twoways, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu 
        self.downsample = downsample
        self.idem = False
    def forward(self, x, identity):
        if self.idem:
            return x
        x = self.conv(x)
        x = self.bn(x) 
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x   


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bit=32):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.idem = False
        self.inter = False
        

        self.part1 = Oneway(conv3x3(inplanes, planes, stride),
                            norm_layer(planes),
                            QuantReLU(inplace=True, bit=bit)) 
        
        self.part2 = Twoways(conv3x3(planes, planes),
                             norm_layer(planes),
                             QuantReLU(inplace=True, bit=bit), downsample)            

    def forward(self, x):
        if self.idem:
            return x
        identity = x
        out = self.part1(x)
        if self.inter:
            return out
        out = self.part2(out, identity)
        return out
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, bit=32):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.bit = bit
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.layer0 = Dummy(nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(64),
                                          QuantReLU(inplace=True, bit=bit)))
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = Dummy(nn.Sequential(nn.AvgPool2d(8, stride=1),
                                          nn.Flatten(1),
                                          nn.Linear(512, num_classes)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, bit=self.bit))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, bit=self.bit))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)  
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantReLU):
                m.show_params()


class Spiking(nn.Module):
    def __init__(self, block, T):
        super(Spiking, self).__init__()
        self.block = block
        self.T = T
        self.is_first = False
        self.idem = False
        self.sign = True
        
    def forward(self, x):
        if self.idem:
            return x
        
        ###initialize membrane to half threshold
        threshold = self.block[2].act_alpha.data
        membrane = 0.5 * threshold
        sum_spikes = 0
        
        #prepare charges
        if self.is_first:
            x.unsqueeze_(1)
            x = x.repeat(1, self.T, 1, 1, 1)
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        #integrate charges
        for dt in range(self.T):
            membrane = membrane + x[:,dt]
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)
                
            spikes = membrane >= threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            sum_spikes = sum_spikes + spikes
            
            ###signed spikes###
            if self.sign:
                inhibit = membrane <= -1e-3
                inhibit = inhibit & (sum_spikes > 0)
                membrane[inhibit] = membrane[inhibit] + threshold
                inhibit = inhibit.float()
                sum_spikes = sum_spikes - inhibit
            else:
                inhibit = 0

            spike_train[:,dt] = spikes - inhibit
                
        spike_train = spike_train * threshold
        return spike_train


class Spiking_Oneway(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None, T=0):
        super(Spiking_Oneway, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu 
        self.idem = False
        self.T = T
        self.sign = True
    def forward(self, x):
        if self.idem:
            return x
        ###initialize membrane to half threshold
        threshold = self.relu.act_alpha.data
        membrane = 0.5 * threshold
        sum_spikes = 0
        
        #prepare charges
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.conv(x)
        x = self.bn(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        #integrate charges
        for dt in range(self.T):
            membrane = membrane + x[:,dt]
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)
                
            spikes = membrane >= threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            sum_spikes = sum_spikes + spikes
            
            ###signed spikes###
            if self.sign:
                inhibit = membrane <= -1e-3
                inhibit = inhibit & (sum_spikes > 0)
                membrane[inhibit] = membrane[inhibit] + threshold
                inhibit = inhibit.float()
                sum_spikes = sum_spikes - inhibit
            else:
                inhibit = 0

            spike_train[:,dt] = spikes - inhibit
                
        spike_train = spike_train * threshold
        return spike_train        
        
class Spiking_Twoways(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None, downsample=None, T=0):
        super(Spiking_Twoways, self).__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu 
        self.downsample = downsample
        self.idem = False
        self.T = T
        self.sign = True
    def forward(self, x, identity):
        if self.idem:
            return x
        ###initialize membrane to half threshold
        threshold = self.relu.act_alpha.data
        membrane = 0.5 * threshold
        sum_spikes = 0

        #prepare charges
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        identity = identity.flatten(0, 1)
        x = self.conv(x)
        x = self.bn(x)
        if self.downsample is not None:
            x += self.downsample(identity)
        else:
            x += identity
        
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)

        #integrate charges
        for dt in range(self.T):
            membrane = membrane + x[:,dt]
            if dt == 0:
                spike_train = torch.zeros(membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],device=membrane.device)
                
            spikes = membrane >= threshold
            membrane[spikes] = membrane[spikes] - threshold
            spikes = spikes.float()
            sum_spikes = sum_spikes + spikes
            
            ###signed spikes###
            if self.sign:
                inhibit = membrane <= -1e-3
                inhibit = inhibit & (sum_spikes > 0)
                membrane[inhibit] = membrane[inhibit] + threshold
                inhibit = inhibit.float()
                sum_spikes = sum_spikes - inhibit
            else:
                inhibit = 0

            spike_train[:,dt] = spikes - inhibit
                
        spike_train = spike_train * threshold
        return spike_train  

        
class Spiking_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bit=32):
        super(Spiking_BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.T = 2**bit - 1
        self.idem = False
        self.inter = False
        
        self.part1 = Spiking_Oneway(conv3x3(inplanes, planes, stride),
                                    norm_layer(planes),
                                    IF(), self.T) 
        
        self.part2 = Spiking_Twoways(conv3x3(planes, planes),
                                     norm_layer(planes),
                                     IF(), downsample, self.T)  

    def forward(self, x):
        if self.idem:
            return x
        identity = x
        out = self.part1(x)
        if self.inter:
            return out
        out = self.part2(out, identity)
        return out


class last_Spiking(nn.Module):
    def __init__(self, block, T):
        super(last_Spiking, self).__init__()
        self.block = block
        self.T = T
        self.idem = False
        
    def forward(self, x):
        if self.idem:
            return x
        #prepare charges
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        #integrate charges
        return x.sum(dim=1)

class IF(nn.Module):
    def __init__(self):
        super(IF, self).__init__()
        ###changes threshold to act_alpha
        ###being fleet
        self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x
    
    def extra_repr(self) -> str:
        return 'threshold={:.3f}'.format(self.act_alpha)  


class S_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, bit=4):
        super(S_ResNet, self).__init__()
        self.inplanes = 64
        self.bit = bit
        self.T = 2**self.bit - 1
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        
        self.layer0 = Spiking(nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(64),
                                          IF()), self.T)
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        
        self.layer4 = last_Spiking(nn.Sequential(nn.AvgPool2d(8, stride=1),
                                                 nn.Flatten(1),
                                                 nn.Linear(512, num_classes)), self.T)
        self.layer0.is_first = True

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, bit=self.bit))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, bit=self.bit))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)  
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantReLU):
                m.show_params()

def resnet18(spike=False, **kwargs):
    if spike:
        return S_ResNet(Spiking_BasicBlock, [3, 3 ,2], **kwargs)
    else:
        return ResNet(BasicBlock, [3, 3, 2], **kwargs)


    
