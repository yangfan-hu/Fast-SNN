'''
AlexNet for cifar in pytorch
Reference:
ImageNet Classification with Deep Convolutional Neural Networks NIPS2012

'''
import torch
import torch.nn as nn
from models.quant_layer import QuantConv2d, QuantLinear, QuantReLU, first_conv, last_fc
from models.spiking import Spiking, last_Spiking, IF

class Dummy(nn.Module):
    def __init__(self, block):
        super(Dummy, self).__init__()
        self.block = block
        self.idem = False
    def forward(self, x):
        if self.idem:
            return x
        return self.block(x)


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, float=False) -> None:
        super(AlexNet, self).__init__()
        if float:
            self.layer1 = Dummy(nn.Sequential(nn.Conv2d(3, 96, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(96),
                                              nn.ReLU(inplace=True)))
            self.layer2 = Dummy(nn.Sequential(nn.Conv2d(96, 256, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(256),
                                              nn.ReLU(inplace=True)))
            self.layer3 = Dummy(nn.Sequential(nn.Conv2d(256, 384, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(384),
                                              nn.ReLU(inplace=True)))
            self.layer4 = Dummy(nn.Sequential(nn.Conv2d(384, 384, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(384),
                                              nn.ReLU(inplace=True)))	
            self.layer5 = Dummy(nn.Sequential(nn.Conv2d(384, 256, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(256),
                                              nn.ReLU(inplace=True)))	
            self.layer6 = Dummy(nn.Sequential(nn.Linear(4*4*256, 2048),
                                              nn.BatchNorm1d(2048),
                                              nn.ReLU(inplace=True)))     
            self.layer7 = Dummy(nn.Linear(2048, 10))
            self.flat = Dummy(nn.Flatten(1))            
        else:
            self.layer1 = Dummy(nn.Sequential(first_conv(3, 96, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(96),
                                              QuantReLU(inplace=True)))
            self.layer2 = Dummy(nn.Sequential(QuantConv2d(96, 256, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(256),
                                              QuantReLU(inplace=True)))
            self.layer3 = Dummy(nn.Sequential(QuantConv2d(256, 384, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(384),
                                              QuantReLU(inplace=True)))
            self.layer4 = Dummy(nn.Sequential(QuantConv2d(384, 384, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(384),
                                              QuantReLU(inplace=True)))	
            self.layer5 = Dummy(nn.Sequential(QuantConv2d(384, 256, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(256),
                                              QuantReLU(inplace=True)))	
            self.layer6 = Dummy(nn.Sequential(QuantLinear(4*4*256, 2048),
                                              nn.BatchNorm1d(2048),
                                              QuantReLU(inplace=True)))
        
            self.layer7 = Dummy(last_fc(2048, 10))
            self.flat = Dummy(nn.Flatten(1))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flat(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()


class S_AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, T: int = 3) -> None:
        super(S_AlexNet, self).__init__()


        self.layer1 = Spiking(nn.Sequential(first_conv(3, 96, 3, stride=1, padding=1),
                                            nn.BatchNorm2d(96),
                                            IF()), T)
        self.layer2 = Spiking(nn.Sequential(QuantConv2d(96, 256, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(256),
                                            IF()), T)
        self.layer3 = Spiking(nn.Sequential(QuantConv2d(256, 384, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(384),
                                            IF()), T)
        self.layer4 = Spiking(nn.Sequential(QuantConv2d(384, 384, 3, stride=1, padding=1),
                                            nn.BatchNorm2d(384),
                                            IF()), T)	
        self.layer5 = Spiking(nn.Sequential(QuantConv2d(384, 256, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(256),
                                            IF()), T)	
        self.layer6 = Spiking(nn.Sequential(QuantLinear(4*4*256, 2048),
                                            nn.BatchNorm1d(2048),
                                            IF()), T)
        self.layer7 = last_Spiking(last_fc(2048, 10),T)
        self.flat = Dummy(nn.Flatten(2))
        
        self.layer1.is_first = True            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flat(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()
