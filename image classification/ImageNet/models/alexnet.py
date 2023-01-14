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
    def __init__(self, num_classes: int = 1000, bit: int = 32) -> None:
        super(AlexNet, self).__init__()
        self.bit = bit
        self.num_layers = 8
        self.flatten_id = 6
        self.layer1 = Dummy(nn.Sequential(first_conv(3, 64, kernel_size=11, stride=4, padding=2),
                                          nn.BatchNorm2d(64),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer2 = Dummy(nn.Sequential(QuantConv2d(64, 192, kernel_size=5, stride=2,padding=1, bit=32),
                                          nn.BatchNorm2d(192),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer3 = Dummy(nn.Sequential(QuantConv2d(192, 384, kernel_size=3, stride=2, padding=0, bit=32),
                                          nn.BatchNorm2d(384),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer4 = Dummy(nn.Sequential(QuantConv2d(384, 256, kernel_size=3, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        self.layer5 = Dummy(nn.Sequential(QuantConv2d(256, 256, kernel_size=3, stride=2, padding=0, bit=32),
                                          nn.BatchNorm2d(256),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        self.layer6 = Dummy(nn.Sequential(QuantLinear(256 * 6 * 6, 4096, bit=32),
                                          nn.BatchNorm1d(4096),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer7 = Dummy(nn.Sequential(QuantLinear(4096, 4096, bit=32),
                                          nn.BatchNorm1d(4096),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer8 = Dummy(last_fc(4096, num_classes))
        
        self.flat = Dummy(nn.Flatten(1))

        #initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.flat(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear, QuantReLU)):
                m.show_params()



class S_AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, bit: int = 32) -> None:
        super(S_AlexNet, self).__init__()
        self.bit = bit
        self.T = 2**self.bit - 1
        self.num_layers = 8
        self.flatten_id = 6
        
        self.layer1 = Spiking(nn.Sequential(first_conv(3, 64, kernel_size=11, stride=4, padding=2),
                                          nn.BatchNorm2d(64),
                                          IF()), self.T)
        self.layer2 = Spiking(nn.Sequential(QuantConv2d(64, 192, kernel_size=5, stride=2,padding=1, bit=32),
                                          nn.BatchNorm2d(192),
                                          IF()), self.T)
        self.layer3 = Spiking(nn.Sequential(QuantConv2d(192, 384, kernel_size=3, stride=2, padding=0, bit=32),
                                          nn.BatchNorm2d(384),
                                          IF()), self.T)
        self.layer4 = Spiking(nn.Sequential(QuantConv2d(384, 256, kernel_size=3, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          IF()), self.T)	
        self.layer5 = Spiking(nn.Sequential(QuantConv2d(256, 256, kernel_size=3, stride=2, padding=0, bit=32),
                                          nn.BatchNorm2d(256),
                                          IF()), self.T)	
        self.layer6 = Spiking(nn.Sequential(QuantLinear(256 * 6 * 6, 4096, bit=32),
                                          nn.BatchNorm1d(4096),
                                          IF()), self.T)
        self.layer7 = Spiking(nn.Sequential(QuantLinear(4096, 4096, bit=32),
                                          nn.BatchNorm1d(4096),
                                          IF()), self.T)
        self.layer8 = last_Spiking(last_fc(4096, num_classes), self.T)
        
        #flatten starts with dim=2 in SNNs
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
        x = self.layer8(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear, IF)):
                m.show_params()
                
def alexnet(spike=False, **kwargs) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    The required minimum input size of the model is 63x63.
    """
    if spike:
        model = S_AlexNet(**kwargs)
    else:
        model = AlexNet(**kwargs)
        
    return model

# def alexnet(**kwargs) -> AlexNet:
#     r"""AlexNet model architecture from the
#     `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
#     The required minimum input size of the model is 63x63.
#     """

#     model = AlexNet(**kwargs)
        
#     return model