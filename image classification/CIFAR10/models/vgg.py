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


class VGG11(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        init_weights: bool = True,
        float: bool = False
    ) -> None:
        super(VGG11, self).__init__()
        if float:
            self.layer1 = Dummy(nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(64),
                                              nn.ReLU(inplace=True)))
            self.layer2 = Dummy(nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(128),
                                              nn.ReLU(inplace=True)))
            self.layer3 = Dummy(nn.Sequential(nn.Conv2d(128, 256, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(256),
                                              nn.ReLU(inplace=True)))
            self.layer4 = Dummy(nn.Sequential(nn.Conv2d(256, 256, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(256),
                                              nn.ReLU(inplace=True)))
            self.layer5 = Dummy(nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(512),
                                              nn.ReLU(inplace=True)))
            self.layer6 = Dummy(nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(512),
                                              nn.ReLU(inplace=True)))	            
            self.layer7 = Dummy(nn.Sequential(nn.Conv2d(512, 512, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(512),
                                              nn.ReLU(inplace=True)))	
            self.layer8 = Dummy(nn.Sequential(nn.Conv2d(512, 512, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(512),
                                              nn.ReLU(inplace=True)))	            
            self.layer9 = Dummy(nn.Sequential(nn.Linear(2*2*512, 4096),
                                              nn.BatchNorm1d(4096),
                                              nn.ReLU(inplace=True)))
            self.layer10 = Dummy(nn.Sequential(nn.Linear(4096, 4096),
                                               nn.BatchNorm1d(4096),
                                               nn.ReLU(inplace=True)))
            self.layer11 = Dummy(nn.Linear(4096, 10))    
        
            self.flat = Dummy(nn.Flatten(1))
        else:
            self.layer1 = Dummy(nn.Sequential(first_conv(3, 64, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(64),
                                              QuantReLU(inplace=True)))
            self.layer2 = Dummy(nn.Sequential(QuantConv2d(64, 128, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(128),
                                              QuantReLU(inplace=True)))
            self.layer3 = Dummy(nn.Sequential(QuantConv2d(128, 256, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(256),
                                              QuantReLU(inplace=True)))
            self.layer4 = Dummy(nn.Sequential(QuantConv2d(256, 256, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(256),
                                              QuantReLU(inplace=True)))
            self.layer5 = Dummy(nn.Sequential(QuantConv2d(256, 512, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(512),
                                              QuantReLU(inplace=True)))	
            self.layer6 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(512),
                                              QuantReLU(inplace=True)))	            
            self.layer7 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=1, padding=1),
                                              nn.BatchNorm2d(512),
                                              QuantReLU(inplace=True)))	
            self.layer8 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1),
                                              nn.BatchNorm2d(512),
                                              QuantReLU(inplace=True)))	            
            self.layer9 = Dummy(nn.Sequential(QuantLinear(2*2*512, 4096),
                                              nn.BatchNorm1d(4096),
                                              QuantReLU(inplace=True)))
            self.layer10 = Dummy(nn.Sequential(QuantLinear(4096, 4096),
                                               nn.BatchNorm1d(4096),
                                               QuantReLU(inplace=True)))
            self.layer11 = Dummy(last_fc(4096, 10))
            
            self.flat = Dummy(nn.Flatten(1))
            
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.flat(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()


class S_VGG11(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        T: int = 3
    ) -> None:
        super(S_VGG11, self).__init__()

        self.layer1 = Spiking(nn.Sequential(first_conv(3, 64, 3, stride=1, padding=1),
                                            nn.BatchNorm2d(64),
                                            IF()), T)
        self.layer2 = Spiking(nn.Sequential(QuantConv2d(64, 128, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(128),
                                            IF()), T)
        self.layer3 = Spiking(nn.Sequential(QuantConv2d(128, 256, 3, stride=1, padding=1),
                                            nn.BatchNorm2d(256),
                                            IF()), T)
        self.layer4 = Spiking(nn.Sequential(QuantConv2d(256, 256, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(256),
                                            IF()), T)	
        self.layer5 = Spiking(nn.Sequential(QuantConv2d(256, 512, 3, stride=1, padding=1),
                                            nn.BatchNorm2d(512),
                                            IF()), T)	
        self.layer6 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(512),
                                            IF()), T)	            
        self.layer7 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=1, padding=1),
                                            nn.BatchNorm2d(512),
                                            IF()), T)	
        self.layer8 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(512),
                                            IF()), T)	            
        self.layer9 = Spiking(nn.Sequential(QuantLinear(2*2*512, 4096),
                                            nn.BatchNorm1d(4096),
                                            IF()), T)
        self.layer10 = Spiking(nn.Sequential(QuantLinear(4096, 4096),
                                             nn.BatchNorm1d(4096),
                                             IF()), T)
        self.layer11 = last_Spiking(last_fc(4096, 10), T)  
        
        self.flat = Dummy(nn.Flatten(2))
        
        self.layer1.is_first = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        #[batch,step,channels,height,width]
        #flatten starts with dim=2
        x = self.flat(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        return x


    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()

# net = S_VGG11()
# sample = torch.randn(10,3,32,32)
# output = net(sample)