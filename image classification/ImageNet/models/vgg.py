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

class VGG16(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        bit: int = 32
    ) -> None:
        super(VGG16, self).__init__()
        
        self.bit = bit
        self.num_layers = 16
        self.flatten_id = 14
        
        self.layer1 = Dummy(nn.Sequential(first_conv(3, 64, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(64),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer2 = Dummy(nn.Sequential(QuantConv2d(64, 64, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(64),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer3 = Dummy(nn.Sequential(QuantConv2d(64, 128, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(128),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer4 = Dummy(nn.Sequential(QuantConv2d(128, 128, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(128),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        self.layer5 = Dummy(nn.Sequential(QuantConv2d(128, 256, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        self.layer6 = Dummy(nn.Sequential(QuantConv2d(256, 256, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          QuantReLU(inplace=True, bit=self.bit)))	            
        self.layer7 = Dummy(nn.Sequential(QuantConv2d(256, 256, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        self.layer8 = Dummy(nn.Sequential(QuantConv2d(256, 512, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          QuantReLU(inplace=True, bit=self.bit)))	 
        self.layer9 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        self.layer10 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1, bit=32),
                                           nn.BatchNorm2d(512),
                                           QuantReLU(inplace=True, bit=self.bit)))	        
        self.layer11 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=1, padding=1, bit=32),
                                           nn.BatchNorm2d(512),
                                           QuantReLU(inplace=True, bit=self.bit)))	
        self.layer12 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=1, padding=1, bit=32),
                                           nn.BatchNorm2d(512),
                                           QuantReLU(inplace=True, bit=self.bit)))	
        self.layer13 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1, bit=32),
                                           nn.BatchNorm2d(512),
                                           QuantReLU(inplace=True, bit=self.bit)))	           
        self.layer14 = Dummy(nn.Sequential(QuantLinear(7*7*512, 4096, bit=32),
                                           nn.BatchNorm1d(4096),
                                           QuantReLU(inplace=True, bit=self.bit)))
        self.layer15 = Dummy(nn.Sequential(QuantLinear(4096, 4096, bit=32),
                                           nn.BatchNorm1d(4096),
                                           QuantReLU(inplace=True, bit=self.bit)))
        self.layer16 = Dummy(last_fc(4096, num_classes))  
        
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
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)    
        x = self.layer13(x)
        x = self.flat(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()
                

class S_VGG16(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        bit: int = 32
    ) -> None:
        super(S_VGG16, self).__init__()
        
        self.bit = bit
        self.T = 2**self.bit - 1
        self.num_layers = 16
        self.flatten_id = 14
        
        self.layer1 = Spiking(nn.Sequential(first_conv(3, 64, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(64),
                                          IF()), self.T)
        self.layer2 = Spiking(nn.Sequential(QuantConv2d(64, 64, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(64),
                                          IF()), self.T)
        self.layer3 = Spiking(nn.Sequential(QuantConv2d(64, 128, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(128),
                                          IF()), self.T)
        self.layer4 = Spiking(nn.Sequential(QuantConv2d(128, 128, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(128),
                                          IF()), self.T)	
        self.layer5 = Spiking(nn.Sequential(QuantConv2d(128, 256, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          IF()), self.T)	
        self.layer6 = Spiking(nn.Sequential(QuantConv2d(256, 256, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          IF()), self.T)	            
        self.layer7 = Spiking(nn.Sequential(QuantConv2d(256, 256, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          IF()), self.T)	
        self.layer8 = Spiking(nn.Sequential(QuantConv2d(256, 512, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          IF()), self.T)	 
        self.layer9 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          IF()), self.T)	
        self.layer10 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1, bit=32),
                                           nn.BatchNorm2d(512),
                                           IF()), self.T)	        
        self.layer11 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=1, padding=1, bit=32),
                                           nn.BatchNorm2d(512),
                                           IF()), self.T)	
        self.layer12 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=1, padding=1, bit=32),
                                           nn.BatchNorm2d(512),
                                           IF()), self.T)	
        self.layer13 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1, bit=32),
                                           nn.BatchNorm2d(512),
                                           IF()), self.T)	           
        self.layer14 = Spiking(nn.Sequential(QuantLinear(7*7*512, 4096, bit=32),
                                           nn.BatchNorm1d(4096),
                                           IF()), self.T)
        self.layer15 = Spiking(nn.Sequential(QuantLinear(4096, 4096, bit=32),
                                           nn.BatchNorm1d(4096),
                                           IF()), self.T)
        self.layer16 = last_Spiking(last_fc(4096, num_classes), self.T)  
        
        #flatten starts with dim=2 in SNNs
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
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)    
        x = self.layer13(x)
        x = self.flat(x)
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()    



class VGG11(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        bit: int = 32
    ) -> None:
        super(VGG11, self).__init__()
        
        self.bit = bit
        self.num_layers = 11
        self.flatten_id = 9        
        self.layer1 = Dummy(nn.Sequential(first_conv(3, 64, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(64),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer2 = Dummy(nn.Sequential(QuantConv2d(64, 128, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(128),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer3 = Dummy(nn.Sequential(QuantConv2d(128, 256, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          QuantReLU(inplace=True, bit=self.bit)))
        self.layer4 = Dummy(nn.Sequential(QuantConv2d(256, 256, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        self.layer5 = Dummy(nn.Sequential(QuantConv2d(256, 512, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        self.layer6 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          QuantReLU(inplace=True, bit=self.bit)))	            
        self.layer7 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        self.layer8 = Dummy(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          QuantReLU(inplace=True, bit=self.bit)))	           
        self.layer9 = Dummy(nn.Sequential(QuantLinear(7*7*512, 4096, bit=32),
                                           nn.BatchNorm1d(4096),
                                           QuantReLU(inplace=True, bit=self.bit)))
        self.layer10 = Dummy(nn.Sequential(QuantLinear(4096, 4096, bit=32),
                                           nn.BatchNorm1d(4096),
                                           QuantReLU(inplace=True, bit=self.bit)))
        self.layer11 = Dummy(last_fc(4096, num_classes))  
        
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
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.flat(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()                
 
class S_VGG11(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        bit: int = 32
    ) -> None:
        super(S_VGG11, self).__init__()
        
        self.bit = bit
        self.T = 2**self.bit - 1
        self.num_layers = 11
        self.flatten_id = 9      
        
        self.layer1 = Spiking(nn.Sequential(first_conv(3, 64, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(64),
                                          IF()), self.T)
        self.layer2 = Spiking(nn.Sequential(QuantConv2d(64, 128, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(128),
                                          IF()), self.T)
        self.layer3 = Spiking(nn.Sequential(QuantConv2d(128, 256, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          IF()), self.T)
        self.layer4 = Spiking(nn.Sequential(QuantConv2d(256, 256, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(256),
                                          IF()), self.T)	
        self.layer5 = Spiking(nn.Sequential(QuantConv2d(256, 512, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          IF()), self.T)	
        self.layer6 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          IF()), self.T)	            
        self.layer7 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=1, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          IF()), self.T)	
        self.layer8 = Spiking(nn.Sequential(QuantConv2d(512, 512, 3, stride=2, padding=1, bit=32),
                                          nn.BatchNorm2d(512),
                                          IF()), self.T)	           
        self.layer9 = Spiking(nn.Sequential(QuantLinear(7*7*512, 4096, bit=32),
                                           nn.BatchNorm1d(4096),
                                           IF()), self.T)
        self.layer10 = Spiking(nn.Sequential(QuantLinear(4096, 4096, bit=32),
                                           nn.BatchNorm1d(4096),
                                           IF()), self.T)
        self.layer11 = last_Spiking(last_fc(4096, num_classes), self.T)  
        
        #flatten starts with dim=2 in SNNs
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
        x = self.flat(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear) or isinstance(m, QuantReLU):
                m.show_params()         
 
    
def vgg16(spike=False, **kwargs) -> VGG16:
   r"""VGG 16-layer model (configuration "D")
   `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
   The required minimum input size of the model is 32x32.

   Args:
       pretrained (bool): If True, returns a model pre-trained on ImageNet
       progress (bool): If True, displays a progress bar of the download to stderr
   """
   if spike:
       return S_VGG16(**kwargs)
   else:
       return VGG16(**kwargs)               


def vgg11(spike=False, **kwargs) -> VGG16:
   r"""VGG 16-layer model (configuration "D")
   `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
   The required minimum input size of the model is 32x32.

   Args:
       pretrained (bool): If True, returns a model pre-trained on ImageNet
       progress (bool): If True, displays a progress bar of the download to stderr
   """
   if spike:
       return S_VGG11(**kwargs) 
   else:
       return VGG11(**kwargs)          