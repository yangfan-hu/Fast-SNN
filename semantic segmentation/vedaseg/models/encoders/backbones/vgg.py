from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_layer import QuantReLU

from .registry import BACKBONES
import logging
logger = logging.getLogger()


class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=1)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


@BACKBONES.register_module
class MYVGG11(VGG):
    """ResNetEncoder

    Args:
        pretrain(bool)
    """

    def __init__(self, pretrain=True):

        super().__init__(make_layers(cfgs["A"], batch_norm=True))
        if pretrain:
            logger.info('VGG11 init weights from pretreain')
            state_dict = torch.hub.load_state_dict_from_url(
            url="https://download.pytorch.org/models/vgg11_bn-6002323d.pth", map_location="cpu", check_hash=True)
            
            if 'state_dict' in state_dict:
                # handle state_dict format from mmseg
                state_dict = state_dict['state_dict']
            self.load_state_dict(state_dict, strict=False)

        del self.classifier, self.avgpool
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = {}
        x = self.features(x)
        feats['c5'] = x #32
        return feats


class Dummy(nn.Module):
    def __init__(self, block):
        super(Dummy, self).__init__()
        self.block = block
        self.idem = False
    def forward(self, x):
        if self.idem:
            return x
        return self.block(x)


class VGG9(nn.Module):
    def __init__(
        self,
        nclasses: int = 21,
        bit: int = 32
    ) -> None:
        super(VGG9, self).__init__()
        
        self.bit = bit
        
        self.layer1 = Dummy(nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(64),
                                          QuantReLU(inplace=True, bit=bit)))
        self.layer2 = Dummy(nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(64),
                                          QuantReLU(inplace=True, bit=bit)))
        self.layer3 = Dummy(nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(128),
                                          QuantReLU(inplace=True, bit=bit)))
        self.layer4 = Dummy(nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(128),
                                          QuantReLU(inplace=True, bit=bit)))	
        self.layer5 = Dummy(nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(256),  
                                          QuantReLU(inplace=True, bit=bit)))	 

        self.layer6 = Dummy(nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=2, dilation=2),
                                          nn.BatchNorm2d(256),
                                          QuantReLU(inplace=True, bit=bit)))	

        self.layer7 = Dummy(nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=2, dilation=2),
                                          nn.BatchNorm2d(256),
                                          QuantReLU(inplace=True, bit=bit)))	
        
        self.layer8 = Dummy(nn.Sequential(nn.Conv2d(256, 1024, 3, stride=1, padding=12, dilation=12),
                                           nn.BatchNorm2d(1024),
                                           QuantReLU(inplace=True, bit=bit)))
        
        # self.layer9 = Dummy(nn.Sequential(nn.Conv2d(1024, 21, 1),
        #                                     nn.BatchNorm2d(21),
        #                                     QuantReLU(inplace=True, bit=bit)))


        self.layer9 = Dummy(nn.Sequential(nn.Conv2d(1024, nclasses, 1)))

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
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantReLU):
                m.show_params()




@BACKBONES.register_module
class MYVGG9(VGG9):
    """Encoder

    Args:
        pretrain(bool)
    """

    def __init__(self, bit=32, pretrain=True, init=None, nclasses=21):

        super().__init__(bit=bit, nclasses=nclasses)
        
        if init:
            logger.info('VGG9 init weights from high-precision pretreain')
            state_dict = torch.load(init,  map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            for name in list(state_dict.keys()):
                state_dict[name[2:]] = state_dict.pop(name)
                
            self.load_state_dict(state_dict, strict=False)
        
        
        
        elif pretrain:
            logger.info('VGG9 init weights from imagenet pretreain')
            state_dict = torch.load('pretrain/vgg9.pth',  map_location='cpu')
            if 'state_dict' in state_dict:
                # handle state_dict format from mmseg
                state_dict = state_dict['state_dict']
            self.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = {}
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        feats['c5'] = x #32
        return feats
        # return x

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
        return x.sum(dim=1).div(self.T)
    
class IF(nn.Module):
    def __init__(self):
        super(IF, self).__init__()
        ###changes threshold to act_alpha
        ###being fleet
        self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x

    def show_params(self):
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold activation alpha: {:2f}'.format(act_alpha)) 
    
    def extra_repr(self) -> str:
        return 'threshold={:.3f}'.format(self.act_alpha)  


class S_VGG9(nn.Module):
    def __init__(
        self,
        nclasses: int = 21,
        T: int = 3
        
    ) -> None:
        super(S_VGG9, self).__init__()
        
        self.layer1 = Spiking(nn.Sequential(nn.Conv2d(3, 64, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(64),
                                          IF()), T)
        self.layer2 = Spiking(nn.Sequential(nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(64),
                                          IF()), T)
        self.layer3 = Spiking(nn.Sequential(nn.Conv2d(64, 128, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(128),
                                          IF()), T)
        self.layer4 = Spiking(nn.Sequential(nn.Conv2d(128, 128, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(128),
                                          IF()), T)
        self.layer5 = Spiking(nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(256),  
                                          IF()), T)	 
        
        self.layer6 = Spiking(nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=2, dilation=2),
                                          nn.BatchNorm2d(256),
                                          IF()), T)

        self.layer7 = Spiking(nn.Sequential(nn.Conv2d(256, 256, 3, stride=1, padding=2, dilation=2),
                                          nn.BatchNorm2d(256),
                                          IF()), T)	
        
        self.layer8 = Spiking(nn.Sequential(nn.Conv2d(256, 1024, 3, stride=1, padding=12, dilation=12),
                                           nn.BatchNorm2d(1024),
                                           IF()), T)
        
        self.layer9 = last_Spiking(nn.Sequential(nn.Conv2d(1024, nclasses, 1)), T)
        
        self.layer1.is_first = True
        
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
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantReLU):
                m.show_params()


@BACKBONES.register_module
class MYSVGG9(S_VGG9):
    """Encoder

    Args:
        pretrain(bool)
    """

    def __init__(self, bit=32, pretrain=False, init=None, nclasses=21):

        super().__init__(T=bit**2-1, nclasses=nclasses)
        
        # if init:
        #     logger.info('VGG9 init weights from high-precision pretreain')
        
        # elif pretrain:
        #     logger.info('VGG9 init weights from imagenet pretreain')
        #     state_dict = torch.load('pretrain/vgg9.pth',  map_location='cpu')
        #     if 'state_dict' in state_dict:
        #         # handle state_dict format from mmseg
        #         state_dict = state_dict['state_dict']
        #     self.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = {}
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        feats['c5'] = x #32
        return feats
        # return x






a = MYVGG11(pretrain=False)
sample = torch.randn(5,3,513,513)
b = a(sample)
c = b['c5']













