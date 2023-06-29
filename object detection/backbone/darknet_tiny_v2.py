import torch
import torch.nn as nn
# import torch.nn.functional as F
from backbone.quant_layer import QuantReLU

model_urls = {
    "darknet_tiny": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/darknet_tiny.pth",
}

__all__ = ['build_darknet_tiny_v2']


# def convert_relu_to_quantrelu(model, bit):
#     for child_name, child in model.named_children():
#         if isinstance(child, nn.ReLU):
#             setattr(model, child_name, QuantReLU(inplace=True, bit=bit))
#         else:
#             convert_relu_to_quantrelu(child, bit)


class Dummy(nn.Module):
    def __init__(self, block):
        super(Dummy, self).__init__()
        self.block = block
        self.idem = False
    def forward(self, x):
        if self.idem:
            return x
        return self.block(x)



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
    def __init__(self, block, T=3):
        super(last_Spiking, self).__init__()
        self.block = block
        self.T = T
        self.idem = False
        
    def forward(self, x):
        if self.idem:
            return x
        #prepare charges
        train_shape = [x.shape[0], x.shape[1]]
        self.T = x.shape[1]
        
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        
        #integrate charges
        return x.sum(dim=1).div(self.T)


class S_DarkNet_Tiny_v2(nn.Module):
    def __init__(self, bit=3, num_anchors=5, num_classes=20):
        
        super(S_DarkNet_Tiny_v2, self).__init__()
        # backbone network : DarkNet_Tiny
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.T = T = 2**bit-1

        self.layer1 = Spiking(nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(16),
                                            IF()), T)	 
        
        self.layer2 = Spiking(nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(32),
                                            IF()), T)	 
            
        self.layer3 = Spiking(nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(64),
                                            IF()), T)	 
        
        self.layer4 = Spiking(nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(128),
                                            IF()), T)	 
        
        self.layer5 = Spiking(nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                            nn.BatchNorm2d(256),
                                            IF()), T)	 
        
        self.layer6 = Spiking(nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                            nn.BatchNorm2d(512),
                                            IF()), T)	  
           
        self.layer7 = Spiking(nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                                            nn.BatchNorm2d(1024),
                                            IF()), T)	 
        
        self.head = Spiking(nn.Sequential(nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(1024),
                                          IF()), T)	 
        
        self.pred = last_Spiking(nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1))

    def forward(self, x):
        
        x.unsqueeze_(1)
        x = x.repeat(1, self.T, 1, 1, 1)        
        
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = self.layer5(x)       # stride = 16

        x = self.layer6(x)

        x = self.layer7(x)       # stride = 32
        
        x = self.head(x)
        
        x = self.pred(x)
        return x

class DarkNet_Tiny_v2(nn.Module):
    def __init__(self, bit=32, num_anchors=5, num_classes=20):
        
        super(DarkNet_Tiny_v2, self).__init__()
        # backbone network : DarkNet_Tiny
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.bit = bit
        
        self.layer1 = Dummy(nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(16),
                                          QuantReLU(inplace=True, bit=self.bit)))
        
        self.layer2 = Dummy(nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(32),
                                          QuantReLU(inplace=True, bit=self.bit)))
        
        self.layer3 = Dummy(nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(64),
                                          QuantReLU(inplace=True, bit=self.bit)))
        
        self.layer4 = Dummy(nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(128),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        
        self.layer5 = Dummy(nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                          nn.BatchNorm2d(256),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        
        self.layer6 = Dummy(nn.Sequential(nn.Conv2d(256, 512, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(512),
                                          QuantReLU(inplace=True, bit=self.bit)))	 
           
        self.layer7 = Dummy(nn.Sequential(nn.Conv2d(512, 1024, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(1024),
                                          QuantReLU(inplace=True, bit=self.bit)))	
        
        
        self.head = Dummy(nn.Sequential(nn.Conv2d(1024, 1024, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(1024),
                                          QuantReLU(inplace=True, bit=self.bit)))
        
        self.pred = Dummy(nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), kernel_size=1))
        
    def forward(self, x):
        x = self.layer1(x)       # stride = 2

        x = self.layer2(x)       # stride = 4
        
        x = self.layer3(x)       # stride = 8

        x = self.layer4(x)       # stride = 16

        x = self.layer5(x)       # stride = 32
 
        x = self.layer6(x)       # stride = 32
    
        x = self.layer7(x)       # stride = 32
        
        x = self.head(x)
        
        x = self.pred(x)

        return x


def build_darknet_tiny_v2(bit=32, spike=False, pretrained=False, num_anchors=5, num_classes=20):
    # model
    print('precision is %d' %bit)
    if spike:
        print('Evaluating with spikes...')
        model = S_DarkNet_Tiny_v2(bit=bit, num_anchors=num_anchors, num_classes=num_classes)
    else:
        print('Evaluating with defaults...')
        model = DarkNet_Tiny_v2(bit=bit, num_anchors=num_anchors, num_classes=num_classes)
    
    if bit==32:
        pretrained = True
    
    # load weight
    if pretrained:
        print('Loading pretrained weight ...')
        # checkpoint state dict
        checkpoint_state_dict = torch.load('./backbone/weights/darknet_tiny_32bit.pth.tar', map_location="cpu")
    
        model.load_state_dict(checkpoint_state_dict, strict=False)

    return model


if __name__ == '__main__':
    import time
    net = build_darknet_tiny_v2(bit=3, pretrained=False)
    # checkpoint = torch.load('./weights/voc/yolov2_tiny_3bit/yolov2_tiny_epoch_41_58.07.pth', map_location='cpu')
    # for name in list(checkpoint.keys()):
    #     checkpoint[name[9:]] = checkpoint.pop(name)    
    # net.load_state_dict(checkpoint, strict=False)
    
    # net.conv_1_2.load_state_dict(net.conv_1.state_dict())
    
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    output = net(x)
    t1 = time.time()
    print('Time: ', t1 - t0)

    # for k in output.keys():
    #     print('{} : {}'.format(k, output[k].shape))
