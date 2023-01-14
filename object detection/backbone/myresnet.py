# import logging
import torch
import torch.nn as nn
from torchvision.models.resnet import model_urls

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from backbone.quant_layer import QuantReLU


# logger = logging.getLogger()

model_urls.update(
    {
        "resnet18_v1c": "https://download.openmmlab.com/pretrain/third_party/"
                        "resnet18_v1c-b5776b93.pth",
        "resnet50_v1c": "https://download.openmmlab.com/pretrain/third_party/"
                        "resnet50_v1c-2cccc1ad.pth",
        "resnet101_v1c": "https://download.openmmlab.com/pretrain/third_party/"
                         "resnet101_v1c-e67eebb6.pth",
    }
)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


class Dummy(nn.Module):
    def __init__(self, block):
        super(Dummy, self).__init__()
        self.block = block
        self.idem = False
    def forward(self, x):
        if self.idem:
            return x
        return self.block(x)


class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        _height, _width = height // self.stride, width // self.stride
        
        x = x.view(batch_size, channels, _height, self.stride, _width, self.stride).transpose(3, 4).contiguous()
        x = x.view(batch_size, channels, _height * _width, self.stride * self.stride).transpose(2, 3).contiguous()
        x = x.view(batch_size, channels, self.stride * self.stride, _height, _width).transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, _height, _width)

        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_layer, act_layer, stride=1,
                 downsample=None, groups=1,
                 base_width=64, dilation=1, bit=32):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        
        self.part1 = Dummy(nn.Sequential(conv3x3(inplanes, planes, stride),
                                         norm_layer(planes)))
        self.relu1 = act_layer(inplace=True, bit=bit)
        
        self.part2 = Dummy(nn.Sequential(conv3x3(planes, planes),
                                         norm_layer(planes)))
        self.downsample = downsample
        self.relu2 = act_layer(inplace=True, bit=bit)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.part1(x)
        out = self.relu1(out)

        out = self.part2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, norm_layer, act_layer, stride=1,
                 downsample=None, groups=1,
                 base_width=64, dilation=1, ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.relu1 = act_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.relu2 = act_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = act_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


MODEL_CFGS = {
    'resnet101': {
        'block': Bottleneck,
        'layer': [3, 4, 23, 3],
    },
    'resnet50': {
        'block': Bottleneck,
        'layer': [3, 4, 6, 3],
    },
    'resnet18': {
        'block': BasicBlock,
        'layer': [2, 2, 2, 2],
    },
    'resnet34':{
        'block':BasicBlock,
        'layer':[3, 4, 6, 3]}
    
}


class ResNetCls(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, layer_stride=[1,2,2,2], multi_grid=None,
                 norm_cfg=None, act_cfg=None, deep_stem=False, bit=32):
        super(ResNetCls, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        self._norm_layer = nn.BatchNorm2d
        # if act_cfg is None:
        #     act_cfg = dict(type='Relu', inplace=True)
        # self._act_layer = partial(build_act_layer, act_cfg, layer_only=True)
        
        self._act_layer = QuantReLU
        

        self.deep_stem = deep_stem
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element '
                'tuple, got {}'.format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self._make_stem_layer(bit=bit)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=layer_stride[0], bit=bit)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=layer_stride[1],
                                       dilate=replace_stride_with_dilation[0], bit=bit)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=layer_stride[2],
                                       dilate=replace_stride_with_dilation[1], bit=bit)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=layer_stride[3],
                                       dilate=replace_stride_with_dilation[2],
                                       multi_grid=multi_grid, bit=bit)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    multi_grid=None, bit=32):
        norm_layer = self._norm_layer
        act_layer = self._act_layer
        downsample = None

        if multi_grid is None:
            multi_grid = [1 for _ in range(blocks)]
        else:
            assert len(multi_grid) == blocks

        if dilate:
            self.dilation *= stride
            stride = 1

        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Dummy(nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ))

        layers = [block(self.inplanes, planes, norm_layer, act_layer,
                        stride, downsample, self.groups, self.base_width,
                        previous_dilation * multi_grid[0], bit=bit)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                act_layer=act_layer,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation * multi_grid[i], bit=bit))

        return nn.Sequential(*layers)

    def _make_stem_layer(self, bit=32):
        """Make stem layer for ResNet."""
        
        self.layer0 = Dummy(nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=4,
                                                    padding=3, bias=False),
                                          self._norm_layer(self.inplanes)
                                          ))
        
        self.relu0 = self._act_layer(inplace=True, bit=bit)


    def forward(self, x):
        
        x = self.layer0(x)
        x = self.relu0(x)
        

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def reshuffle_resnet_keys(ck, blocks=[3, 4, 6, 3], stirde = [1, 2, 2, 2]):
    ck['layer0.block.0.weight'] = ck.pop('conv1.weight')
    ck['layer0.block.1.running_mean'] = ck.pop('bn1.running_mean')
    ck['layer0.block.1.running_var'] = ck.pop('bn1.running_var')
    ck['layer0.block.1.weight'] = ck.pop('bn1.weight')
    ck['layer0.block.1.bias'] = ck.pop('bn1.bias')

    for layer_id in range(1,5):
        for block_id in range(blocks[layer_id-1]):
            #Basicblock part1
            ck['layer'+str(layer_id) + '.' + str(block_id) + '.part1.block' + '.0.weight'] = ck.pop(
                'layer'+str(layer_id) + '.' + str(block_id) + '.conv1.weight')
            ck['layer'+str(layer_id) + '.' + str(block_id) + '.part1.block' + '.1.running_mean'] = ck.pop(
                'layer'+str(layer_id) + '.' + str(block_id) + '.bn1.running_mean')
            ck['layer'+str(layer_id) + '.' + str(block_id) + '.part1.block' + '.1.running_var'] = ck.pop(
                'layer'+str(layer_id) + '.' + str(block_id) + '.bn1.running_var')
            ck['layer'+str(layer_id) + '.' + str(block_id) + '.part1.block' + '.1.weight'] = ck.pop(
                'layer'+str(layer_id) + '.' + str(block_id) + '.bn1.weight')
            ck['layer'+str(layer_id) + '.' + str(block_id) + '.part1.block' + '.1.bias'] = ck.pop(
                'layer'+str(layer_id) + '.' + str(block_id) + '.bn1.bias')
            #Basicblock part2
            ck['layer'+str(layer_id) + '.' + str(block_id) + '.part2.block' + '.0.weight'] = ck.pop(
                'layer'+str(layer_id) + '.' + str(block_id) + '.conv2.weight')
            ck['layer'+str(layer_id) + '.' + str(block_id) + '.part2.block' + '.1.running_mean'] = ck.pop(
                'layer'+str(layer_id) + '.' + str(block_id) + '.bn2.running_mean')
            ck['layer'+str(layer_id) + '.' + str(block_id) + '.part2.block' + '.1.running_var'] = ck.pop(
                'layer'+str(layer_id) + '.' + str(block_id) + '.bn2.running_var')
            ck['layer'+str(layer_id) + '.' + str(block_id) + '.part2.block' + '.1.weight'] = ck.pop(
                'layer'+str(layer_id) + '.' + str(block_id) + '.bn2.weight')
            ck['layer'+str(layer_id) + '.' + str(block_id) + '.part2.block' + '.1.bias'] = ck.pop(
                'layer'+str(layer_id) + '.' + str(block_id) + '.bn2.bias')
            
            #downsamples
            if block_id == 0 and stirde[layer_id-1] == 2 :
                ck['layer'+str(layer_id) + '.' + str(block_id) + '.downsample.block' + '.0.weight'] = ck.pop(
                    'layer'+str(layer_id) + '.' + str(block_id) + '.downsample' + '.0.weight')
                ck['layer'+str(layer_id) + '.' + str(block_id) + '.downsample.block' + '.1.running_mean'] = ck.pop(
                    'layer'+str(layer_id) + '.' + str(block_id) + '.downsample' + '.1.running_mean')
                ck['layer'+str(layer_id) + '.' + str(block_id) + '.downsample.block' + '.1.running_var'] = ck.pop(
                    'layer'+str(layer_id) + '.' + str(block_id) + '.downsample' + '.1.running_var')
                ck['layer'+str(layer_id) + '.' + str(block_id) + '.downsample.block' + '.1.weight'] = ck.pop(
                    'layer'+str(layer_id) + '.' + str(block_id) + '.downsample' + '.1.weight')
                ck['layer'+str(layer_id) + '.' + str(block_id) + '.downsample.block' + '.1.bias'] = ck.pop(
                    'layer'+str(layer_id) + '.' + str(block_id) + '.downsample' + '.1.bias')



class MYResNet(ResNetCls):
    """ResNetEncoder

    Args:
        pretrain(bool)
    """

    def __init__(self, arch, replace_stride_with_dilation=None, layer_stride=[1,2,2,2], bit=32, num_anchors=5, num_classes=20,
                 multi_grid=None, pretrain=True, norm_cfg=None, act_cfg=None, init=None):
        cfg = MODEL_CFGS[arch[:-4] if arch.endswith('_v1c') else arch]

        super().__init__(
            cfg['block'],
            cfg['layer'],
            replace_stride_with_dilation=replace_stride_with_dilation,
            layer_stride=layer_stride,
            deep_stem=arch.endswith('_v1c'),
            multi_grid=multi_grid,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bit=bit)
        
        print('precision is %d' %bit)
        
        if bit==32:
            pretrain = True


        if pretrain:
            print('ResNet init weights from ImageNet pretrain')
            if arch not in model_urls:
                raise KeyError('No model url exist for {}'.format(arch))
            state_dict = load_state_dict_from_url(model_urls[arch])
            if 'state_dict' in state_dict:
                # handle state_dict format from mmseg
                state_dict = state_dict['state_dict']
                
            reshuffle_resnet_keys(state_dict, cfg['layer'], layer_stride)
            self.load_state_dict(state_dict, strict=False)
            
        elif init:
            print('ResNet init weights from high-precision pretrain')
            state_dict = torch.load(init,  map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            for name in list(state_dict.keys()):
                state_dict[name[2:]] = state_dict.pop(name)
                
            self.load_state_dict(state_dict, strict=False)            
            
        else:
            print('ResNet init weights')

        del self.fc, self.avgpool
        
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.head1 = Dummy(nn.Sequential(nn.Conv2d(512, 1024, 1, bias=False),
                                          self._norm_layer(1024),
                                          self._act_layer(inplace=True, bit=bit)))
        
        self.head2 = Dummy(nn.Sequential(nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
                                          self._norm_layer(1024),
                                          self._act_layer(inplace=True, bit=bit)))
        
        self.head3 = Dummy(nn.Sequential(nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
                                          self._norm_layer(1024),
                                          self._act_layer(inplace=True, bit=bit)))
        
        
        self.route_layer = Dummy(nn.Sequential(nn.Conv2d(256, 128, 1, bias=False),
                                                self._norm_layer(128),
                                                self._act_layer(inplace=True, bit=bit)))
    
        self.reorg = reorg_layer(stride=2)
        

        self.head4 = Dummy(nn.Sequential(nn.Conv2d(1024+128*4, 1024, 3, padding=1, bias=False),
                                          self._norm_layer(1024),
                                          self._act_layer(inplace=True, bit=bit)))
        
        
        self.pred = Dummy(nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), 1))
        

    
    def forward(self, x):
     
        x = self.layer0(x)
        x = self.relu0(x)

        x = self.layer1(x)  # 4
        x = self.layer2(x)  # 8
        x = self.layer3(x)  # 16
        c4 = x
        x = self.layer4(x)  # 32
        c5 = x     
        
        #convs1
        c5 = self.head1(c5)
        c5 = self.head2(c5)
        c5 = self.head3(c5)
        
        c4 = self.route_layer(c4)
        c4 = self.reorg(c4)
        
        c5 = torch.cat([c4, c5], dim=1)
        
        c5 = self.head4(c5)
        
        x = self.pred(c5)

        return x

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


class IF_act(nn.Module):
    def __init__(self):
        super(IF_act, self).__init__()
        ###changes threshold to act_alpha
        ###being fleet
        self.act_alpha = torch.nn.Parameter(torch.tensor(1.0))
        self.T = 3
        self.sign=True
    def forward(self, x):
        self.T = x.shape[1]
        
        threshold = self.act_alpha.data
        membrane = 0.5 * threshold
        sum_spikes = 0
        
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

    def show_params(self):
        act_alpha = round(self.act_alpha.data.item(), 3)
        print('clipping threshold activation alpha: {:2f}'.format(act_alpha)) 
    
    def extra_repr(self) -> str:
        return 'threshold={:.3f}'.format(self.act_alpha)  



class Wrapper(nn.Module):
    def __init__(self, block):
        super(Wrapper, self).__init__()
        self.block = block

    def forward(self, x):

        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)        

        return x


class S_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, norm_layer, act_layer, stride=1,
                 downsample=None, groups=1,
                 base_width=64, dilation=1, bit=32):
        super(S_BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        
        self.part1 = Wrapper(nn.Sequential(conv3x3(inplanes, planes, stride),
                                         norm_layer(planes)))
        self.relu1 = act_layer()
        
        self.part2 = Wrapper(nn.Sequential(conv3x3(planes, planes),
                                         norm_layer(planes)))
        self.downsample = downsample
        self.relu2 = act_layer()
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.part1(x)
        out = self.relu1(out)

        out = self.part2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu2(out)

        return out

class SResNetCls(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, layer_stride=[1,2,2,2], multi_grid=None,
                 norm_cfg=None, act_cfg=None, deep_stem=False, bit=32):
        super(SResNetCls, self).__init__()

        # if norm_cfg is None:
        #     self._norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        
        # if act_cfg is None:
        #     act_cfg = dict(type='Relu', inplace=True)
        # self._act_layer = partial(build_act_layer, act_cfg, layer_only=True)
        
        self._act_layer = IF_act
        

        self.deep_stem = deep_stem
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element '
                'tuple, got {}'.format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self._make_stem_layer(bit=bit)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=layer_stride[0], bit=bit)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=layer_stride[1],
                                       dilate=replace_stride_with_dilation[0], bit=bit)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=layer_stride[2],
                                       dilate=replace_stride_with_dilation[1], bit=bit)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=layer_stride[3],
                                       dilate=replace_stride_with_dilation[2],
                                       multi_grid=multi_grid, bit=bit)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    multi_grid=None, bit=32):
        norm_layer = self._norm_layer
        act_layer = self._act_layer
        downsample = None

        if multi_grid is None:
            multi_grid = [1 for _ in range(blocks)]
        else:
            assert len(multi_grid) == blocks

        if dilate:
            self.dilation *= stride
            stride = 1

        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Wrapper(nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            ))

        layers = [block(self.inplanes, planes, norm_layer, act_layer,
                        stride, downsample, self.groups, self.base_width,
                        previous_dilation * multi_grid[0], bit=bit)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                act_layer=act_layer,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation * multi_grid[i], bit=bit))

        return nn.Sequential(*layers)

    def _make_stem_layer(self, bit=32):
        """Make stem layer for ResNet."""
        
        self.layer0 = Wrapper(nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=4,
                                                    padding=3, bias=False),
                                          self._norm_layer(self.inplanes)
                                          ))
        
        self.relu0 = self._act_layer()


    def forward(self, x):
        x = self.layer0(x)
        x = self.relu0(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class MYSResNet(SResNetCls):
    """ResNetEncoder

    Args:
        pretrain(bool)
    """

    def __init__(self, arch, replace_stride_with_dilation=None, layer_stride=[1,2,2,2], bit=32, num_anchors=5, num_classes=20,
                 multi_grid=None, pretrain=True, norm_cfg=None, act_cfg=None, init=None):
        cfg = MODEL_CFGS[arch[:-4] if arch.endswith('_v1c') else arch]

        super().__init__(
            S_BasicBlock,
            cfg['layer'],
            replace_stride_with_dilation=replace_stride_with_dilation,
            layer_stride=layer_stride,
            deep_stem=arch.endswith('_v1c'),
            multi_grid=multi_grid,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bit=bit)
        
        print('precision is %d' %bit)

        if pretrain:
            print('ResNet init weights from ImageNet pretrain')
            if arch not in model_urls:
                raise KeyError('No model url exist for {}'.format(arch))
            state_dict = load_state_dict_from_url(model_urls[arch])
            if 'state_dict' in state_dict:
                # handle state_dict format from mmseg
                state_dict = state_dict['state_dict']
                
            reshuffle_resnet_keys(state_dict, cfg['layer'], layer_stride)
            self.load_state_dict(state_dict, strict=False)
            
        elif init:
            print('ResNet init weights from high-precision pretrain')
            state_dict = torch.load(init,  map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            for name in list(state_dict.keys()):
                state_dict[name[2:]] = state_dict.pop(name)
                
            self.load_state_dict(state_dict, strict=False)            
            
        else:
            print('ResNet init weights')


        del self.fc, self.avgpool
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes


        T = 2**bit-1
        self.T = T
        
        #Moving ASPP modules here  
        self.head1 = Spiking(nn.Sequential(nn.Conv2d(512, 1024, 1, bias=False),
                                           self._norm_layer(1024),
                                           IF()), T)
        
        self.head2 = Spiking(nn.Sequential(nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
                                           self._norm_layer(1024),
                                           IF()), T)
        
        self.head3 = Spiking(nn.Sequential(nn.Conv2d(1024, 1024, 3, padding=1, bias=False),
                                           self._norm_layer(1024),
                                           IF()), T)
        

        self.route_layer = Spiking(nn.Sequential(nn.Conv2d(256, 128, 1, bias=False),
                                                 self._norm_layer(128),
                                                 IF()), T)
        
        self.reorg = Wrapper(reorg_layer(stride=2))
        
        
        self.head4 = Spiking(nn.Sequential(nn.Conv2d(1024+128*4, 1024, 3, padding=1, bias=False),
                                           self._norm_layer(1024),
                                           IF()), T)        
        
        
        self.pred = last_Spiking(nn.Conv2d(1024, self.num_anchors*(1 + 4 + self.num_classes), 1), T)
        

                
    def forward(self, x):
        #generate spike trains
        x.unsqueeze_(1)
        x = x.repeat(1, self.T, 1, 1, 1)    
        
        x = self.layer0(x)
        x = self.relu0(x)

        x = self.layer1(x)  # 4
        x = self.layer2(x)  # 8
        x = self.layer3(x)  # 16
        c4 = x
        x = self.layer4(x)  # 32
        c5 = x
        
        ##convs1
        c5 = self.head1(c5)
        c5 = self.head2(c5)
        c5 = self.head3(c5)
        
        c4 = self.route_layer(c4)
        c4 = self.reorg(c4)
        
        c5 = torch.cat([c4, c5], dim=2)
        
        #convs2
        c5 = self.head4(c5)
        
        x = self.pred(c5)

        return x


# net = MYResNet(arch='resnet34',bit=32, pretrain=False)
# sample = torch.randn(10,3,224,224)
# net(sample)

