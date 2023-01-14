from .resnet import build_resnet
from .darknet19 import build_darknet19
from .darknet53 import build_darknet53
from .darknet_tiny import build_darknet_tiny
from .darknet_tiny_v2 import build_darknet_tiny_v2


def build_backbone(model_name='resnet18', pretrained=False):
    if 'resnet' in model_name:
        backbone = build_resnet(model_name, pretrained)

    elif model_name == 'darknet19':
        backbone = build_darknet19(pretrained)

    elif model_name == 'darknet53':
        backbone = build_darknet53(pretrained)

    elif model_name == 'darknet_tiny':
        backbone = build_darknet_tiny(pretrained)

    elif model_name == 'darknet_tiny_v2':
        backbone = build_darknet_tiny_v2(pretrained)
                        
    return backbone
