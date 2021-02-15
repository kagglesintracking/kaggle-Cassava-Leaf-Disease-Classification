import torch.nn as nn
import timm
from utils import *
from conf import *

class CustomTimmModel(nn.Module):
    def __init__(self, backbone, out_dim, pool_type=None, pretrained=True):
        super().__init__()
        self.model = timm.create_model(backbone, pretrained=pretrained)
        if 'efficientnet' in backbone or 'densenet' in backbone:
            in_ch = self.model.classifier.in_features
            self.model.classifier = nn.Identity() 
        elif 'resnext' in backbone or 'resnet' in backbone or 'resnest' in backbone:
            in_ch = self.model.fc.in_features
            self.model.fc = nn.Identity() 
        elif 'vit' in backbone:
            in_ch = self.model.head.in_features
            self.model.head = nn.Identity()
        elif 'csp' in backbone:
            in_ch = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()
        elif 'nf' in backbone:
            in_ch = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()
        
        if pool_type == 'gem':
            print('Using generalized mean pooling(GEM).')
            if 'vit' in backbone:
                raise ValueError('Vit does not support GEM. Please set pool_type == None')
            elif 'csp' in backbone:
                self.model.head.global_pool = nn.Sequential(GeM(), nn.Flatten())
            else:
                try:
                    self.model.global_pool = nn.Sequential(GeM(), nn.Flatten())
                except:
                    raise ValueError('Backbone not supported.')
            
        self.myfc = nn.Linear(in_ch, out_dim)

    def forward(self, x):
        x = self.model(x)
        x = self.myfc(x)
        return x

## V2 extracts features before pooling layer
class CustomTimmModelV2(nn.Module):
    def __init__(self, backbone, out_dim, pool_type=None, pretrained=True):
        super().__init__()
        print('Running V2 that extracts features.')
        self.model = timm.create_model(backbone, pretrained=pretrained)
        if 'efficientnet' in backbone or 'densenet' in backbone:
            in_ch = self.model.classifier.in_features
            self.model.classifier = nn.Identity() 
        elif 'resnext' in backbone or 'resnet' in backbone or 'resnest' in backbone:
            in_ch = self.model.fc.in_features
            self.model.fc = nn.Identity() 
        elif 'vit' in backbone:
            in_ch = self.model.head.in_features
            self.model.head = nn.Identity()
        elif 'csp' in backbone:
            in_ch = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()
        
        # delete global pooling layer
        if 'vit' in backbone:
            raise NotImplementedError
        elif 'csp' in backbone:
            self.model.head.global_pool = nn.Identity()
        else:
            try:
                self.model.global_pool = nn.Identity()
            except:
                raise ValueError('Backbone not supported.')
        
        if pool_type == 'gem':
            self.pool = nn.Sequential(GeM(), nn.Flatten())
        elif pool_type == 'mean':
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        
        self.myfc = nn.Linear(in_ch, out_dim)

    def forward(self, x):
        features = self.model(x)
        x = self.pool(features)
        x = self.myfc(x)
        return x, features

def get_net(**kwargs):
    try:
        if args.snap_mix:
            return CustomTimmModelV2(**kwargs)
        else:
            return CustomTimmModel(**kwargs)
    except:
        print('Model not loaded.')   

CustomTimmModel(**args.net_params)
