import torch
import torch.nn as nn
import timm
import warnings

from .base_encoder import BaseEncoder

from function_utils import replace_layers

class timmEncoder(BaseEncoder):
    """wrapper for every feature extractors from timm: https://huggingface.co/docs/timm/feature_extraction
    """    
    def __init__(self, architecture, pretrained=True, finetune=False, out_dimList = [64, 128, 256, 512, 1024], use_5_feat=False, replace_gelu=False):
        super(timmEncoder, self).__init__(finetune)
        
        if architecture.startswith('timm_'):
            architecture = architecture[architecture.find('_') + 1:]
            
        self.encoder = timm.create_model(architecture, features_only=True, pretrained=pretrained)
        
        if replace_gelu:
            replace_layers(self, timm.layers.activations.GELU, nn.GELU(approximate='tanh'))
        
        self.use_5_feat = False
        
        #find dimList manually
        image = torch.randn((1, 3, 480, 640))
        features = self.encoder(image)
        assert len(features) >= 4
        
        if len(features) > 4:
            if not use_5_feat:
                features = features[-4:] #remove first features
            else:
                features = features[-5:]
                self.use_5_feat = True  
        else:
            if use_5_feat:
                warnings.warn("This encoder doesn't support 5 features")
                
        self.dimList = list()
        for feature in features:
            self.dimList.append(feature.shape[1])
        
        self.make_conv_convert_list(out_dimList)
            
        self._freeze_stages()
    
    def forward(self, x):
        raw_features = self.encoder(x)
        if not self.use_5_feat:
            raw_features = raw_features[-4:] #remove the first one
        if self.conv_convert_list is not None:
            converted_features = list()
            for i, feature in enumerate(raw_features): 
                converted_feature = self.conv_convert_list[i](feature)
                converted_features.append(converted_feature)
            return converted_features
        return raw_features