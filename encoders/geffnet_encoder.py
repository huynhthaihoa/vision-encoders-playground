"""gen-efficientnet-pytorch-based encoders: https://github.com/rwightman/gen-efficientnet-pytorch"""

import geffnet
import torch.nn as nn

from function_utils import replace_layers
from class_utils import CustomSiLU
from .base_encoder import BaseEncoder

class EfficientNet(BaseEncoder):
    def __init__(self, architecture="efficientnet_b0", pretrained=True, finetune=False, out_dimList = [128, 256, 512, 1024], replace_silu=False, use_customsilu=False, use_5_feat=False):
        super(EfficientNet, self).__init__(finetune)
        
        if architecture.find("0") != -1:
            self.encoder = geffnet.tf_efficientnet_b0_ns(pretrained=pretrained)
            self.dimList = [24, 40, 112, 1280]
        elif architecture.find("1") != -1:
            self.encoder = geffnet.tf_efficientnet_b1_ns(pretrained=pretrained)
            self.dimList = [24, 40, 112, 1280]
        elif architecture.find("2") != -1:
            self.encoder = geffnet.tf_efficientnet_b2_ns(pretrained=pretrained)
            self.dimList = [24, 48, 120, 1408]
        elif architecture.find("3") != -1:
            self.encoder = geffnet.tf_efficientnet_b3_ns(pretrained=pretrained)
            self.dimList = [32, 48, 136, 1536]
        elif architecture.find("4") != -1:
            self.encoder = geffnet.tf_efficientnet_b4_ns(pretrained=pretrained)
            self.dimList = [32, 56, 160, 1792]
        elif architecture.find("5") != -1:
            self.encoder = geffnet.tf_efficientnet_b5_ns(pretrained=pretrained)
            self.dimList = [40, 64, 176, 2048]
        elif architecture.find("6") != -1:
            self.encoder = geffnet.tf_efficientnet_b6_ns(pretrained=pretrained)
            self.dimList = [40, 72, 200, 2304]
        elif architecture.find("7") != -1:
            self.encoder = geffnet.tf_efficientnet_b7_ns(pretrained=pretrained)
            self.dimList = [48, 80, 224, 2560]
            
        if replace_silu:
            if use_customsilu:
                replace_layers(self.encoder, nn.SiLU, CustomSiLU())
            else:
                replace_layers(self.encoder, nn.SiLU, nn.ReLU(inplace=True)) #CustomSiLU()
                
        del self.encoder.global_pool
        del self.encoder.classifier
        
        self.block_idx = [4, 5, 7, 11] #5th feature is extracted after bn2
        # after passing blocks[4]    : H/4  x W/4
        # after passing blocks[5]    : H/8  x W/8
        # after passing blocks[7]    : H/16 x W/16
        # after passing conv_stem    : H/32 x W/32
        
        if use_5_feat:
            self.dimList.insert(0, 16)
            self.block_idx.insert(0, 3)
        
        self.make_conv_convert_list(out_dimList)
    
        self._freeze_stages()
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        block_cnt = 0
        
        for k, v in self.encoder._modules.items():
            if k == 'act2':
                break
            if k == 'blocks':
                for m, n in v._modules.items():
                    feature = n(feature)
                    if self.block_idx[block_cnt] == cnt:
                        if self.conv_convert_list is None:
                            out_featList.append(feature)
                        else:
                            converted_feat = self.conv_convert_list[block_cnt](feature)
                            out_featList.append(converted_feat)
                        block_cnt += 1
                    cnt += 1
            else:
                feature = v(feature)
                if self.block_idx[block_cnt] == cnt:
                    if self.conv_convert_list is None:
                        out_featList.append(feature)
                    else:
                        converted_feat = self.conv_convert_list[block_cnt](feature)
                        out_featList.append(converted_feat)
                    block_cnt += 1
                cnt += 1        
            
        return out_featList
    
    # def train(self, mode=True):
    #     """Convert the model into training mode while keep layers freezed."""
    #     super(EfficientNet, self).train(mode)

class EfficientNetAdvProp(BaseEncoder):
    def __init__(self, architecture="efficientnetap_b0", pretrained=True, finetune=False, out_dimList = [128, 256, 512, 1024], replace_silu=False, use_customsilu=False, use_5_feat=False):
        super(EfficientNetAdvProp, self).__init__(finetune)
        
        if architecture.find("0") != -1:
            self.encoder = geffnet.tf_efficientnet_b0_ap(pretrained=pretrained)
            self.dimList = [24, 40, 112, 1280]
        elif architecture.find("1") != -1:
            self.encoder = geffnet.tf_efficientnet_b1_ap(pretrained=pretrained)
            self.dimList = [24, 40, 112, 1280]
        elif architecture.find("2") != -1:
            self.encoder = geffnet.tf_efficientnet_b2_ap(pretrained=pretrained)
            self.dimList = [24, 48, 120, 1408]
        elif architecture.find("3") != -1:
            self.encoder = geffnet.tf_efficientnet_b3_ap(pretrained=pretrained)
            self.dimList = [32, 48, 136, 1536]
        elif architecture.find("4") != -1:
            self.encoder = geffnet.tf_efficientnet_b4_ap(pretrained=pretrained)
            self.dimList = [32, 56, 160, 1792]
        elif architecture.find("5") != -1:
            self.encoder = geffnet.tf_efficientnet_b5_ap(pretrained=pretrained)
            self.dimList = [40, 64, 176, 2048]
        elif architecture.find("6") != -1:
            self.encoder = geffnet.tf_efficientnet_b6_ap(pretrained=pretrained)
            self.dimList = [40, 72, 200, 2304]
        elif architecture.find("7") != -1:
            self.encoder = geffnet.tf_efficientnet_b7_ap(pretrained=pretrained)
            self.dimList = [48, 80, 224, 2560]

        if replace_silu:
            if use_customsilu:
                replace_layers(self.encoder, nn.SiLU, CustomSiLU()) #CustomSiLU()
            else:
                replace_layers(self.encoder, nn.SiLU, nn.ReLU(inplace=True))
                
        del self.encoder.global_pool
        del self.encoder.classifier
        
        self.block_idx = [4, 5, 7, 11] #5th feature is extracted after bn2
        
        if use_5_feat:
            self.dimList.insert(0, 16)
            self.block_idx.insert(0, 3)
        
        self.make_conv_convert_list(out_dimList)
        
        self._freeze_stages()
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        block_cnt = 0
        
        for k, v in self.encoder._modules.items():
            if k == 'act2':
                break
            if k == 'blocks':
                for m, n in v._modules.items():
                    feature = n(feature)
                    if self.block_idx[block_cnt] == cnt:
                        if self.conv_convert_list is None:
                            out_featList.append(feature)
                        else:
                            converted_feat = self.conv_convert_list[block_cnt](feature)
                            out_featList.append(converted_feat)
                        block_cnt += 1
                    cnt += 1
            else:
                feature = v(feature)
                if self.block_idx[block_cnt] == cnt:
                    if self.conv_convert_list is None:
                        out_featList.append(feature)
                    else:
                        converted_feat = self.conv_convert_list[block_cnt](feature)
                        out_featList.append(converted_feat)
                    block_cnt += 1
                cnt += 1         
            
        return out_featList
    
    # def train(self, mode=True):
    #     """Convert the model into training mode while keep layers freezed."""
    #     super(EfficientNetAdvProp, self).train(mode)

class EfficientNetLite(BaseEncoder):
    def __init__(self, architecture="efficientnetlite0", pretrained=True, finetune=False, out_dimList = [128, 256, 512, 1024], replace_silu=False, use_customsilu=False, use_5_feat=False):
        super(EfficientNetLite, self).__init__(finetune)
        
        if architecture.find("0") != -1:
            self.encoder = geffnet.tf_efficientnet_lite0(pretrained=pretrained)
            self.dimList = [24, 40, 112, 1280]
        elif architecture.find("1") != -1:
            self.encoder = geffnet.tf_efficientnet_lite1(pretrained=pretrained)
            self.dimList = [24, 40, 112, 1280]
        elif architecture.find("2") != -1:
            self.encoder = geffnet.tf_efficientnet_lite2(pretrained=pretrained)
            self.dimList = [24, 48, 120, 1280]
        elif architecture.find("3") != -1:
            self.encoder = geffnet.tf_efficientnet_lite3(pretrained=pretrained)
            self.dimList = [32, 48, 136, 1280]
        elif architecture.find("4") != -1:
            self.encoder = geffnet.tf_efficientnet_lite4(pretrained=pretrained)
            self.dimList = [32, 56, 160, 1280]

        if replace_silu:
            if use_customsilu:
                replace_layers(self.encoder, nn.SiLU, CustomSiLU()) #CustomSiLU()
            else:
                replace_layers(self.encoder, nn.SiLU, nn.ReLU(inplace=True))
        # if replace_silu:
        #     replace_layers(self.encoder, nn.ReLU6, nn.ReLU(inplace=True)) #CustomSiLU()
                
        del self.encoder.global_pool
        del self.encoder.classifier

        self.block_idx = [4, 5, 7, 11] #5th feature is extracted after bn2
        
        if use_5_feat:
            self.dimList.insert(0, 16)
            self.block_idx.insert(0, 3)
        
        self.make_conv_convert_list(out_dimList)
        
        self._freeze_stages()
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        block_cnt = 0
        
        for k, v in self.encoder._modules.items():
            if k == 'act2':
                break
            if k == 'blocks':
                for m, n in v._modules.items():
                    feature = n(feature)
                    if self.block_idx[block_cnt] == cnt:
                        if self.conv_convert_list is None:
                            out_featList.append(feature)
                        else:
                            converted_feat = self.conv_convert_list[block_cnt](feature)
                            out_featList.append(converted_feat)
                        block_cnt += 1
                    cnt += 1
            else:
                feature = v(feature)
                if self.block_idx[block_cnt] == cnt:
                    if self.conv_convert_list is None:
                        out_featList.append(feature)
                    else:
                        converted_feat = self.conv_convert_list[block_cnt](feature)
                        out_featList.append(converted_feat)
                    block_cnt += 1
                cnt += 1           
            
        return out_featList
    
    # def train(self, mode=True):
    #     """Convert the model into training mode while keep layers freezed."""
    #     super(EfficientNetLite, self).train(mode)
