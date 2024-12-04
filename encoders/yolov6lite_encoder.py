#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import torch

from .yolov6.common import *
from .yolov6.reppan import *
from .yolov6.efficientrep import *
from .yolov6.config import Config

from .base_encoder import BaseEncoder

class YOLOv6Lite(BaseEncoder):
    # export = False
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, architecture='yolov6lites', 
                       pretrained=False, 
                       finetune=False, 
                       out_dimList = [], 
                       use_5_feat=False,
                    #    replace_silu=False, 
                    #    use_customsilu=False, 
                       channels=3):  # model, input channels, number of classes
        super(YOLOv6Lite, self).__init__(finetune)
        # Build network
        if architecture.find('yolov6lites') != -1:
            config = Config.fromfile('networks/encoders/yolov6/configs/yolov6_lite_s.py')
        elif architecture.find('yolov6litem') != -1:
            config = Config.fromfile('networks/encoders/yolov6/configs/yolov6_lite_m.py')
        elif architecture.find('yolov6litel') != -1:
            config = Config.fromfile('networks/encoders/yolov6/configs/yolov6_lite_l.py')
            
        self.backbone, self.neck, backbone_channels, neck_unified_channel, ckpt_path = build_network(config, channels)#, num_classes)

        # if replace_silu:
        #     if use_customsilu:
        #         replace_layers(self, nn.SiLU, CustomSiLU())
        #     else:
        #         replace_layers(self, nn.SiLU, nn.ReLU())

        if architecture.find('truncate') != -1: #backbone only
            self.neck = None
            self.dimList = backbone_channels
        else:
            self.dimList = [neck_unified_channel] * 3
            self.dimList.insert(0, backbone_channels[0])
            self.dimList.insert(1, backbone_channels[1])
        
        if not use_5_feat:
            self.dimList = self.dimList[1:]
        
        self.make_conv_convert_list(out_dimList)     
                
        if pretrained:
            ckpt_name = os.path.basename(ckpt_path)
            if not os.path.exists(ckpt_name):
                os.system(f"wget {ckpt_path}")
            dst = self.state_dict()
            src = torch.load(ckpt_name, 'cpu')['model'].float().state_dict()            
            ckpt = {}
            for k, v in src.items():
                if k in dst and v.shape == dst[k].shape:
                    print(k)
                    ckpt[k] = v
            self.load_state_dict(state_dict=ckpt, strict=False)
                    
        self._freeze_stages()            

    def forward(self, x):
        x = self.backbone(x)
        
        if self.neck is not None:
            x = self.neck(x)

        if len(self.dimList) == 4:
            x = x[1:]
                       
        if self.conv_convert_list is not None:
            out_featList = list()
            for i, feature in enumerate(x):
                converted_feat = self.conv_convert_list[i](feature)
                out_featList.append(converted_feat)
            return out_featList

        return x


    # def _apply(self, fn):
    #     self = super()._apply(fn)
    #     self.detect.stride = fn(self.detect.stride)
    #     self.detect.grid = list(map(fn, self.detect.grid))
    #     return self

def build_network(config, in_channels):
    width_mul = config.model.width_multiple

    num_repeat_backbone = config.model.backbone.num_repeats
    out_channels_backbone = config.model.backbone.out_channels
    scale_size_backbone = config.model.backbone.scale_size
    in_channels_neck = config.model.neck.in_channels
    unified_channels_neck = config.model.neck.unified_channels

    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    
    ckpt_path = config.model.pretrained

    out_channels_backbone = [make_divisible(i * width_mul)
                            for i in out_channels_backbone]
    mid_channels_backbone = [make_divisible(int(i * scale_size_backbone), divisor=8)
                            for i in out_channels_backbone]
    in_channels_neck = [make_divisible(i * width_mul)
                       for i in in_channels_neck]

    backbone = BACKBONE(in_channels,
                        mid_channels_backbone,
                        out_channels_backbone,
                        num_repeat=num_repeat_backbone)
    neck = NECK(in_channels_neck, unified_channels_neck)

    return backbone, neck, out_channels_backbone, unified_channels_neck, ckpt_path


# def build_model(cfg, num_classes, device):
#     model = Model(cfg, channels=3, num_classes=num_classes).to(device)
#     return model

def make_divisible(v, divisor=16):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
