#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import os
import torch

from .yolov6.common import *
from .yolov6.efficientrep import *
from .yolov6.reppan import *
from .yolov6.config import Config

from .base_encoder import BaseEncoder
from class_utils import CustomSiLU
from function_utils import replace_layers

class YOLOv6(BaseEncoder):
    # export = False
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, architecture='yolov6s', 
                       pretrained=False, 
                       finetune=False, 
                    #    out_dimList = [], 
                    #    use_5_feat=False,
                       replace_silu=False, 
                       use_customsilu=False,
                       channels=3):#, num_classes=None, fuse_ab=False, distill_ns=False):  # model, input channels, number of classes
        super(YOLOv6, self).__init__(finetune)
        
        self.is_p6 = False
        
        # Build network
        if architecture.find('yolov6s') != -1:
            if architecture.find('yolov6s6') != -1:
                self.is_p6 = True
                config = Config.fromfile('encoders/yolov6/configs/yolov6s6.py')
            else:
                config = Config.fromfile('encoders/yolov6/configs/yolov6s.py')
        elif architecture.find('yolov6n') != -1:
            if architecture.find('yolov6n6') != -1:
                self.is_p6 = True
                config = Config.fromfile('encoders/yolov6/configs/yolov6n6.py')
            else:
                config = Config.fromfile('encoders/yolov6/configs/yolov6n.py')
        elif architecture.find('yolov6m') != -1:
            if architecture.find('yolov6m6') != -1:
                self.is_p6 = True
                config = Config.fromfile('encoders/yolov6/configs/yolov6m6.py')
            else:
                config = Config.fromfile('encoders/yolov6/configs/yolov6m.py')
        elif architecture.find('yolov6l') != -1:
            if architecture.find('yolov6l6') != -1:
                self.is_p6 = True
                config = Config.fromfile('encoders/yolov6/configs/yolov6l6.py')
            else:
                config = Config.fromfile('encoders/yolov6/configs/yolov6l.py')
        
        if not hasattr(config, 'training_mode'):
            setattr(config, 'training_mode', 'repvgg')
            
        # num_layers = config.model.head.num_layers
        self.backbone, self.neck, channels_list, ckpt_path = build_network(config, channels)#, num_classes, num_layers, fuse_ab=fuse_ab, distill_ns=distill_ns)
        
        if replace_silu:
            if use_customsilu:
                replace_layers(self, nn.SiLU, CustomSiLU())
            else:
                replace_layers(self, nn.SiLU, nn.ReLU())
        
        #for the decoder
        if architecture.find('truncate') != -1: #backbone only
            self.neck = None
            self.dimList = channels_list[:5]
        else: #backbone + neck
            if self.is_p6:
                self.dimList = [channels_list[0], channels_list[8], channels_list[9], channels_list[10], channels_list[11]] #channels_list[7:12]
            else:
                self.dimList = [channels_list[0], channels_list[1], channels_list[6], channels_list[8], channels_list[10]]
            
        # if not use_5_feat:
        #     # if self.is_p6:
        #     #     self.dimList = self.dimList[:4]
        #     # else:
        #     self.dimList = self.dimList[1:]
                                        
        # self.make_conv_convert_list(out_dimList)     
        
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

        # if self.is_p6:
        #     x = x[:-1]
            
        if len(self.dimList) == 4:
            x = x[1:]
                    
        # if self.conv_convert_list is not None:
        #     out_featList = list()
        #     for i, feature in enumerate(x):
        #         converted_feat = self.conv_convert_list[i](feature)
        #         out_featList.append(converted_feat)
        #     return out_featList

        return x

def make_divisible(x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def build_network(config, channels):#, num_classes, num_layers, fuse_ab=False, distill_ns=False):
    depth_mul = config.model.depth_multiple
    width_mul = config.model.width_multiple
    num_repeat_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_P2 = config.model.backbone.get('fuse_P2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeat_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    num_repeat = [(max(round(i * depth_mul), 1) if i > 1 else i) for i in (num_repeat_backbone + num_repeat_neck)]
    channels_list = [make_divisible(i * width_mul, 8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)
    
    ckpt_path = config.model.pretrained

    if 'CSP' in config.model.backbone.type:

        if "stage_block_type" in config.model.backbone:
            stage_block_type = config.model.backbone.stage_block_type
        else:
            stage_block_type = "BepC3"  #default

        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.backbone.csp_e,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf,
            stage_block_type=stage_block_type
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            csp_e=config.model.neck.csp_e,
            stage_block_type=stage_block_type
        )
    else:
        backbone = BACKBONE(
            in_channels=channels,
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block,
            fuse_P2=fuse_P2,
            cspsppf=cspsppf
        )

        neck = NECK(
            channels_list=channels_list,
            num_repeats=num_repeat,
            block=block
        )

    # if distill_ns:
    #     from yolov6.models.heads.effidehead_distill_ns import Detect, build_effidehead_layer
    #     if num_layers != 3:
    #         LOGGER.error('ERROR in: Distill mode not fit on n/s models with P6 head.\n')
    #         exit()
    #     head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max)
    #     head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    # elif fuse_ab:
    #     from yolov6.models.heads.effidehead_fuseab import Detect, build_effidehead_layer
    #     anchors_init = config.model.head.anchors_init
    #     head_layers = build_effidehead_layer(channels_list, 3, num_classes, reg_max=reg_max, num_layers=num_layers)
    #     head = Detect(num_classes, anchors_init, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    # else:
    #     from yolov6.models.effidehead import Detect, build_effidehead_layer
    #     head_layers = build_effidehead_layer(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
    #     head = Detect(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, channels_list, ckpt_path#, head


# def build_model(cfg, device):#num_classes, ):#, fuse_ab=False, distill_ns=False):
#     model = YOLOv6(cfg, channels=3).to(device)
#     return model
