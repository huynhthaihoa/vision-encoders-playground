import torch.nn as nn
import os
import contextlib
import shutil

from copy import deepcopy
from pathlib import Path

from .yolov9.common import *
from .yolov9.yolo import *

from .base_encoder import BaseEncoder
from function_utils import replace_layers
from class_utils import CustomSiLU

class YOLOv9(BaseEncoder):
    # YOLO detection model
    def __init__(self, 
        architecture='yolov9-c',
        pretrained=True, 
        finetune=False, 
        # out_dimList = [128, 256, 512, 1024],                
        replace_silu=False, 
        use_customsilu=False,  
        ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(YOLOv9, self).__init__(finetune)
        
        if architecture == 'yolov9-c':
            cfg = 'encoders/yolov9/config/yolov9-c.yaml'
        elif architecture == 'yolov9-e':
            cfg = 'encoders/yolov9/config/yolov9-e.yaml'
        elif architecture == 'yolov9-t':
            cfg = 'encoders/yolov9/config/yolov9-t.yaml'
            self.reverse = True
        elif architecture == 'yolov9-s':
            cfg = 'encoders/yolov9/config/yolov9-s.yaml'
            self.reverse = True
        elif architecture == 'gelan-c':
            cfg = 'encoders/yolov9/config/gelan-c.yaml'
        elif architecture == 'gelan-e':
            cfg = 'encoders/yolov9/config/gelan-e.yaml'
        
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            # LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            # LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, feats = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        if replace_silu:
            if use_customsilu:
                replace_layers(self, torch.nn.SiLU, CustomSiLU())
            else:
                replace_layers(self, torch.nn.SiLU, torch.nn.ReLU())

        self.model = self.model[:-1]
        self.anchors = self.yaml['anchors']
        # print(self.anchors)
        
        self.dimList = feats[-self.anchors:]

        if self.reverse:
            self.dimList = self.dimList[::-1]
        
        # self.make_conv_convert_list(out_dimList)  
        
        if pretrained:
            if os.path.exists('models'):
                shutil.rmtree('models')
            shutil.copytree('encoders/yolov9', 'models')
            # os.makedirs('models')#, exist_ok=True)
            # src_files = os.listdir('encoders/yolov9/')
            # for file_name in src_files:
            #     src_file_name = os.path.join('encoders/yolov9/', file_name)
            #     if os.path.isfile(src_file_name):
            #         dst_file_name = os.path.join('models', file_name)
            #         copy(src_file_name, dst_file_name)
            ckpt_path = f"https://github.com/WongKinYiu/yolov9/releases/download/v0.1/{architecture}.pt"
            ckpt_name = os.path.basename(ckpt_path)
            if not os.path.exists(ckpt_name):
                os.system(f"wget {ckpt_path}")
            dst = self.state_dict()
            src = torch.load(ckpt_name, 'cpu')['model'].state_dict()            
            ckpt = {}
            for k, v in src.items():
                if k in dst and v.shape == dst[k].shape:
                    print(k)
                    ckpt[k] = v
            self.load_state_dict(state_dict=ckpt)#, strict=False)
            shutil.rmtree('models')
        self._freeze_stages()


    def forward(self, x):#, profile=False, visualize=False):
        y = [] # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        feats = []
        for feat in y:
            if feat is not None:
                feats.append(feat)

        if self.reverse:
            feats = feats[::-1]
        
        feats = feats[-self.anchors:]
        # if self.conv_convert_list is not None:
        #     converted_feats = list()
        #     for i, feature in enumerate(feats):
        #         converted_feat = self.conv_convert_list[i](feature)
        #         converted_feats.append(converted_feat)
        #     return converted_feats        
        return feats

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary
    # LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        RepConvN.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        # LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    feats = []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # if len(args) > 0:
        #     feats.append(args[0])

        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv, AConv, ConvTranspose, 
            Bottleneck, SPP, SPPF, DWConv, BottleneckCSP, nn.ConvTranspose2d, DWConvTranspose2d, SPPCSPC, ADown,
            ELAN1, RepNCSPELAN4, SPPELAN}:
            if m is RepNCSPELAN4:
                feats.append(args[0])
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, SPPCSPC}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        # TODO: channel, gw, gd
        elif m in {Detect, DualDetect, TripleDetect, DDetect, DualDDetect, TripleDDetect, Segment, Panoptic}:
            args.append([ch[x] for x in f])
            # if isinstance(args[1], int):  # number of anchors
            #     args[1] = [list(range(args[1] * 2))] * len(f)
            if m in {Segment, Panoptic}:
                args[2] = make_divisible(args[2] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        # LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), feats

