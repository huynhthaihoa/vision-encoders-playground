import torch.nn as nn
import os
import shutil 

from copy import deepcopy
from pathlib import Path

from .yolov7.common import *
from .yolov7.yolo import *
from .yolov7.general import make_divisible

from .base_encoder import BaseEncoder
from function_utils import replace_layers
from class_utils import CustomSiLU

class YOLOv7(BaseEncoder):
    # YOLO detection model
    def __init__(self, 
        architecture='yolov7',
        pretrained=True, 
        finetune=False, 
        # out_dimList = [],
        replace_silu=False, 
        use_customsilu=False,                
        ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(YOLOv7, self).__init__(finetune)
        
        if architecture == 'yolov7':
            cfg = 'encoders/yolov7/config/yolov7.yaml'
        elif architecture == 'yolov7x':
            cfg = 'encoders/yolov7/config/yolov7x.yaml'
        elif architecture == 'yolov7-w6':
            cfg = 'encoders/yolov7/config/yolov7-w6.yaml'
        elif architecture == 'yolov7-tiny':
            cfg = 'encoders/yolov7/config/yolov7-tiny.yaml'
        elif architecture == 'yolov7-e6e':
            cfg = 'encoders/yolov7/config/yolov7-e6e.yaml'
        elif architecture == 'yolov7-e6':
            cfg = 'encoders/yolov7/config/yolov7-e6.yaml'
        elif architecture == 'yolov7-d6':
            cfg = 'encoders/yolov7/config/yolov7-d6.yaml'
        
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, feats = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        if replace_silu:
            if use_customsilu:
                replace_layers(self, torch.nn.SiLU, CustomSiLU())
            else:
                replace_layers(self, torch.nn.SiLU, torch.nn.ReLU())
        
        self.model = self.model[:-1]
        self.anchors = len(self.yaml['anchors'])
        
        self.dimList = feats[-self.anchors:]
        
        # self.make_conv_convert_list(out_dimList)  
        
        if pretrained:
            if os.path.exists('models'):
                shutil.rmtree('models')
            # os.makedirs('models')#, exist_ok=True)
            shutil.copytree('encoders/yolov7', 'models')
            # src_files = os.listdir('encoders/yolov7/')
            # for file_name in src_files:
            #     src_file_name = os.path.join('encoders/yolov7/', file_name)
            #     if os.path.isfile(src_file_name):
            #         dst_file_name = os.path.join('models', file_name)
            #         shutil.copy(src_file_name, dst_file_name)
            ckpt_path = f"https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{architecture}.pt"
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
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced = False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IAuxDetect) or isinstance(m, IKeypoint):
                    break

            x = m(x)  # run
            
            y.append(x if m.i in self.save else None)  # save output

        feats = y[-self.anchors:]
        # if self.conv_convert_list is not None:
        #     converted_feats = list()
        #     for i, feature in enumerate(feats):
        #         converted_feat = self.conv_convert_list[i](feature)
        #         converted_feats.append(converted_feat)
        #     return converted_feats        
        return feats
    
# def make_divisible(x, divisor):
#     # Returns nearest x divisible by divisor
#     if isinstance(divisor, torch.Tensor):
#         divisor = int(divisor.max())  # to int
#     return math.ceil(x / divisor) * divisor

def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    feats = []
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # print(f"{i}: {args}")
        if len(args) > 0:
            feats.append(args[0])
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RobustConv, RobustConv2, DWConv, GhostConv, RepConv, RepConv_OREPA, DownC, 
                 SPP, SPPF, SPPCSPC, GhostSPPCSPC, MixConv2d, Focus, Stem, GhostStem, CrossConv, 
                 Bottleneck, BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                 RepBottleneck, RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC,  
                 Res, ResCSPA, ResCSPB, ResCSPC, 
                 RepRes, RepResCSPA, RepResCSPB, RepResCSPC, 
                 ResX, ResXCSPA, ResXCSPB, ResXCSPC, 
                 RepResX, RepResXCSPA, RepResXCSPB, RepResXCSPC, 
                 Ghost, GhostCSPA, GhostCSPB, GhostCSPC,
                 SwinTransformerBlock, STCSPA, STCSPB, STCSPC,
                 SwinTransformer2Block, ST2CSPA, ST2CSPB, ST2CSPC]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [DownC, SPPCSPC, GhostSPPCSPC, 
                     BottleneckCSPA, BottleneckCSPB, BottleneckCSPC, 
                     RepBottleneckCSPA, RepBottleneckCSPB, RepBottleneckCSPC, 
                     ResCSPA, ResCSPB, ResCSPC, 
                     RepResCSPA, RepResCSPB, RepResCSPC, 
                     ResXCSPA, ResXCSPB, ResXCSPC, 
                     RepResXCSPA, RepResXCSPB, RepResXCSPC,
                     GhostCSPA, GhostCSPB, GhostCSPC,
                     STCSPA, STCSPB, STCSPC,
                     ST2CSPA, ST2CSPB, ST2CSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Chuncat:
            c2 = sum([ch[x] for x in f])
        elif m is Shortcut:
            c2 = ch[f[0]]
        elif m is Foldcut:
            c2 = ch[f] // 2
        elif m in [Detect, IDetect, IAuxDetect, IBin, IKeypoint]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is ReOrg:
            c2 = ch[f] * 4
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), feats[:-1]


