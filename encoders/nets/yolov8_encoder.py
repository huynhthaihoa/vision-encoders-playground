'''
Reference: https://github.com/jahongir7174/YOLOv8-pt
'''

import torch
import os

from shutil import copyfile

from ...function_utils import fuse_conv
from ...class_utils import CustomSiLU
from ..base_encoder import BaseEncoder

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, silu_opt=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, (k - 1) // 2, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        if silu_opt == 0: #SILU
            self.relu = torch.nn.SiLU(inplace=True)
        elif silu_opt == 1: #RELU
            self.relu = torch.nn.ReLU(inplace=True)
        elif silu_opt == 2: #CustomSILU
            self.relu = CustomSiLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))

class Residual(torch.nn.Module):
    def __init__(self, ch, add=True, silu_opt=0):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3, silu_opt=silu_opt),
                                         Conv(ch, ch, 3, silu_opt=silu_opt))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)

class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True, silu_opt=0):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, silu_opt=silu_opt)
        self.conv2 = Conv(in_ch, out_ch // 2, silu_opt=silu_opt)
        self.conv3 = Conv((2 + n) * out_ch // 2, out_ch, silu_opt=silu_opt)
        self.res_m = torch.nn.ModuleList(Residual(out_ch // 2, add) for _ in range(n))

    def forward(self, x):
        y = [self.conv1(x), self.conv2(x)]
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv3(torch.cat(y, dim=1))

class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5, keepnum_maxpool=[False, False, False], silu_opt=0):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, silu_opt=silu_opt)
        self.conv2 = Conv(in_ch * 2, out_ch, silu_opt=silu_opt)
        self.k = k
        self.res_m = torch.nn.MaxPool2d(self.k, 1, self.k // 2) # original: max_pool with k = 5, s = 1
        if type(keepnum_maxpool) is str:
            self.keepnum_maxpool = list()
            keepnum_maxpool = keepnum_maxpool.split()            
            for elem in keepnum_maxpool:
                if elem == "True":
                    self.keepnum_maxpool.append(True)
                elif elem == "False":
                    self.keepnum_maxpool.append(False)
        else:
            self.keepnum_maxpool = keepnum_maxpool
        
    def forward(self, x):
        x = self.conv1(x)
        
        y1 = self.res_m(x)
        if self.k == 3 and self.keepnum_maxpool[0] is False:
            y1 = self.res_m(y1)
        
        y2 = self.res_m(y1)
        if self.k == 3 and not self.keepnum_maxpool[1]:
            y2 = self.res_m(y2)
            
        y3 = self.res_m(y2)
        if self.k == 3 and not self.keepnum_maxpool[2]:
            y3 = self.res_m(y3)
        
        return self.conv2(torch.cat([x, y1, y2, y3], 1))

class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, replace_maxpool=False, keepnum_maxpool=[False, False, False], silu_opt=0, use_5_feat=False):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []
        self.p1.append(Conv(width[0], width[1], 3, 2, silu_opt=silu_opt))
        self.p2.append(Conv(width[1], width[2], 3, 2, silu_opt=silu_opt))
        self.p2.append(CSP(width[2], width[2], depth[0], silu_opt=silu_opt))
        self.p3.append(Conv(width[2], width[3], 3, 2, silu_opt=silu_opt))
        self.p3.append(CSP(width[3], width[3], depth[1], silu_opt=silu_opt))
        self.p4.append(Conv(width[3], width[4], 3, 2, silu_opt=silu_opt))
        self.p4.append(CSP(width[4], width[4], depth[2], silu_opt=silu_opt))
        self.p5.append(Conv(width[4], width[5], 3, 2, silu_opt=silu_opt))
        self.p5.append(CSP(width[5], width[5], depth[0], silu_opt=silu_opt))
        if replace_maxpool:
            self.p5.append(SPP(width[5], width[5], 3, keepnum_maxpool, silu_opt=silu_opt)) #original is SPP(width[5], width[5], 5)
        else:
            self.p5.append(SPP(width[5], width[5], silu_opt=silu_opt))
        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)
        
        self.use_5_feat = use_5_feat

    def forward(self, x):
        e1 = self.p1(x)
        e2 = self.p2(e1)
        e3 = self.p3(e2)
        e4 = self.p4(e3)
        e5 = self.p5(e4)
        if self.use_5_feat:
            return e1, e2, e3, e4, e5
        return e2, e3, e4, e5

class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, silu_opt=0, use_5_feat=False):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False, silu_opt=silu_opt)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False, silu_opt=silu_opt)
        self.h3 = Conv(width[3], width[3], 3, 2, silu_opt=silu_opt)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False, silu_opt=silu_opt)
        self.h5 = Conv(width[4], width[4], 3, 2, silu_opt=silu_opt)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False, silu_opt=silu_opt)
        self.use_5_feat = use_5_feat

    def forward(self, x):
        if self.use_5_feat:
            n1, n2, e3, e4, e5 = x
        else:
            n2, e3, e4, e5 = x
        n = self.h1(torch.cat([self.up(e5), e4], 1))
        n3 = self.h2(torch.cat([self.up(n), e3], 1))
        n4 = self.h4(torch.cat([self.h3(n3), n], 1))
        n5 = self.h6(torch.cat([self.h5(n4), e5], 1))
        if self.use_5_feat:
            return n1, n2, n3, n4, n5
        return n2, n3, n4, n5

# class DFL(torch.nn.Module):
#     # Generalized Focal Loss
#     # https://ieeexplore.ieee.org/document/9792391
#     def __init__(self, ch=16):
#         super().__init__()
#         self.ch = ch
#         self.conv = torch.nn.Conv2d(ch, 1, 1, bias=False).requires_grad_(False)
#         x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
#         self.conv.weight.data[:] = torch.nn.Parameter(x)

#     def forward(self, x):
#         b, c, a = x.shape
#         x = x.view(b, 4, self.ch, a).transpose(2, 1)
#         return self.conv(x.softmax(1)).view(b, 4, a)

class YOLOv8(BaseEncoder):
    def __init__(self, architecture, pretrained=False, out_dimList = [], finetune=False, keepnum_maxpool=[False, False, False], replace_silu=False, use_customsilu=False, use_5_feat=False):
        super(YOLOv8, self).__init__(finetune)
        
        if architecture.find('yolov8n') != -1:
            depth = [1, 2, 2]
            width = [3, 16, 32, 64, 128, 256]
        elif architecture.find('yolov8s') != -1:
            if architecture.find('yolov8sw') != -1:
                depth = [1, 2, 2]
                width = [3, 48, 96, 192, 384, 576]
            elif architecture.find('yolov8sd') != -1:
                depth = [2, 4, 4]
                width = [3, 32, 64, 128, 256, 512]
            else:
                depth = [1, 2, 2]
                width = [3, 32, 64, 128, 256, 512]
        elif architecture.find('yolov8m') != -1:
            depth = [2, 4, 4]
            width = [3, 48, 96, 192, 384, 576]
        elif architecture.find('yolov8l') != -1:
            depth = [3, 6, 6]
            width = [3, 64, 128, 256, 512, 512]
        elif architecture.find('yolov8x') != -1:
            depth = [3, 6, 6]
            width = [3, 80, 160, 320, 640, 640]
        
        if replace_silu is False:
            silu_opt = 0 #SILU
        else:
            if use_customsilu is False:
                silu_opt = 1 #RELU
            else:
                silu_opt = 2 #CustomSILU
        
        if architecture.startswith('ti') is True:
            replace_maxpool = True
        else:
            replace_maxpool = False
        
        self.net = DarkNet(width, depth, replace_maxpool, keepnum_maxpool, silu_opt, use_5_feat)#, replace_silu)
        
        if architecture.endswith('truncate') is True:
            self.fpn = None
        else:
            self.fpn = DarkFPN(width, depth, silu_opt, use_5_feat)#, replace_silu)
            
        # for the decoder
        if use_5_feat:
            self.dimList = width[-5:]
        else:
            self.dimList = width[-4:]
        
        self.make_conv_convert_list(out_dimList)
                
        if pretrained:
            ckpt = f"v8_{architecture[-1]}.pt"
            if not os.path.exists(ckpt):
                os.system(f"wget https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_{architecture[-1]}.pt")
            copyfile("nets/nnv8.py", "nets/nn.py")   
            dst = self.state_dict()
            src = torch.load(ckpt, 'cpu')['model'].float().state_dict()
            ckpt = {}
            for k, v in src.items():
                if k in dst and v.shape == dst[k].shape:
                    print(k)
                    ckpt[k] = v
            self.load_state_dict(state_dict=ckpt, strict=False)
        self._freeze_stages()#finetune)
            
    def forward(self, x):
        x = self.net(x)
        
        if self.fpn is not None:
            x = self.fpn(x)
        
        if self.conv_convert_list is not None:
            out_featList = list()
            for i, feature in enumerate(x):
                converted_feat = self.conv_convert_list[i](feature)
                out_featList.append(converted_feat)
            return out_featList
        
        return x

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self

    # def _freeze_stages(self, finetune):
    #     for module in self.modules():
    #         if isinstance(module, nn.BatchNorm2d):
    #             module.eval() # always freeze BN
    #         else:
    #             for param in module.parameters():
    #                 if self.conv_convert_list is not None and param in self.conv_convert_list.parameters():
    #                     param.requires_grad = True
    #                 else:
    #                     param.requires_grad = finetune