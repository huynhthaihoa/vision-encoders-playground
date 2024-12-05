'''
Reference: https://github.com/jahongir7174/YOLOv5-pt
'''

import os
import torch.nn as nn
import torch
from shutil import copyfile

from function_utils import fuse_conv#, replace_layers
from class_utils import CustomSiLU
from ..base_encoder import BaseEncoder

def pad(k, p):
    if p is None:
        p = k // 2
    return p

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1, silu_opt=0):#, replace_silu=False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, pad(k, p), 1, g, False)
        self.norm = torch.nn.BatchNorm2d(out_ch, 1e-3, 0.03)
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

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, g=1, silu_opt=0):#, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        slice_kernel = 3
        slice_stride = 2
        self.conv_slice = Conv(in_ch, in_ch*4, slice_kernel, slice_stride, p, g, silu_opt)#, act)
        self.conv = Conv(in_ch * 4, out_ch, k, s, p, g, silu_opt)#, act)

    def forward(self, x):  
        x = self.conv_slice(x)
        x = self.conv(x)
        return x

class Residual(torch.nn.Module):
    def __init__(self, ch, add=True, silu_opt=0):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 1, silu_opt=silu_opt),
                                         Conv(ch, ch, 3, silu_opt=silu_opt))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)

class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n=1, add=True, silu_opt=0):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, silu_opt=silu_opt)
        self.conv2 = Conv(in_ch, out_ch // 2, silu_opt=silu_opt)
        self.conv3 = Conv(in_ch=out_ch, out_ch=out_ch, silu_opt=silu_opt)
        self.res_m = torch.nn.Sequential(*[Residual(out_ch // 2, add, silu_opt) for _ in range(n)])

    def forward(self, x):
        return self.conv3(torch.cat((self.res_m(self.conv1(x)), self.conv2(x)), dim=1))

class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5, keepnum_maxpool=[False, False, False], silu_opt=0):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, silu_opt=silu_opt)
        self.conv2 = Conv(in_ch * 2, out_ch, silu_opt=silu_opt)
        self.k = k
        self.res_m = torch.nn.MaxPool2d(self.k, 1, self.k // 2)
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
        if self.k == 3 and not self.keepnum_maxpool[0]:
            y1 = self.res_m(y1)
        
        y2 = self.res_m(y1)
        if self.k == 3 and not self.keepnum_maxpool[1]:
            y2 = self.res_m(y2)
                    
        y3 = self.res_m(y2)
        if self.k == 3 and not self.keepnum_maxpool[2]:
            y3 = self.res_m(y3)
        
        return self.conv2(torch.cat([x, y1, y2, y3], 1))

class TISPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13), silu_opt=0):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, silu_opt=silu_opt)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1, silu_opt=silu_opt)
        num_3x3_maxpool = []
        max_pool_module_list = []
        for pool_kernel in k:
            assert (pool_kernel - 3) % 2 == 0; "Required Kernel size cannot be implemented with kernel_size of 3"
            num_3x3_maxpool = 1 + (pool_kernel-3) // 2
            max_pool_module_list.append(nn.Sequential(*num_3x3_maxpool*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]))
        self.m = nn.ModuleList(max_pool_module_list)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class DarkNet(torch.nn.Module):
    def __init__(self, filters, num_dep, replace_maxpool=False, keepnum_maxpool=[False, False, False], silu_opt=0, use_5_feat=False):
        super().__init__()
        # if replace_maxpool:
        #     p1 = [Focus(filters[0], filters[1], 3)]
        # else:
        p1 = [Conv(filters[0], filters[1], 6, 2, 2, silu_opt=silu_opt)]
        p2 = [Conv(filters[1], filters[2], 3, 2, silu_opt=silu_opt),
              CSP(filters[2], filters[2], num_dep[0], silu_opt=silu_opt)]
        p3 = [Conv(filters[2], filters[3], 3, 2, silu_opt=silu_opt),
              CSP(filters[3], filters[3], num_dep[1], silu_opt=silu_opt)]
        p4 = [Conv(filters[3], filters[4], 3, 2, silu_opt=silu_opt),
              CSP(filters[4], filters[4], num_dep[2], silu_opt=silu_opt)]

        if replace_maxpool:
            p5 = [Conv(filters[4], filters[5], 3, 2, silu_opt=silu_opt),
                CSP(filters[5], filters[5], num_dep[0], silu_opt=silu_opt),
                SPP(filters[5], filters[5], 3, keepnum_maxpool, silu_opt=silu_opt)]     
        else:       
            p5 = [Conv(filters[4], filters[5], 3, 2, silu_opt=silu_opt),
                CSP(filters[5], filters[5], num_dep[0], silu_opt=silu_opt),
                SPP(filters[5], filters[5], silu_opt=silu_opt)]

        self.p1 = torch.nn.Sequential(*p1)
        self.p2 = torch.nn.Sequential(*p2)
        self.p3 = torch.nn.Sequential(*p3)
        self.p4 = torch.nn.Sequential(*p4)
        self.p5 = torch.nn.Sequential(*p5)
        
        # self.use_5_feat = use_5_feat

    def forward(self, x):
        e1 = self.p1(x)
        e2 = self.p2(e1)
        e3 = self.p3(e2)
        e4 = self.p4(e3)
        e5 = self.p5(e4)
        # if self.use_5_feat:
        #     return e1, e2, e3, e4, e5
        return e2, e3, e4, e5

class DarkFPN(torch.nn.Module):
    def __init__(self, filters, num_dep, silu_opt=0):#, use_5_feat=False):
        super().__init__()
        self.up = torch.nn.Upsample(None, 2)
        self.h1 = Conv(filters[5], filters[4], 1, 1, silu_opt=silu_opt)
        self.h2 = CSP(2 * filters[4], filters[4], num_dep[0], False, silu_opt=silu_opt)
        self.h3 = Conv(filters[4], filters[3], 1, 1, silu_opt=silu_opt)
        self.h4 = CSP(2 * filters[3], filters[3], num_dep[0], False, silu_opt=silu_opt)
        self.h5 = Conv(filters[3], filters[3], 3, 2, silu_opt=silu_opt)
        self.h6 = CSP(2 * filters[3], filters[4], num_dep[0], False, silu_opt=silu_opt)
        self.h7 = Conv(filters[4], filters[4], 3, 2, silu_opt=silu_opt)
        self.h8 = CSP(2 * filters[4], filters[5], num_dep[0], False, silu_opt=silu_opt)
        # self.use_5_feat = use_5_feat

    def forward(self, x):
        # if self.use_5_feat:
        #     e1, e2, e3, e4, e5 = x
        # else:
        e2, e3, e4, e5 = x
            
        n1 = self.h1(e5)
        n2 = self.h2(torch.cat([self.up(n1), e4], 1))

        n3 = self.h3(n2)
        n4 = self.h4(torch.cat([self.up(n3), e3], 1))

        n5 = self.h5(n4)
        n6 = self.h6(torch.cat([n5, n3], 1))

        n7 = self.h7(n6)
        n8 = self.h8(torch.cat([n7, n1], 1))

        # if self.use_5_feat:
        #     return e1, e2, n4, n6, n8    
        return e2, n4, n6, n8

class YOLOv5(BaseEncoder):
    def __init__(self, architecture, pretrained=False, finetune=False, keepnum_maxpool=[False, False, False], replace_silu=False, use_customsilu=False):#, out_dimList = [], use_5_feat=False):
        super(YOLOv5, self).__init__(finetune)
        
        if architecture.find('yolov5n') != -1:
            depth = [1, 2, 3]
            width = [3, 16, 32, 64, 128, 256]
        elif architecture.find('yolov5s') != -1:
            if architecture.find('yolov5sw') != -1:
                depth = [1, 2, 3]
                width = [3, 48, 96, 192, 384, 768]
            elif architecture.find('yolov5sd') != -1:
                depth = [2, 4, 4]
                width = [3, 32, 64, 128, 256, 512]
            else:
                depth = [1, 2, 3]
                width = [3, 32, 64, 128, 256, 512]
        elif architecture.find('yolov5m') != -1:
            depth = [2, 4, 6]
            width = [3, 48, 96, 192, 384, 768]
        elif architecture.find('yolov5l') != -1:
            depth = [3, 6, 9]
            width = [3, 64, 128, 256, 512, 1024]
        elif architecture.find('yolov5x') != -1:
            depth = [4, 8, 12]
            width = [3, 80, 160, 320, 640, 1280]

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
                                    
        self.net = DarkNet(width, depth, replace_maxpool, keepnum_maxpool, silu_opt)#, use_5_feat)
                
        if architecture.endswith('truncate') is True:
            self.fpn = None
        else:
            self.fpn = DarkFPN(width, depth, silu_opt)#, use_5_feat)

        # for the decoder
        # if use_5_feat:
        #     self.dimList = width[-5:]
        # else:
        self.dimList = width[-4:]
                    
        # self.make_conv_convert_list(out_dimList)   
             
        if pretrained:
            ckpt = f"v5_{architecture[-1]}.pt"
            if not os.path.exists(ckpt):
                os.system(f"wget https://github.com/jahongir7174/YOLOv5-pt/releases/download/v0.0.1/v5_{architecture[-1]}.pt")
            copyfile("nets/nnv5.py", "nets/nn.py")
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
        
        # if self.conv_convert_list is not None:
        #     out_featList = []
        #     for i, feature in enumerate(x):
        #         converted_feat = self.conv_convert_list[i](feature)
        #         out_featList.append(converted_feat)
        #     return out_featList
        
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


# def yolo_v5_n(num_class: int = 80):
#     depth = [1, 2, 3]
#     width = [3, 16, 32, 64, 128, 256]
#     return YOLO(width, depth, num_class)


# def yolo_v5_s(num_class: int = 80):
#     depth = [1, 2, 3]
#     width = [3, 32, 64, 128, 256, 512]
#     return YOLO(width, depth, num_class)


# def yolo_v5_m(num_class: int = 80):
#     depth = [2, 4, 6]
#     width = [3, 48, 96, 192, 384, 768]
#     return YOLO(width, depth, num_class)


# def yolo_v5_l(num_class: int = 80):
#     depth = [3, 6, 9]
#     width = [3, 64, 128, 256, 512, 1024]
#     return YOLO(width, depth, num_class)


# def yolo_v5_x(num_class: int = 80):
#     depth = [4, 8, 12]
#     width = [3, 80, 160, 320, 640, 1280]
#     return YOLO(width, depth, num_class)