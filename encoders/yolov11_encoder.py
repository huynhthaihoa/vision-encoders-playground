# import math

import os
from shutil import copyfile
import torch

from .base_encoder import BaseEncoder
# from utils.util import make_anchors
from class_utils import CustomSiLU

def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, silu_opt=0, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)
        if silu_opt == 0: #SILU
            self.relu = torch.nn.SiLU(inplace=True)
        elif silu_opt == 1: #RELU
            self.relu = torch.nn.ReLU(inplace=True)
        elif silu_opt == 2: #CustomSILU
            self.relu = CustomSiLU()    
        elif silu_opt == 3:
            self.relu = torch.nn.Identity()  
        # self.relu = activation

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5, silu_opt=0):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), silu_opt=silu_opt, k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, silu_opt=silu_opt, k=3, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch, silu_opt=0):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, silu_opt=silu_opt)
        self.conv2 = Conv(in_ch, out_ch // 2, silu_opt=silu_opt)
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, silu_opt=silu_opt)
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0, silu_opt=silu_opt),
                                         Residual(out_ch // 2, e=1.0, silu_opt=silu_opt))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r, silu_opt=0):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), silu_opt=silu_opt)
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, silu_opt=silu_opt)

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r, silu_opt=silu_opt) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(CSPModule(out_ch // r, out_ch // r, silu_opt=silu_opt) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5, silu_opt=0, keepnum_maxpool=[False, False, False]):
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
        if self.k == 3 and not self.keepnum_maxpool[0]:
            y1 = self.res_m(y1)

        y2 = self.res_m(y1)
        if self.k == 3 and not self.keepnum_maxpool[1]:
            y2 = self.res_m(y2)

        y3 = self.res_m(y2)
        if self.k == 3 and not self.keepnum_maxpool[2]:
            y3 = self.res_m(y3)
        
        return self.conv2(torch.cat(tensors=[x, y1, y2, y3], dim=1))


class Attention(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, silu_opt=3)

        self.conv1 = Conv(ch, ch, silu_opt=3, k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, silu_opt=3)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)


class PSABlock(torch.nn.Module):

    def __init__(self, ch, num_head, silu_opt=0):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = torch.nn.Sequential(Conv(ch, ch * 2, silu_opt=silu_opt),
                                         Conv(ch * 2, ch, silu_opt=3))

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)


class PSA(torch.nn.Module):
    def __init__(self, ch, n, silu_opt=0, use_attention=True):
        super().__init__()
        if use_attention:
            self.conv1 = Conv(ch, 2 * (ch // 2), silu_opt=silu_opt)
            self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128, silu_opt=silu_opt) for _ in range(n)))
        else:
            self.conv1 = self.res_m = None
        self.conv2 = Conv(2 * (ch // 2), ch, silu_opt=silu_opt)

    def forward(self, x):
        if self.conv1 is None:
            return self.conv2(x)
        x, y = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))
    
class PS(torch.nn.Module):
    def __init__(self, ch, silu_opt=0):
        super().__init__()
        self.conv2 = Conv(2 * (ch // 2), ch, silu_opt=silu_opt)
    
    def forward(self, x):
        return self.conv2(x)

class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, csp, silu_opt=0, replace_maxpool=False, keepnum_maxpool=[False, False, False], use_attention=True):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], silu_opt=silu_opt, k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], silu_opt=silu_opt, k=3, s=2, p=1))
        self.p2.append(CSP(width[2], width[3], depth[0], csp[0], r=4, silu_opt=silu_opt))
        # p3/8
        self.p3.append(Conv(width[3], width[3], silu_opt=silu_opt, k=3, s=2, p=1))
        self.p3.append(CSP(width[3], width[4], depth[1], csp[0], r=4, silu_opt=silu_opt))
        # p4/16
        self.p4.append(Conv(width[4], width[4], silu_opt=silu_opt, k=3, s=2, p=1))
        self.p4.append(CSP(width[4], width[4], depth[2], csp[1], r=2, silu_opt=silu_opt))
        # p5/32
        self.p5.append(Conv(width[4], width[5], silu_opt=silu_opt, k=3, s=2, p=1))
        self.p5.append(CSP(width[5], width[5], depth[3], csp[1], r=2, silu_opt=silu_opt))
        # self.p5.append(SPP(width[5], width[5], silu_opt=silu_opt, keepnum_maxpool=keepnum_maxpool))
        
        if replace_maxpool:
            self.p5.append(SPP(width[5], width[5], 3, keepnum_maxpool=keepnum_maxpool, silu_opt=silu_opt)) #original is SPP(width[5], width[5], 5)
        else:
            self.p5.append(SPP(width[5], width[5], silu_opt=silu_opt))

        if use_attention != 0:
            self.p5.append(PSA(width[5], depth[4], silu_opt=silu_opt, use_attention=(use_attention==2)))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        #print("darknet:", p2.shape, p3.shape, p4.shape, p5.shape)
        return p2, p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, csp, silu_opt=0):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp[0], r=2, silu_opt=silu_opt)
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp[0], r=2, silu_opt=silu_opt)
        self.h3 = Conv(width[3], width[3], silu_opt=silu_opt, k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp[0], r=2, silu_opt=silu_opt)
        self.h5 = Conv(width[4], width[4], silu_opt=silu_opt, k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp[1], r=2, silu_opt=silu_opt)

    def forward(self, x):
        p2, p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        #print("fpn:", p2.shape, p3.shape, p4.shape, p5.shape)
        return p2, p3, p4, p5


# class DFL(torch.nn.Module):
#     # Generalized Focal Loss
#     # https://ieeexplore.ieee.org/document/9792391
#     def __init__(self, ch=16):
#         super().__init__()
#         self.ch = ch
#         self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
#         x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
#         self.conv.weight.data[:] = torch.nn.Parameter(x)

#     def forward(self, x):
#         b, c, a = x.shape
#         x = x.view(b, 4, self.ch, a).transpose(2, 1)
#         return self.conv(x.softmax(1)).view(b, 4, a)


# class Head(torch.nn.Module):
#     anchors = torch.empty(0)
#     strides = torch.empty(0)

#     def __init__(self, nc=80, filters=()):
#         super().__init__()
#         self.ch = 16  # DFL channels
#         self.nc = nc  # number of classes
#         self.nl = len(filters)  # number of detection layers
#         self.no = nc + self.ch * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build

#         box = max(64, filters[0] // 4)
#         cls = max(80, filters[0], self.nc)

#         self.dfl = DFL(self.ch)
#         self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
#                                                            Conv(box, box,torch.nn.SiLU(), k=3, p=1),
#                                                            torch.nn.Conv2d(box, out_channels=4 * self.ch,
#                                                                            kernel_size=1)) for x in filters)
#         self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
#                                                            Conv(x, cls, torch.nn.SiLU()),
#                                                            Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
#                                                            Conv(cls, cls, torch.nn.SiLU()),
#                                                            torch.nn.Conv2d(cls, out_channels=self.nc,
#                                                                            kernel_size=1)) for x in filters)

#     def forward(self, x):
#         for i, (box, cls) in enumerate(zip(self.box, self.cls)):
#             x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
#         if self.training:
#             return x

#         self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
#         x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
#         box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

#         a, b = self.dfl(box).chunk(2, 1)
#         a = self.anchors.unsqueeze(0) - a
#         b = self.anchors.unsqueeze(0) + b
#         box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

#         return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

#     def initialize_biases(self):
#         # Initialize biases
#         # WARNING: requires stride availability
#         for box, cls, s in zip(self.box, self.cls, self.stride):
#             # box
#             box[-1].bias.data[:] = 1.0
#             # cls (.01 objects, 80 classes, 640 image)
#             cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class YOLOv11(BaseEncoder):
    def __init__(self, 
                 architecture='yolov11n',
                 pretrained=False,
                #  out_dimList = [],
                 finetune=False,
                 replace_silu=False,
                 use_customsilu=False,
                 keepnum_maxpool=[False, False, False]):#,
                #  width, depth, csp, num_classes):
        super(YOLOv11, self).__init__(finetune)
        
        if architecture.find('yolov11n') != -1:
            variant = 'n'
            csp = [False, True]
            depth = [1, 1, 1, 1, 1, 1]
            width = [3, 16, 32, 64, 128, 256]
        elif architecture.find('yolov11s') != -1:
            variant = 's'
            csp = [False, True]
            depth = [1, 1, 1, 1, 1, 1]
            width = [3, 32, 64, 128, 256, 512]
        elif architecture.find('yolov11m') != -1:
            variant = 'm'
            csp = [True, True]
            depth = [1, 1, 1, 1, 1, 1]
            width = [3, 64, 128, 256, 512, 512]
        elif architecture.find('yolov11l') != -1:
            variant = 'l'
            csp = [True, True]
            depth = [2, 2, 2, 2, 2, 2]
            width = [3, 64, 128, 256, 512, 512]
        elif architecture.find('yolov11x') != -1: 
            variant = 'x'      
            csp = [True, True]
            depth = [2, 2, 2, 2, 2, 2]
            width = [3, 96, 192, 384, 768, 768]
        elif architecture.find('yolov11t') != -1:
            variant = 't'
            csp = [False, True]
            depth = [1, 1, 1, 1, 1, 1]
            width = [3, 24, 48, 96, 192, 384]

        if replace_silu is False:
            silu_opt = 0 #SILU
        else:
            if use_customsilu is False:
                silu_opt = 1 #RELU
            else:
                silu_opt = 2 #CustomSILU            

        if architecture.startswith('ti'):# is True:
            replace_maxpool = True
        else:
            replace_maxpool = False
            
        if architecture.endswith('lite'):
            use_attention = 0
        elif architecture.endswith('conv'):
            use_attention = 1
        else:
            use_attention = 2
        
        self.net = DarkNet(width, depth, csp, silu_opt=silu_opt, replace_maxpool=replace_maxpool, keepnum_maxpool=keepnum_maxpool, use_attention=use_attention)
        self.fpn = DarkFPN(width, depth, csp, silu_opt=silu_opt)

        self.dimList = [width[-3], width[-3], width[-2], width[-1]]

        # self.make_conv_convert_list(out_dimList)

        if pretrained:
            ckpt = f"v11_{variant}.pt"
            if not os.path.exists(ckpt):
                os.system(f"wget https://github.com/jahongir7174/YOLOv11-pt/releases/download/v0.0.1/v11_{variant}.pt")
            copyfile("nets/nnv11.py", "nets/nn.py")   
            dst = self.state_dict()
            src = torch.load(ckpt, 'cpu')['model'].float().state_dict()
            ckpt = {}
            for k, v in src.items():
                if k in dst and v.shape == dst[k].shape:
                    print(k)
                    ckpt[k] = v
            self.load_state_dict(state_dict=ckpt, strict=False)
        self._freeze_stages()#finetune)

        # img_dummy = torch.zeros(1, width[0], 256, 256)
        # self.head = Head(num_classes, (width[3], width[4], width[5]))
        # self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        # self.stride = self.head.stride
        # self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        
        if self.fpn is not None:
            x = self.fpn(x)
        
        # if self.conv_convert_list is not None:
        #     out_featList = list()
        #     for i, feature in enumerate(x):
        #         converted_feat = self.conv_convert_list[i](feature)
        #         out_featList.append(converted_feat)
        #     return out_featList
        #print("dimList:", self.dimList)
        
        return x
    
    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


# def yolo_v11_n(num_classes: int = 80):
#     csp = [False, True]
#     depth = [1, 1, 1, 1, 1, 1]
#     width = [3, 16, 32, 64, 128, 256]
#     return YOLO(width, depth, csp, num_classes)


# def yolo_v11_t(num_classes: int = 80):
#     csp = [False, True]
#     depth = [1, 1, 1, 1, 1, 1]
#     width = [3, 24, 48, 96, 192, 384]
#     return YOLO(width, depth, csp, num_classes)


# def yolo_v11_s(num_classes: int = 80):
#     csp = [False, True]
#     depth = [1, 1, 1, 1, 1, 1]
#     width = [3, 32, 64, 128, 256, 512]
#     return YOLO(width, depth, csp, num_classes)


# def yolo_v11_m(num_classes: int = 80):
#     csp = [True, True]
#     depth = [1, 1, 1, 1, 1, 1]
#     width = [3, 64, 128, 256, 512, 512]
#     return YOLO(width, depth, csp, num_classes)


# def yolo_v11_l(num_classes: int = 80):
#     csp = [True, True]
#     depth = [2, 2, 2, 2, 2, 2]
#     width = [3, 64, 128, 256, 512, 512]
#     return YOLO(width, depth, csp, num_classes)


# def yolo_v11_x(num_classes: int = 80):
#     csp = [True, True]
#     depth = [2, 2, 2, 2, 2, 2]
#     width = [3, 96, 192, 384, 768, 768]
#     return YOLO(width, depth, csp, num_classes)