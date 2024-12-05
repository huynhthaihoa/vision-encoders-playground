import os
from torch import nn
import torch

from .base_encoder import BaseEncoder
from class_utils import get_activation

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

# class ResLayer(nn.Module):
#     "Residual layer with `in_channels` inputs."

#     def __init__(self, in_channels: int):
#         super().__init__()
#         mid_channels = in_channels // 2
#         self.layer1 = BaseConv(
#             in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
#         )
#         self.layer2 = BaseConv(
#             mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
#         )

#     def forward(self, x):
#         out = self.layer2(self.layer1(x))
#         return x + out

class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu", split_max_pool_kernel=False,
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        if not split_max_pool_kernel:
            self.m = nn.ModuleList(
                [
                    nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                    for ks in kernel_sizes
                ]
            )
        else:
            max_pool_module_list = []
            for ks in kernel_sizes:
                assert (ks-1)%2 == 0; "kernel cannot be splitted into 3x3 kernels"
                num_3x3_maxpool = (ks-1)//2
                max_pool_module_list.append(nn.Sequential(*num_3x3_maxpool*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]))
            self.m = nn.ModuleList(max_pool_module_list)

        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y

class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=["stem", "dark2", "dark3", "dark4", "dark5"],
        depthwise=False,
        act="silu",
        conv_focus=False,
        split_max_pool_kernel=False,
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        if not conv_focus:
            self.stem = Focus(3, base_channels, ksize=3, act=act)
        else:
            self.stem =  nn.Sequential(
                BaseConv(3, 12 , 3, 2, act=act),
                BaseConv(12, base_channels, 3, 1, act=act))

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act, split_max_pool_kernel=split_max_pool_kernel),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
    
class YOLOX(BaseEncoder):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        architecture='yolox_s', 
        pretrained=True, 
        finetune=False, 
        # out_dimList = [128, 256, 512, 1024], 
        # use_5_feat=False,
        replace_silu=False, 
        use_customsilu=False,
        in_features=["dark2", "dark3", "dark4", "dark5"],
        in_channels=[256, 512, 1024],
        depthwise=False,
        # conv_focus=False#,
        #split_max_pool_kernel=False,
    ):
        super(YOLOX, self).__init__(finetune)
        
        if architecture.startswith('ti') is True:
            split_max_pool_kernel = True
            conv_focus = True
        else:
            split_max_pool_kernel = False
            conv_focus = False
            
        if architecture.find('yoloxs') != -1:
            if architecture.find('yoloxsd') != -1:
                depth = 0.67
                width = 0.50
            elif architecture.find('yoloxsw1') != -1:
                depth = 0.33
                width = 0.625
            elif architecture.find('yoloxsw2') != -1:
                depth = 0.33
                width = 0.75
            else:
                depth = 0.33
                width = 0.50
        elif architecture.find('yoloxm') != -1:
            if architecture.find('yoloxmd') != -1:
                depth = 1.0
                width = 0.75
            elif architecture.find('yoloxmw') != -1:
                depth = 0.67
                width = 1.0
            else:
                depth = 0.67
                width = 0.75
        elif architecture.find('yoloxl') != -1:
            depth = 1.0
            width = 1.0
        elif architecture.find('yoloxx') != -1:#[-1] == 'x':
            depth = 1.33
            width = 1.25
        elif architecture.find('yoloxt') != -1:#[-1] == 't':
            depth = 0.33
            width = 0.375
        elif architecture.find('yoloxn') != -1:#[-1] == 'n':
            depth = 0.33
            width = 0.25
            
        if replace_silu:
            if use_customsilu:
                act = "customsilu"
            else:
                act = "relu"
        else:
            act = "silu"
            
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act, conv_focus=conv_focus, split_max_pool_kernel=split_max_pool_kernel)
        
        self.in_features = in_features
        self.dimList = [int(in_channels[0] // 2 * width), int(in_channels[0] * width), int(in_channels[1] * width), int(in_channels[2] * width)]        
        # if use_5_feat:
        #     self.in_features.insert(0, "stem")
        #     self.dimList.insert(0, int(in_channels[0] // 4 * width))
            
        # self.make_conv_convert_list(out_dimList)
            
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            self.dimList[-1], self.dimList[-2], 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            self.dimList[-2],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            self.dimList[-2], self.dimList[-3], 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            self.dimList[-3],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            self.dimList[-3], self.dimList[-3], 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            self.dimList[-2],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            self.dimList[-2], self.dimList[-2], 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            self.dimList[-1],
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        
        if pretrained:
            # if architecture.endswith('lite'):# is True:
            if architecture.startswith('ti') is True:
                if architecture.find('yoloxs') != -1:#http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-yolox/yolox-s-ti-lite_39p1_57p9_checkpoint.pth
                    ckpt_file = "yolox-s-ti-lite_39p1_57p9_checkpoint.pth"
                elif architecture.find('yoloxm') != -1:#http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-yolox/yolox_m_ti_lite_45p5_64p2_checkpoint.pth
                    ckpt_file = "yolox_m_ti_lite_45p5_64p2_checkpoint.pth"
                elif architecture.find('yoloxn') != -1:#http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-yolox/yolox_nano_ti_lite_26p1_41p8_checkpoint.pth
                    ckpt_file = "yolox_nano_ti_lite_26p1_41p8_checkpoint.pth"
                elif architecture.find('yoloxt') != -1:#http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-yolox/yolox_tiny_ti_lite_32p0_49p5_checkpoint.pth
                    ckpt_file = "yolox_tiny_ti_lite_32p0_49p5_checkpoint.pth"
                if not os.path.exists(ckpt_file):
                    os.system(f"wget http://software-dl.ti.com/jacinto7/esd/modelzoo/latest/models/vision/detection/coco/edgeai-yolox/{ckpt_file}")
            else:
                if architecture.find('yoloxs') != -1:#[-1] == 's':
                    ckpt_file = "yolox_s.pth"
                elif architecture.find('yoloxm') != -1:#[-1] == 'm':
                    ckpt_file = "yolox_m.pth"
                elif architecture.find('yoloxl') != -1:#[-1] == 'l':
                    ckpt_file = "yolox_l.pth"
                elif architecture.find('yoloxx') != -1:#[-1] == 'x':
                    ckpt_file = "yolox_x.pth"
                elif architecture.find('yoloxn') != -1:#[-1] == 'n':
                    ckpt_file = "yolox_nano.pth"
                elif architecture[-1].find('yoloxt') != -1:# == 't':
                    ckpt_file = "yolox_tiny.pth"
                if not os.path.exists(ckpt_file):
                    os.system(f"wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/{ckpt_file}")
            ckpt = torch.load(ckpt_file, map_location='cpu')
            src_state_dict = ckpt["model"]
            dst_state_dict = self.state_dict()
            ckpt = {}
            for dst_k in dst_state_dict.keys():
                src_k = f"backbone.{dst_k}"
                if src_k in src_state_dict.keys() and src_state_dict[src_k].shape == dst_state_dict[dst_k].shape:
                    print(src_k)
                    ckpt[dst_k] = src_state_dict[src_k]
            self.load_state_dict(state_dict=ckpt, strict=False)
        self._freeze_stages()

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        # [x4, x3, x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(features[-1])  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, features[-2]], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, features[-3]], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32
        
        if len(self.in_features) == 5:
            return [features[0], features[1], pan_out2, pan_out1, pan_out0]
        return [features[0], pan_out2, pan_out1, pan_out0]

