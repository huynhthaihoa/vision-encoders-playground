import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base_encoder import BaseEncoder
from class_utils import get_activation

class ConvBNLayer(nn.Module):

    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(ch_out, )
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='customhardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = get_activation(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)  #(N, C, 1, 1)    
        x_se = self.fc(x_se)  #(N, C, 1, 1)
        return x * self.act(x_se) #(N, C, H, W) * (N, C, 1, 1)

class CustomGlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()    

    def forward(self, x):#(N, C, H, W)
        x = x.mean(dim=-1, keepdim=True) #(N, C, H, 1)
        x = x.transpose(2, 3) #(N, C, 1, H)
        x = x.mean(dim=-1, keepdim=True) #(N, C, 1, 1) how to make it (N, C, H, W)?
        return x
    
class CustomGlobalAvgPool2dBC(nn.Module):
    def __init__(self):
        super().__init__()    

    def forward(self, x):
        empties_W = torch.empty((x.shape[0], x.shape[1], x.shape[3], 1))
        empties_H = torch.empty((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
        x = x.mean(dim=-1, keepdim=True)#(N, C, H, 1)
        x = x.transpose(2, 3)#(N, C, 1, H)
        x = x.mean(dim=-1, keepdim=True) #(N, C, 1, 1)
        x = x + empties_W #(N, C, 1, 1) + (N, C, W, 1) -> (N, C, W, 1)
        x = x.transpose(2, 3)  #(N, C, 1, W)
        return x + empties_H #(N, C, 1, W) + (N, C, H, W) -> (N, C, H, W)
    
class NadimGlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()    

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1) #(N, C, HxW)
        x = x.mean(dim=-1, keepdim=True) #now it becomes (N, C, 1)
        return x.view(x.shape[0], x.shape[1], 1, 1) # (N, C, 1, 1)
    
class CustomEffectiveSELayer(nn.Module):
    def __init__(self, channels, act='customhardsigmoid', use_avgpool=False, use_act=True):
        super(CustomEffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        if use_act:
            self.act = get_activation(act) if act is None or isinstance(act, (str, dict)) else act
        else:
            self.act = None 
            
        if use_avgpool:
            self.avg_pool = CustomGlobalAvgPool2dBC()
        else:
            self.avg_pool = None
    
    def forward(self, x):
        if self.avg_pool is not None:
            x_se = self.avg_pool(x)
            x_se = self.fc(x_se) #(N, C, H, W)
        else:
            x_se = self.fc(x) #(N, C, 1, 1)  
            
        if self.act is not None: #won't change the shape
            x_se = self.act(x_se) 
        
        # if self.avg_pool is not None: #broadcast
        #     x_trans = x.contiguous().transpose(1,3)  #(N, W, H, C)
        #     x_se_trans = x_se.contiguous().transpose(1,3) #(N, 1, 1, C)
        #     y = x_trans * x_se_trans #(N, W, H, C)
        #     return  y.contiguous().transpose(1,3) # (N, C, H, W)
                  
        return x * x_se      

class CustomEffectiveSELayer_N(nn.Module):
    def __init__(self, channels, act='customhardsigmoid', use_avgpool=False, use_act=True):
        super(CustomEffectiveSELayer_N, self).__init__()
        self.fc = nn.Linear(channels, channels)
        self.sigmoid = nn.Sigmoid()    
    
    def forward(self, x):
        b,c,h, w = x.shape
        reshape_x = x.view(b*c, -1) # b*c, h*w
        mean_x = torch.mean(reshape_x, -1, keepdim=True) # b*c, 1 
        squeezed_x = mean_x.squeeze(-1) # b*c
        fc_x = self.fc(squeezed_x.view(b,c)) # b, c
        fc_x = self.sigmoid(fc_x) # b,c
        fc_x = fc_x.unsqueeze(-1) # b,c,1
        out_x = reshape_x * fc_x # b,c h*w
        out_x = out_x.view(b,c,h,w) # b, c, h, w
        return out_x
        
class SELayer(nn.Module):
    """Squeeze-Excitation block to replace EffectiveSELayer
    Based on: https://github.com/pytorch/vision/blob/bf01bab6125c5f1152e4f336b470399e52a8559d/torchvision/ops/misc.py#L224
    """
    def __init__(self, channels):#, act='hardsigmoid'):
        super(SELayer, self).__init__()
        self.fc1 = torch.nn.Conv2d(channels, channels, 1)
        self.fc2 = torch.nn.Conv2d(channels, channels, 1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.activation = get_activation('relu')
        self.scale_activation = get_activation('sigmoid')
        
    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input

class CSPResStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 attn=True):
        super(CSPResStage, self).__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2, ch_mid // 2, act=act, shortcut=True)
            for _ in range(n)
        ])

        if attn:
            # self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
            self.attn = CustomEffectiveSELayer(ch_mid, act='hardsigmoid')#'hardsigmoid')
            # self.attn = None
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], axis=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', deploy=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.deploy = deploy
        if self.deploy == False:
            self.conv1 = ConvBNLayer(
                ch_in, ch_out, 3, stride=1, padding=1, act=None)
            self.conv2 = ConvBNLayer(
                ch_in, ch_out, 1, stride=1, padding=0, act=None)
        else:
            self.conv = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1
            )
        self.act = get_activation(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        if self.deploy:
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        y = self.act(y)
        return y

    def switch_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1
            )
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__(self.conv1)
        self.__delattr__(self.conv2)
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', shortcut=True):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y

class SPP(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 act='swish',
                 ):
        super(SPP, self).__init__()
        self.pool = []
        # max_pool_module_list = []
        for i, size in enumerate(pool_size):
            assert (size - 3) % 2 == 0; "Required Kernel size cannot be implemented with kernel_size of 3"
            num_3x3_maxpool = 1 + (size - 3) // 2
            pool = nn.Sequential(*num_3x3_maxpool*[nn.MaxPool2d(kernel_size=3, stride=1, padding=1,
                ceil_mode=False)])
            # pool = nn.MaxPool2d(
            #     kernel_size=size,
            #     stride=1,
            #     padding=size // 2,
            #     ceil_mode=False)
            self.add_module('pool{}'.format(i),
                            pool
                            )
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

    def forward(self, x):
        outs = [x]

        for pool in self.pool:
            outs.append(pool(x))
        y = torch.cat(outs, axis=1)

        y = self.conv(y)
        return y

class CSPStage(nn.Module):
    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', spp=False):
        super(CSPStage, self).__init__()

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()

        next_ch_in = ch_mid
        for i in range(n):
            if block_fn == 'BasicBlock':
                self.convs.add_module(str(i), BasicBlock(next_ch_in, ch_mid, act=act, shortcut=False))
            else:
                raise NotImplementedError
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp',
                    SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act)
                )
            next_ch_in = ch_mid
        # self.convs = nn.Sequential(*convs)
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], axis=1)
        y = self.conv3(y)
        return y

def drop_block_2d(
        x, drop_prob: float = 0.1, block_size: int = 7,  gamma_scale: float = 1.0,
        with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. This layer has been tested on a few training
    runs with success, but needs further validation and possibly optimization for lower runtime impact.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    # seed_drop_rate, the gamma parameter
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
        (W - block_size + 1) * (H - block_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = torch.meshgrid(torch.arange(W).to(x.device), torch.arange(H).to(x.device))
    valid_block = ((w_i >= clipped_block_size // 2) & (w_i < W - (clipped_block_size - 1) // 2)) & \
                  ((h_i >= clipped_block_size // 2) & (h_i < H - (clipped_block_size - 1) // 2))
    valid_block = torch.reshape(valid_block, (1, 1, H, W)).to(dtype=x.dtype)

    if batchwise:
        # one mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device)
    else:
        uniform_noise = torch.rand_like(x)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1).to(dtype=x.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size=clipped_block_size,  # block_size,
        stride=1,
        padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            x = x * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x

def drop_block_fast_2d(
        x: torch.Tensor, drop_prob: float = 0.1, block_size: int = 7,
        gamma_scale: float = 1.0, with_noise: bool = False, inplace: bool = False, batchwise: bool = False):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf

    DropBlock with an experimental gaussian noise option. Simplied from above without concern for valid
    block mask at edges.
    """
    B, C, H, W = x.shape
    total_size = W * H
    clipped_block_size = min(block_size, min(W, H))
    gamma = gamma_scale * drop_prob * total_size / clipped_block_size ** 2 / (
            (W - block_size + 1) * (H - block_size + 1))

    if batchwise:
        # one mask for whole batch, quite a bit faster
        block_mask = torch.rand((1, C, H, W), dtype=x.dtype, device=x.device) < gamma
    else:
        # mask per batch element
        block_mask = torch.rand_like(x) < gamma
    block_mask = F.max_pool2d(
        block_mask.to(x.dtype), kernel_size=clipped_block_size, stride=1, padding=clipped_block_size // 2)

    if with_noise:
        normal_noise = torch.randn((1, C, H, W), dtype=x.dtype, device=x.device) if batchwise else torch.randn_like(x)
        if inplace:
            x.mul_(1. - block_mask).add_(normal_noise * block_mask)
        else:
            x = x * (1. - block_mask) + normal_noise * block_mask
    else:
        block_mask = 1 - block_mask
        normalize_scale = (block_mask.numel() / block_mask.to(dtype=torch.float32).sum().add(1e-7)).to(dtype=x.dtype)
        if inplace:
            x.mul_(block_mask * normalize_scale)
        else:
            x = x * block_mask * normalize_scale
    return x

class DropBlock2d(nn.Module):
    """ DropBlock. See https://arxiv.org/pdf/1810.12890.pdf
    """
    def __init__(self,
                 drop_prob=0.1,
                 block_size=7,
                 gamma_scale=1.0,
                 with_noise=False,
                 inplace=False,
                 batchwise=False,
                 fast=True):
        super(DropBlock2d, self).__init__()
        self.drop_prob = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size = block_size
        self.with_noise = with_noise
        self.inplace = inplace
        self.batchwise = batchwise
        self.fast = fast  # FIXME finish comparisons of fast vs not

    def forward(self, x):
        if not self.training or not self.drop_prob:
            return x
        if self.fast:
            return drop_block_fast_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)
        else:
            return drop_block_2d(
                x, self.drop_prob, self.block_size, self.gamma_scale, self.with_noise, self.inplace, self.batchwise)

class CSPResNet(nn.Module):

    def __init__(self,
                 layers=[3, 6, 6, 3],
                 channels=[64, 128, 256, 512, 1024],
                 act='swish',
                 return_idx=[0, 1, 2, 3, 4],
                 use_large_stem=False,
                 width_mult=1.0,
                 depth_mult=1.0,
                 attn=True):
        super().__init__()
        channels = [max(round(c * width_mult), 1) for c in channels]
        layers = [max(round(l * depth_mult), 1) for l in layers]
        selected_act = act

        if use_large_stem:
            self.stem = nn.Sequential(
                ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=selected_act),
                ConvBNLayer(channels[0] // 2, channels[0] // 2, 3, stride=1, padding=1, act=selected_act),
                ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=selected_act)
            )
        else:
            self.stem = nn.Sequential(
                ConvBNLayer(3, channels[0] // 2, 3, stride=2, padding=1, act=selected_act),
                ConvBNLayer(channels[0] // 2, channels[0], 3, stride=1, padding=1, act=selected_act)
            )
        n = len(channels) - 1
        self.stages = nn.Sequential(
            *[CSPResStage(BasicBlock, channels[i], channels[i + 1], layers[i], 2, act=selected_act, attn=attn) for i in range(n)]
        )
        self._out_channels = channels[1:]
        self._out_strides = [4, 8, 16, 32]
        self.return_idx = return_idx

    def forward(self, inputs):
        outs = []

        x = inputs
        x = self.stem(x)
        
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)

        return outs

class CustomCSPPAN(nn.Module):
    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=[1024, 512, 256],
                 norm_type='bn',
                 act='swish',
                 stage_fn='CSPStage',
                 block_fn='BasicBlock',
                 stage_num=1,
                 block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False,
                 width_mult=1.0,
                 depth_mult=1.0,
                 ):
        super().__init__()
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        # act = get_activation(act) if act is None or isinstance(act,
                                                               #(str, dict)) else act
        self.num_blocks = len(in_channels)
        # print("out channels:", out_channels)
        self._out_channels = out_channels[::-1]
        in_channels = in_channels[::-1]
        self.fpn_stages = nn.ModuleList()
        self.fpn_routes = nn.ModuleList()


        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = nn.Sequential()
            for j in range(stage_num):
                if stage_fn == 'CSPStage':
                    stage.add_module(
                        str(j),
                        CSPStage(block_fn,
                                 ch_in if j == 0 else ch_out,
                                 ch_out,
                                 block_num,
                                 act=act,
                                 spp=(spp and i == 0))
                    )
                else:
                    raise NotImplementedError

            if drop_block:
                stage.append(DropBlock2d(drop_prob=1 - keep_prob, block_size=block_size))
            self.fpn_stages.append(stage)

            if i < self.num_blocks - 1:
                self.fpn_routes.append(
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out // 2,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act))
            ch_pre = ch_out

        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNLayer(
                    ch_in=out_channels[i + 1],
                    ch_out=out_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act))

            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(
                    str(j),
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   spp=False))
            if drop_block:
                stage.add_module('drop', DropBlock2d(block_size, keep_prob))

            pan_stages.append(stage)

        self.pan_stages = nn.Sequential(*pan_stages[::-1])
        self.pan_routes = nn.Sequential(*pan_routes[::-1])

    def forward(self, blocks):
        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], axis=1)
            # route = block
            # for layer in self.fpn_stages[i]:
            #     route = layer(block)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(
                    route, scale_factor=2.)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.cat([route, block], axis=1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats#[::-1]

class PPYOLOE(BaseEncoder):
    def __init__(
        self,
        architecture='ppyoloe_s', 
        pretrained=True, 
        finetune=False, 
        out_dimList = [], 
        # use_5_feat=False,
        replace_silu=False, 
        use_customsilu=False,
        # in_features=["dark2", "dark3", "dark4", "dark5"],
        # in_channels=[256, 512, 1024],
        # depthwise=False,
        # conv_focus=False#,
        #split_max_pool_kernel=False,
    ):
        super(PPYOLOE, self).__init__(finetune)
        
        if architecture.find('ppyoloe_s') != -1:    
            depth = 0.33
            width = 0.50
        elif architecture.find('ppyoloe_m') != -1:
            depth = 0.67
            width = 0.75
        elif architecture.find('ppyoloe_l') != -1:
            depth = 1.0
            width = 1.0
        elif architecture.find('ppyoloe_x') != -1:
            depth = 1.33
            width = 1.25
            
        if replace_silu:
            if use_customsilu:
                act = 'customsilu'
            else:
                act = 'relu'
        else:
            act = 'swish'
            
        if architecture.find('noattn') is True:
            attn = False
        else:
            attn = True

        self.backbone = CSPResNet(
            return_idx=[0, 1, 2, 3],
            use_large_stem=True,
            act=act,
            width_mult=width,
            depth_mult=depth,
            attn=attn
        )        
        if architecture.endswith('truncate') is True:
            # self.backbone = CSPResNet(
            #     return_idx=[0, 1, 2, 3],
            #     use_large_stem=True,
            #     act=act,
            #     width_mult=width,
            #     depth_mult=depth,
            #     attn=attn
            # )
            self.neck = None
            self.dimList = self.backbone._out_channels
        else:
            # self.backbone = CSPResNet(
            #     return_idx=[0, 1, 2, 3],
            #     use_large_stem=True,
            #     act=act,
            #     width_mult=width,
            #     depth_mult=depth,
            #     attn=attn
            # )
            self.neck = CustomCSPPAN(
                out_channels=[768, 384, 192],
                stage_num=1,
                block_num=3,
                act=act,
                spp=True,
                width_mult=width,
                depth_mult=depth
            )
            self.dimList = self.neck._out_channels
            self.dimList.insert(0, self.backbone._out_channels[0])

        self.make_conv_convert_list(out_dimList)
        
        if pretrained:
            if architecture.find('ppyoloe_s') != -1:
                ckpt_file = "ppyoloe/ppyoloe_crn_s_400e.pth"
            elif architecture.find('ppyoloe_m') != -1:
                ckpt_file = "ppyoloe/ppyoloe_crn_m_300e.pth"
            elif architecture.find('ppyoloe_l') != -1:
                ckpt_file = "ppyoloe/ppyoloe_crn_l_300e.pth"
            elif architecture.find('ppyoloe_x') != -1:
                ckpt_file = "ppyoloe/ppyoloe_crn_x_300e.pth"
            ckpt = torch.load(ckpt_file, map_location='cpu')
            src_state_dict = ckpt["model"]
            dst_state_dict = self.state_dict()
            weights = {}
            for k in dst_state_dict.keys():
                if k in src_state_dict.keys():
                    print(k)
                    weights[k] = src_state_dict[k]
            self.load_state_dict(state_dict=weights)
        self._freeze_stages()
                            
    def forward(self, x):
        features = self.backbone(x)

        if self.neck is not None:
            features[-3:] = self.neck(features[-3:])
        
        if self.conv_convert_list is not None:
            out_featList = list()
            for i, feature in enumerate(features):
                converted_feat = self.conv_convert_list[i](feature)
                out_featList.append(converted_feat)
            return out_featList
        
        return features