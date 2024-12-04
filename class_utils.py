import numpy as np
import torch
import torch.nn as nn
import math

from torch.nn import functional as F

from function_utils import conv_org, upsample

class Interpolate(nn.Module):
    def __init__(self, scale_factor=None, size=None, mode = 'bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        assert (scale_factor is not None or size is not None)
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        if self.size is not None:
            x = self.interp(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        else:
            x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x
    
class CustomAdaptiveAvgPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
       

    def forward(self, x): 
        inp_size = x.size()
        return F.avg_pool2d(input=x,
                  kernel_size= (inp_size[2], inp_size[3]))

class CustomAdaptiveMaxPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
       

    def forward(self, x): 
        inp_size = x.size()
        return F.max_pool2d(input=x,
                  kernel_size= (inp_size[2], inp_size[3]))
        
class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2dCustom, self).__init__()
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x
        
class CustomSiLU(nn.Module):  # export-friendly version of nn.SiLU()
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class CustomELU(nn.Module): # export-friendly version of nn.ELU()
    @staticmethod
    def forward(x):
        if x > 0:
            return x
        return (torch.exp(x) - 1)
    
class CustomApproxGELU(nn.Module): # export-friendly approximate-with-tanh version of nn.GELU()
    @staticmethod
    def forward(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x^3)))

class CustomHardsigmoidDiv(nn.Module):
    @staticmethod
    def forward(x):
        # x = torch.clamp(x, min=0, max=6)
        # x = torch.add(x, 3)
        # x = torch.mul(x, 0.16666666667)
        # return x
        
        # x = torch.clamp(x, min=-3, max=3)
        
        x = (x / 6) + 0.5
        
        return torch.clamp(x, min=0, max=1)
        
        # return x.clamp(0, 6).add(3).mul(0.16666666667)  #div(6)

class CustomHardsigmoidMul(nn.Module):
    @staticmethod
    def forward(x):
        x = (x * 0.16666666667) + 0.5    
        return torch.clamp(x, min=0, max=1)
            
class CustomHardswish(nn.Module):
    @staticmethod
    def forward(x):
        y = x.clamp(0, 6).add(3).mul(0.16666666667)  #div(6)
        return x * y

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(F.sigmoid(x))
            return x
        else:
            return x * F.sigmoid(x)

def get_activation(name='silu', inplace=True):
    if name is None:
        return nn.Identity()
    if name == 'silu':
        module = nn.SiLU(inplace=inplace)
    elif name == 'customsilu':
        module = CustomSiLU()
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'gelu':
        module = nn.GELU()
    elif name == 'swish':
        module = Swish(inplace=inplace)
    elif name == 'sigmoid':
        module = nn.Sigmoid()#inplace=inplace)
    elif name == 'tanh':
        module = nn.Tanh()
    elif name == 'hardsigmoid':
        module = nn.Hardsigmoid(inplace=inplace)
    elif name == 'customhardsigmoiddiv':
        module = CustomHardsigmoidDiv()
    elif name == 'customhardsigmoidmul':
        module = CustomHardsigmoidMul()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module
   
class myConv(nn.Module):
    def __init__(self, in_ch, out_ch, kSize, stride=1, 
                    padding=0, dilation=1, bias=True, norm='GN', act='ELU', num_groups=32):
        super(myConv, self).__init__()
        conv = conv_org
        if act == 'ELU':
            act = nn.ELU()
        else:
            act = nn.ReLU(True)
        module = []
        if norm == 'GN': 
            module.append(nn.GroupNorm(num_groups=num_groups, num_channels=in_ch))
        else:
            module.append(nn.BatchNorm2d(in_ch, momentum=0.01, affine=True, track_running_stats=True))
        module.append(act)
        module.append(conv(in_ch, out_ch, kernel_size=kSize, stride=stride, 
                            padding=padding, dilation=dilation, bias=bias))
        self.module = nn.Sequential(*module)
    def forward(self, x):
        out = self.module(x)
        return out

class FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super(FTB, self).__init__()
        self.in1 = inchannels
        self.mid = midchannels
        self.conv1 = nn.Conv2d(in_channels=self.in1, out_channels=self.mid, kernel_size=3, padding=1, stride=1,
                               bias=True)
        # NN.BatchNorm2d
        self.conv_branch = nn.Sequential(nn.ReLU(inplace=True), \
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3,
                                                   padding=1, stride=1, bias=True), \
                                         nn.BatchNorm2d(num_features=self.mid), \
                                         nn.ReLU(inplace=True), \
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3,
                                                   padding=1, stride=1, bias=True))
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        x = self.relu(x)

        return x

class FFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, interpolate=True, up_first=False):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)

        if interpolate:
            self.upsample = Interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
        self.up_first = up_first

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        x = x + high_x
        if self.up_first:
            x = self.upsample(x)
            x = self.ftb2(x)
        else:
            x = self.ftb2(x)
            x = self.upsample(x)

        return x

class FFMDeConv(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels):
        super(FFMDeConv, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)
        self.upsample = nn.ConvTranspose2d(in_channels=self.outchannels, out_channels=self.outchannels, kernel_size=2, stride=2)

    def forward(self, low_x, high_x):
        x = self.ftb1(low_x)
        x = x + high_x
        x = self.ftb2(x)
        x = self.upsample(x)

        return x

class ATA(nn.Module):
    def __init__(self, inchannels, reduction=8):
        super(ATA, self).__init__()
        self.inchannels = inchannels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.inchannels * 2, self.inchannels // reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.inchannels // reduction, self.inchannels),
                                nn.Sigmoid())

    def forward(self, low_x, high_x):
        n, c, _, _ = low_x.size()
        x = torch.cat([low_x, high_x], 1)
        x = self.avg_pool(x)
        x = x.view(n, -1)
        x = self.fc(x).view(n, c, 1, 1)
        x = low_x * x + high_x

        return x

class AO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, upfactor=2):
        super(AO, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.adapt_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels // 2, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.BatchNorm2d(num_features=self.inchannels // 2), \
            nn.ReLU(inplace=True), \
            nn.Conv2d(in_channels=self.inchannels // 2, out_channels=self.outchannels, kernel_size=3, padding=1,
                      stride=1, bias=True), \
            nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True))

    def forward(self, x):
        x = self.adapt_conv(x)
        return x

class ResidualConv(nn.Module):
    def __init__(self, inchannels):
        super(ResidualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=inchannels, out_channels=inchannels / 2, kernel_size=3, padding=1, stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=inchannels / 2),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=inchannels / 2, out_channels=inchannels, kernel_size=3, padding=1, stride=1,
                      bias=False)
        )

    def forward(self, x):
        x = self.conv(x) + x
        return x

class FeatureFusion(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(FeatureFusion, self).__init__()
        self.conv = ResidualConv(inchannels=inchannels)
        self.up = nn.Sequential(ResidualConv(inchannels=inchannels),
                                nn.ConvTranspose2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3,
                                                   stride=2, padding=1, output_padding=1),
                                nn.BatchNorm2d(num_features=outchannels),
                                nn.ReLU(inplace=True))

    def forward(self, lowfeat, highfeat):
        return self.up(highfeat + self.conv(lowfeat))

class SceneUnderstand(nn.Module):
    def __init__(self, channels):
        super(SceneUnderstand, self).__init__()
        self.channels = channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace=True))
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Sequential(nn.Linear(512 * 8 * 8, self.channels),
                                nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0),
            nn.ReLU(inplace=True))

    def forward(self, x):
        n, c, h, w = x.size()
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(n, -1)
        x = self.fc(x)
        x = x.view(n, self.channels, 1, 1)
        x = self.conv2(x)
        x = x.repeat(1, 1, h, w)
        return x

class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x

class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, interpolate=True, bn=False, expand=False, align_corners=True):#, deconv=False, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        # self.deconv = deconv
        # self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        # self.size=size
        
        self.interpolate = interpolate
        # if self.size is None:
        if interpolate:
            self.upsample = Interpolate(scale_factor=2, mode='bilinear', align_corners=align_corners)
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
        # else:
        # if  self.interpolate:
        #     self.upsample = Interpolate(scale_factor=2, size=self.size, mode='bilinear', align_corners=True)
        # else:
        #     if self.size is None:
        #         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     else:
        #         self.upsample = nn.Upsample(size=self.size, mode='bilinear', align_corners=True)
                
            

    def forward(self, *xs):#, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        # if (size is None) and (self.size is None):
        #     modifier = {"scale_factor": 2}
        # elif size is None:
        #     modifier = {"size": self.size}
        # else:
        #     modifier = {"size": size}

        # output = nn.functional.interpolate(
        #     output, **modifier, mode="bilinear", align_corners=self.align_corners
        # )
        # if size is None:
        output = self.upsample(output)
        # else:
        #     if self.interpolate:
        #         upsampler = Interpolate(scale_factor=2, size=size, mode='bilinear', align_corners=True)
        #     else:
        #         upsampler = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        #     output = upsampler(output)

        output = self.out_conv(output)

        return output

class CustomFTB(nn.Module):
    def __init__(self, inchannels):
        super(CustomFTB, self).__init__()
        self.in_channels = inchannels
        self.mid_channels = self.in_channels // 2
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.mid_channels, kernel_size=3, padding=1, stride=1,
                        bias=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.mid_channels, 1, kernel_size=1, stride=1, padding=0)            
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(x)
        return self.relu(x)  
    
class NeWCRFsDispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(NeWCRFsDispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x
 
class NeWCRFs2DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(NeWCRFs2DispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.conv1(x)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return self.sigmoid(x)
    
class NeWCRFsDeconvDispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(NeWCRFsDeconvDispHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.upsample = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.upsample(x)
        return self.sigmoid(x)
    
class NeWCRFsDeconvUpsampleDispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(NeWCRFsDeconvUpsampleDispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x
    
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n 

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}".format(**self.__dict__)
        )
        
def _make_fusion_block(features, use_bn, interpolate=True):#, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        interpolate=interpolate,
        # deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True)#,
        #size=size
    # )

def _make_scratch(in_shape, out_shape, groups=1):
    return nn.Conv2d(in_shape, out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
