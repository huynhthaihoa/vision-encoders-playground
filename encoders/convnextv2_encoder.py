import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

from .base_encoder import BaseEncoder

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., approximate='none'):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU(approximate)
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(BaseEncoder):

    """ ConvNeXt V2: https://github.com/facebookresearch/ConvNeXt-V2/tree/main
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, architecture='atto', pretrained=True, finetune=False, out_dimList = [64, 128, 256, 512, 1024], replace_gelu=False,
                 in_chans=3, 
                 drop_path_rate=0.
                 ):
        super(ConvNeXtV2, self).__init__(finetune)
        if replace_gelu:
            approximate = 'tanh'
        else:
            approximate = 'none'
        model_url = None
        if architecture.find('atto') != -1:
            depths = [2, 2, 6, 2]
            self.dimList = [40, 80, 160, 320]
            # if architecture.find('_1k') != -1:
            model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt"
        elif architecture.find('femto') != -1:
            depths = [2, 2, 6, 2]
            self.dimList = [48, 96, 192, 384]
            # if architecture.find('_1k') != -1:
            model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_femto_1k_224_ema.pt"
        elif architecture.find('pico') != -1:
            depths = [2, 2, 6, 2]
            self.dimList = [64, 128, 256, 512]
            # if architecture.find('_1k') != -1:
            model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt"
        elif architecture.find('nano') != -1:
            depths = [2, 2, 6, 2]
            self.dimList = [80, 160, 320, 640]
            if architecture.find('_1k') != -1:
                model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_pico_1k_224_ema.pt"
            elif architecture.find('_22k') != -1:
                if architecture.find('224') != -1:
                    model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_224_ema.pt"
                elif architecture.find('384') != -1:
                    model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_nano_22k_384_ema.pt"
        elif architecture.find('tiny') != -1:
            depths = [3, 3, 27, 3]
            self.dimList = [96, 192, 384, 768]
            if architecture.find('_1k') != -1:
                model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt"
            elif architecture.find('_22k') != -1:
                if architecture.find('224') != -1:
                    model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt"
                elif architecture.find('384') != -1:
                    model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_384_ema.pt"
        elif architecture.find('base') != -1:
            depths = [3, 3, 27, 3]
            self.dimList = [128, 256, 512, 1024]
            if architecture.find('_1k') != -1:
                model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt"
            elif architecture.find('_22k') != -1:
                if architecture.find('224') != -1:
                    model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_224_ema.pt"
                elif architecture.find('384') != -1:
                    model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt"
        elif architecture.find('large') != -1:
            depths = [3, 3, 27, 3]
            self.dimList = [192, 384, 768, 1536]
            if architecture.find('_1k') != -1:
                model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_large_1k_224_ema.pt"
            elif architecture.find('_22k') != -1:
                if architecture.find('224') != -1:
                    model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt"
                elif architecture.find('384') != -1:
                    model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
        elif architecture.find('huge') != -1:
            depths = [3, 3, 27, 3]
            self.dimList = [352, 704, 1408, 2816]  
            if architecture.find('_1k') != -1:
                model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_huge_1k_224_ema.pt"
            elif architecture.find('_22k') != -1:
                if architecture.find('384') != -1:
                    model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_384_ema.pt"
                elif architecture.find('512') != -1:
                    model_url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt"
            
        self.make_conv_convert_list(out_dimList)
 
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, self.dimList[0], kernel_size=4, stride=4),
            LayerNorm(self.dimList[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(self.dimList[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(self.dimList[i], self.dimList[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtV2Block(dim=self.dimList[i], drop_path=dp_rates[cur + j], approximate=approximate) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        if pretrained and model_url is not None:
            checkpoint = torch.hub.load_state_dict_from_url(url=model_url, map_location="cpu", check_hash=True)
            src = checkpoint['model']
            # for k in checkpoint_model.keys():
            #     print(k)
            # print("============")
            dst = self.state_dict()
            ckpt = {}
            for k, v in src.items():
                if k in dst and v.shape == dst[k].shape:
                    print(k)
                    ckpt[k] = v
            self.load_state_dict(state_dict=ckpt, strict=False)
            
            # for k in ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']:
            #     del checkpoint_model[k]  
            # load_state_dict(self, checkpoint_model)

        # self.apply(self._init_weights)
        
        self._freeze_stages()
    
    # def getDims(self):
    #     return self.dimList
    
    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.

    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """

    #     def _init_weights(m):
    #         if isinstance(m, nn.Linear):
    #             trunc_normal_(m.weight, std=.02)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     if isinstance(pretrained, str):
    #         self.apply(_init_weights)
    #         # logger = get_root_logger()
    #         load_checkpoint(self, pretrained, strict=False)
    #     elif pretrained is None:
    #         self.apply(_init_weights)
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        out_featList = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if self.conv_convert_list is None:
                out_featList.append(x)
            else:
                converted_feat = self.conv_convert_list[i](x)
                out_featList.append(converted_feat)
        return out_featList