import os
import torch
import torch.nn as nn

from typing import List

from .modules.mobileone_utils import MobileOneBlock
from .base_encoder import BaseEncoder

class MobileOne(BaseEncoder):
    """ MobileOne Model

        Pytorch implementation of `An Improved One millisecond Mobile Backbone` -
        https://arxiv.org/pdf/2206.04040.pdf
    """
    def __init__(self,
                 architecture,
                 num_blocks_per_stage: List[int] = [2, 8, 10, 1],
                 inference_mode: bool = False,
                 pretrained = False,
                 finetune = False,
                 out_dimList = [],
                 use_5_feat = False) -> None:
                #  num_blocks_per_stage: List[int] = [2, 8, 10, 1],
                #  num_classes: int = 1000,
                #  width_multipliers: Optional[List[float]] = None,
                #  inference_mode: bool = False,
                #  use_se: bool = False,
                #  num_conv_branches: int = 1) -> None:
        """ Construct MobileOne model.

        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super(MobileOne, self).__init__(finetune)
        
        # default hyperparams
        use_se = False
        num_conv_branches = 1
        
        if architecture[-1] == '0': #s0
            width_multipliers = [0.75, 1.0, 1.0, 2.0]
            num_conv_branches = 4
        else:
            if architecture[-1] == '1': #s1
                width_multipliers = [1.5, 1.5, 2.0, 2.5]
            elif architecture[-1] == '2': #s2
                width_multipliers = [1.5, 2.0, 2.5, 4.0]
            elif architecture[-1] == '3': #s3
                width_multipliers = [2.0, 2.5, 3.0, 4.0]
            elif architecture[-1] == '4': #s4
                width_multipliers = [3.0, 3.5, 3.5, 4.0]
                use_se = True

        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        self.dimList = list()
        # Build stages
        
        #stage0: / 2
        self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes,
                                     kernel_size=3, stride=2, padding=1,
                                     inference_mode=self.inference_mode)
        if use_5_feat:
            self.dimList.append(self.in_planes) # /2
            
        self.cur_layer_idx = 1
        
        #stage1: / 4
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0],
                                       num_se_blocks=0)
        self.dimList.append(int(64 * width_multipliers[0]))
        
        #stage2: / 8
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1],
                                       num_se_blocks=0)
        self.dimList.append(int(128 * width_multipliers[1]))
        
        #stage3: / 16
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2],
                                       num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.dimList.append(int(256 * width_multipliers[2]))

        #stage4: / 32
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3],
                                       num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.dimList.append(int(512 * width_multipliers[3]))
        
        self.make_conv_convert_list(out_dimList)

        if pretrained:
            ckpt = f"{architecture}_unfused.pth.tar"
            if not os.path.exists(ckpt):
                os.system(f"wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/{ckpt}")
            checkpoint = torch.load(ckpt)
            self.load_state_dict(checkpoint, strict=False)
        
        self._freeze_stages()

    def _make_stage(self,
                    planes: int,
                    num_blocks: int,
                    num_se_blocks: int) -> nn.Sequential:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1]*(num_blocks-1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # Depthwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=self.in_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        out_featList = list()
        
        x_2 = self.stage0(x) # /2
        if len(self.dimList) == 5:
            if self.conv_convert_list is not None:
                converted_feat = self.conv_convert_list[0](x_2)
                out_featList.append(converted_feat)
            else:
                out_featList.append(x_2)
            
        x_4 = self.stage1(x_2) # /4
        if self.conv_convert_list is not None:
            converted_feat = self.conv_convert_list[-4](x_4)
            out_featList.append(converted_feat)
        else:
            out_featList.append(x_4)
        
        x_8 = self.stage2(x_4) # /8
        if self.conv_convert_list is not None:
            converted_feat = self.conv_convert_list[-3](x_8)
            out_featList.append(converted_feat)
        else:
            out_featList.append(x_8)
        
        x_16 = self.stage3(x_8) # /16
        if self.conv_convert_list is not None:
            converted_feat = self.conv_convert_list[-2](x_16)
            out_featList.append(converted_feat)
        else:
            out_featList.append(x_16)
        
        x_32 = self.stage4(x_16) # /32
        if self.conv_convert_list is not None:
            converted_feat = self.conv_convert_list[-1](x_32)
            out_featList.append(converted_feat)
        else:
            out_featList.append(x_32)
                    
        return out_featList
    
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
