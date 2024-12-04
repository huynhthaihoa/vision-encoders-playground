import torch
from torch import nn
from .common import RepBlock, RepVGGBlock, ConvBNReLU, BiFusion, \
                                ConvBNHS, CSPBlock, DPBlock

class RepBiFPANNeck(nn.Module):
    """RepBiFPANNeck Module
    """
    # [64, 128, 256, 512, 1024]
    # [256, 128, 128, 256, 256, 512]

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4], # 1024
            out_channels=channels_list[5], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]], # 512, 256
            out_channels=channels_list[5], # 256
        )
        self.Rep_p4 = RepBlock(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[5], # 256
            n=num_repeats[5],
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[5], # 256
            out_channels=channels_list[6], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]], # 256, 128
            out_channels=channels_list[6], # 128
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[6], # 128
            n=num_repeats[6],
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[6], # 128
            out_channels=channels_list[7], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7], # 128 + 128
            out_channels=channels_list[8], # 256
            n=num_repeats[7],
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[8], # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[8],
            block=block
        )


    def forward(self, input):

        x4, x3, x2, x1, x0 = input #64, 128, 256, 512, 1024

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0) #256

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1) #128

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1) #256

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2) #512

        outputs = [x4, x3, pan_out2, pan_out1, pan_out0] #64, 256, 128, 256, 512

        return outputs

class RepBiFPANNeck6(nn.Module):
    """RepBiFPANNeck_P6 Module
    """
    # [64, 128, 256, 512, 768, 1024]
    # [512, 256, 128, 256, 512, 1024]

    def __init__(
        self,
        channels_list=None,
        num_repeats=None,
        block=RepVGGBlock
    ):
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.reduce_layer0 = ConvBNReLU(
            in_channels=channels_list[4], # 1024 5
            out_channels=channels_list[6], # 512
            kernel_size=1,
            stride=1
        )

        self.Bifusion0 = BiFusion(
            in_channels=[channels_list[3], channels_list[2]], # 768, 512 4-6
            out_channels=channels_list[6], # 512
        )

        self.Rep_p5 = RepBlock(
            in_channels=channels_list[6], # 512
            out_channels=channels_list[6], # 512
            n=num_repeats[6],
            block=block
        )

        self.reduce_layer1 = ConvBNReLU(
            in_channels=channels_list[6],  # 512 6
            out_channels=channels_list[7], # 256
            kernel_size=1,
            stride=1
        )

        self.Bifusion1 = BiFusion(
            in_channels=[channels_list[2], channels_list[1]], # 512, 256 3-7
            out_channels=channels_list[7], # 256
        )

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[7], # 256
            out_channels=channels_list[7], # 256
            n=num_repeats[7],
            block=block
        )

        self.reduce_layer2 = ConvBNReLU(
            in_channels=channels_list[7],  # 256 7
            out_channels=channels_list[8], # 128
            kernel_size=1,
            stride=1
        )

        self.Bifusion2 = BiFusion(
            in_channels=[channels_list[1], channels_list[0]], # 256, 128 2-8
            out_channels=channels_list[8], # 128
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[8], # 128
            out_channels=channels_list[8], # 128
            n=num_repeats[8],
            block=block
        )

        self.downsample2 = ConvBNReLU(
            in_channels=channels_list[8],  # 128
            out_channels=channels_list[8], # 128
            kernel_size=3,
            stride=2
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[8] + channels_list[8], # 128 + 128
            out_channels=channels_list[9], # 256
            n=num_repeats[9],
            block=block
        )

        self.downsample1 = ConvBNReLU(
            in_channels=channels_list[9],  # 256
            out_channels=channels_list[9], # 256
            kernel_size=3,
            stride=2
        )

        self.Rep_n5 = RepBlock(
            in_channels=channels_list[7] + channels_list[9], # 256 + 256
            out_channels=channels_list[10], # 512
            n=num_repeats[10],
            block=block
        )

        self.downsample0 = ConvBNReLU(
            in_channels=channels_list[10],  # 512
            out_channels=channels_list[10], # 512
            kernel_size=3,
            stride=2
        )

        self.Rep_n6 = RepBlock(
            in_channels=channels_list[6] + channels_list[10], # 512 + 512
            out_channels=channels_list[11], # 1024
            n=num_repeats[11],
            block=block
        )



    def forward(self, input):  

        fpn_out0 = self.reduce_layer0(input[-1])
        f_concat_layer0 = self.Bifusion0([fpn_out0, input[-2], input[-3]])
        f_out0 = self.Rep_p5(f_concat_layer0)
        
        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, input[-3], input[-4]])
        f_out1 = self.Rep_p4(f_concat_layer1)
        
        fpn_out2 = self.reduce_layer2(f_out1)
        f_concat_layer2 = self.Bifusion2([fpn_out2, input[-4], input[-5]])
        pan_out3 = self.Rep_p3(f_concat_layer2) # P3
        
        down_feat2 = self.downsample2(pan_out3)
        p_concat_layer2 = torch.cat([down_feat2, fpn_out2], 1)
        pan_out2 = self.Rep_n4(p_concat_layer2) # P4
        
        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n5(p_concat_layer1) # P5
        
        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n6(p_concat_layer0) # P6
        
        outputs = [input[0], pan_out3, pan_out2, pan_out1, pan_out0]
        
        return outputs

class Lite_EffiNeck(nn.Module):

    def __init__(
        self,
        in_channels,
        unified_channels,
    ):
        super().__init__()
        self.reduce_layer0 = ConvBNHS(
            in_channels=in_channels[0],
            out_channels=unified_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.reduce_layer1 = ConvBNHS(
            in_channels=in_channels[1],
            out_channels=unified_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.reduce_layer2 = ConvBNHS(
            in_channels=in_channels[2],
            out_channels=unified_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.upsample0 = nn.Upsample(scale_factor=2, mode='nearest')

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.Csp_p4 = CSPBlock(
            in_channels=unified_channels*2,
            out_channels=unified_channels,
            kernel_size=5
        )
        self.Csp_p3 = CSPBlock(
            in_channels=unified_channels*2,
            out_channels=unified_channels,
            kernel_size=5
        )
        self.Csp_n3 = CSPBlock(
            in_channels=unified_channels*2,
            out_channels=unified_channels,
            kernel_size=5
        )
        self.Csp_n4 = CSPBlock(
            in_channels=unified_channels*2,
            out_channels=unified_channels,
            kernel_size=5
        )
        self.downsample2 = DPBlock(
            in_channel=unified_channels,
            out_channel=unified_channels,
            kernel_size=5,
            stride=2
        )
        self.downsample1 = DPBlock(
            in_channel=unified_channels,
            out_channel=unified_channels,
            kernel_size=5,
            stride=2
        )
        # self.p6_conv_1 = DPBlock(
        #     in_channel=unified_channels,
        #     out_channel=unified_channels,
        #     kernel_size=5,
        #     stride=2
        # )
        # self.p6_conv_2 = DPBlock(
        #     in_channel=unified_channels,
        #     out_channel=unified_channels,
        #     kernel_size=5,
        #     stride=2
        # )

    def forward(self, input):

        # (_, _, x2, x1, x0) = input

        fpn_out0 = self.reduce_layer0(input[4]) #c5 x0
        x1 = self.reduce_layer1(input[3])       #c4
        x2 = self.reduce_layer2(input[2])       #c3

        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1) #x1
        f_out1 = self.Csp_p4(f_concat_layer0)

        upsample_feat1 = self.upsample1(f_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1) #x2
        pan_out3 = self.Csp_p3(f_concat_layer1) #p3

        down_feat1 = self.downsample2(pan_out3)
        p_concat_layer1 = torch.cat([down_feat1, f_out1], 1)
        pan_out2 = self.Csp_n3(p_concat_layer1)  #p4

        down_feat0 = self.downsample1(pan_out2)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out1 = self.Csp_n4(p_concat_layer2)  #p5

        # top_features = self.p6_conv_1(fpn_out0)
        # pan_out0 = top_features + self.p6_conv_2(pan_out1)  #p6

        outputs = [input[0], input[1], pan_out3, pan_out2, pan_out1]#, pan_out0]
        
        return outputs
