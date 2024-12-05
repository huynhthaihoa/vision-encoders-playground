"""torchvision-based encoders: https://pytorch.org/vision/stable/models.html"""

import torch.nn as nn
import torchvision.models as models

from function_utils import replace_layers
from class_utils import myConv, CustomSiLU
from .base_encoder import BaseEncoder

class ResNet(BaseEncoder):
    """ResNet: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    """    
    def __init__(self, architecture='resnet50', pretrained=True, finetune=False):#, out_dimList = [64, 128, 256, 512, 1024], use_5_feat=False):
        super(ResNet, self).__init__(finetune)
        if architecture.find('50') != -1:
            if not pretrained:
                self.encoder = models.resnet50()
            else:
                self.encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif architecture.find('18') != -1:#architecture == 'resnet18':
            if not pretrained:
                self.encoder = models.resnet18()
            else:
                self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif architecture.find('34') != -1: #architecture == 'resnet34':
            if not pretrained:
                self.encoder = models.resnet34()
            else:
                self.encoder = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif architecture.find('101') != -1:#architecture == 'resnet101':
            if not pretrained:
                self.encoder = models.resnet101()
            else:            
                self.encoder = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif architecture.find('152') != -1:#architecture == 'resnet152':
            if not pretrained:
                self.encoder = models.resnet152()
            else:    
                self.encoder = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)

        del self.encoder.fc
        del self.encoder.avgpool
        
        self.layerList = ['layer1','layer2','layer3', 'layer4']
        
        if architecture.find('18') != -1 or architecture.find('34') != -1:
            self.dimList = [64, 128, 256, 512]
        else:
                self.dimList = [256, 512, 1024, 2048]

        # if use_5_feat:
        #     self.layerList.insert(0, 'relu')
        #     self.dimList.insert(0, 64)
            
        # self.make_conv_convert_list(out_dimList)
 
        # self.freeze_bn(finetune)
        self._freeze_stages()
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            if any(x in k for x in self.layerList):
                # if self.conv_convert_list is None:
                out_featList.append(feature)
                # else:
                #     converted_feat = self.conv_convert_list[cnt](feature)
                #     out_featList.append(converted_feat)
                cnt = cnt + 1 
        return out_featList
    
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval() # always freeze BN
            else:
                for param in module.parameters():
                    param.requires_grad = enable  
    
    # def train(self, mode=True):
    #     """Convert the model into training mode while keep layers freezed."""
    #     super(ResNet, self).train(mode)      
   
class RegNet(BaseEncoder):
    """RegNet: https://arxiv.org/abs/2003.13678
    """    
    def __init__(self, architecture='regnet_y_3_2gf', pretrained=True, finetune=False):#, out_dimList = [64, 128, 256, 512, 1024], use_5_feat=False):
        super(RegNet, self).__init__(finetune)
        
        if architecture.find('y_400mf') != -1: #ok -> 4,969,849 - 6.88 GMac
            if not pretrained:
                self.encoder = models.regnet_y_400mf()
            else:
                self.encoder = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.DEFAULT)
            self.dimList = [48, 104, 208, 440]
        elif architecture.find('y_800mf') != -1: #ok -> 7,770,201 - 13.03 GMac
            if not pretrained:
                self.encoder = models.regnet_y_800mf()
            else:
                self.encoder = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.DEFAULT)
            self.dimList = [64, 144, 320, 784]
        elif architecture.find('y_1_6gf') != -1: #ok -> 11,829,415 - 14.59 GMac
            if not pretrained:
                self.encoder = models.regnet_y_1_6gf()
            else:
                self.encoder = models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.DEFAULT)
            self.dimList = [48, 120, 336, 888]
        elif architecture.find('y_3_2gf') != -1: #ok -> 21,596,995 - 30.13 GMac
            if not pretrained:
                self.encoder = models.regnet_y_3_2gf()
            else:
                self.encoder = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.DEFAULT)
            self.dimList = [72, 216, 576, 1512]
        elif architecture.find('y_8gf') != -1: #ok -> 52,232,305 - 145.65 GMac
            if not pretrained:
                self.encoder = models.regnet_y_8gf()
            else:
                self.encoder = models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.DEFAULT)
            self.dimList = [224, 448, 896, 2016]
        elif architecture.find('y_16gf') != -1: #ok -> 108,122,293 - 193.26 GMac
            if not pretrained:
                self.encoder = models.regnet_y_16gf()
            else:
                self.encoder = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.DEFAULT)
            self.dimList = [224, 448, 1232, 3024]
        elif architecture.find('y_32gf') != -1: #ok -> 174,042,755 - 303.86 GMac
            if not pretrained:
                self.encoder = models.regnet_y_32gf()
            else:
                self.encoder = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.DEFAULT)
            self.dimList = [232, 696, 1392, 3712]        

        elif architecture.find('x_400mf') != -1:  #ok -> 5,626,401 - 4.55 GMac
            if not pretrained:
                self.encoder = models.regnet_x_400mf()
            else:
                self.encoder = models.regnet_x_400mf(weights=models.RegNet_X_400MF_Weights.DEFAULT)
            self.dimList = [32, 64, 160, 400]
        elif architecture.find('x_800mf') != -1: #ok -> 8,580,321 - 12.71 GMac
            if not pretrained:
                self.encoder = models.regnet_x_800mf()
            else:
                self.encoder = models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.DEFAULT)
            self.dimList = [64, 128, 288, 672]
        elif architecture.find('x_1_6gf') != -1: #ok -> 11,072,105 - 19.92 GMac
            if not pretrained:
                self.encoder = models.regnet_x_1_6gf()
            else:
                self.encoder = models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.DEFAULT)
            self.dimList = [72, 168, 408, 912]
        elif architecture.find('x_3_2gf') != -1: #ok -> 18,771,041 - 36.99 GMac
            if not pretrained:
                self.encoder = models.regnet_x_3_2gf()
            else:
                self.encoder = models.regnet_x_3_2gf(weights=models.RegNet_X_3_2GF_Weights.DEFAULT)
            self.dimList = [96, 192, 432, 1008]
        elif architecture.find('x_8gf') != -1: #ok -> 42,637,809 - 62.3 GMac
            if not pretrained:
                self.encoder = models.regnet_x_8gf()
            else:
                self.encoder = models.regnet_x_8gf(weights=models.RegNet_X_8GF_Weights.DEFAULT)
            self.dimList = [80, 240, 720, 1920]
        elif architecture.find('x_16gf') != -1: #ok -> 80,553,121 - 219.33 GMac
            if not pretrained:
                self.encoder = models.regnet_x_16gf()
            else:
                self.encoder = models.regnet_x_16gf(weights=models.RegNet_X_16GF_Weights.DEFAULT)
            self.dimList = [256, 512, 896, 2048]
        elif architecture.find('x_32gf') != -1: #ok -> 153,569,393 - 404.08 GMac
            if not pretrained:
                self.encoder = models.regnet_x_32gf()
            else:
                self.encoder = models.regnet_x_32gf(weights=models.RegNet_X_32GF_Weights.DEFAULT)
            self.dimList = [336, 672, 1344, 2520] 

        # if architecture.find('y_128gf') != -1:
        #     self.encoder = models.regnet_y_128gf(pretrained=pretrained)
        #     self.dimList = [232, 696, 1392, 3712] 
                    
        del self.encoder.fc
        del self.encoder.avgpool
        
        self.layerList = ['block1','block2','block3', 'block4']
        
        # if use_5_feat:
        #     self.dimList.insert(0, 32)
        #     self.layerList.insert(0, 'stem') 
        
        # self.make_conv_convert_list(out_dimList)
        
        # self.freeze_bn(finetune)
        self._freeze_stages()
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'stem':
                feature = v(feature)
            if k == 'trunk_output':
                for m, n in v._modules.items():
                    if self.layerList[cnt] == m:
                        feature = n(feature)
                        # if self.conv_convert_list is None:
                        out_featList.append(feature)
                        # else:
                        #     converted_feat = self.conv_convert_list[cnt](feature)
                        #     out_featList.append(converted_feat)
                        cnt += 1
                        if cnt == 4:
                            break
        return out_featList
    
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval() # always freeze BN
            else:
                for param in module.parameters():
                    param.requires_grad = enable  
    
    # def train(self, mode=True):
    #     """Convert the model into training mode while keep layers freezed."""
    #     super(RegNet, self).train(mode)   

class ResNext(BaseEncoder):
    """ResNeXt: https://github.com/facebookresearch/ResNeXt
    """    
    def __init__(self, architecture='resnext101_32x8d', pretrained=True, finetune=False):#, out_dimList = [128, 256, 512, 1024], use_5_feat=False):
        super(ResNext, self).__init__(finetune)
        # self.args = args
        # after passing Layer1 : H/4  x W/4     (44 x 88)
        # after passing Layer2 : H/8  x W/8     (22 x 44)
        # after passing Layer3 : H/16 x W/16    (11 x 22)
        # after passing Layer4 : H/32 x W/32    (6 x 11)
        if architecture == 'resnext101_32x8d':
            if pretrained:
                self.encoder = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.DEFAULT)
            else:
                self.encoder =  models.resnext101_32x8d()
        elif architecture == 'resnext101_64x4d':
            if pretrained:
                self.encoder = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.DEFAULT)
            else:
                self.encoder =  models.resnext101_64x4d()     
        elif architecture == 'resnext50_32x4d':
            if pretrained:
                self.encoder = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
            else:
                self.encoder =  models.resnext50_32x4d()  
                                                   
        del self.encoder.fc
        
        self.layerList = ['layer1', 'layer2', 'layer3', 'layer4']
        self.dimList = [256, 512, 1024, 2048]
        
        # if use_5_feat:
        #     self.layerList.insert(0, 'relu')
        #     self.dimList.insert(0, 64)
        
        # self.make_conv_convert_list(out_dimList)
        
        # self.freeze_bn(finetune)
        self._freeze_stages()

    
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        for k, v in self.encoder._modules.items():
            if k == 'avgpool':
                break
            feature = v(feature)
            if any(x in k for x in self.layerList):
                # if self.conv_convert_list is None:
                out_featList.append(feature)
                # else:
                #     converted_feat = self.conv_convert_list[cnt](feature)
                #     out_featList.append(converted_feat)
                cnt = cnt + 1
        return out_featList
    
    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        '''
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable
        '''
        for name, parameters in self.encoder.named_parameters():
            if name == 'conv1.weight':
                parameters.requires_grad = enable
            if self.fixList is not None and any(x in name for x in self.fixList):
                parameters.requires_grad = enable

class MobileNetV2(BaseEncoder):
    """MobileNetV2: https://arxiv.org/abs/1801.04381
    """    
    def __init__(self, pretrained=True, finetune=False):#, out_dimList = [128, 256, 512, 1024], use_5_feat=False):
        super(MobileNetV2, self).__init__(finetune)
        # after passing 1st : H/4  x W/4
        # after passing 2nd : H/8  x W/8
        # after passing 3rd : H/16 x W/16
        # after passing 4th : H/32 x W/32
        if pretrained:
            self.encoder = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        else:
            self.encoder = models.mobilenet_v2()
            
        del self.encoder.classifier

        self.layerList = [3, 6, 13, 18]
        self.dimList = [144, 192, 576, 1280]
        # if use_5_feat:
        #     self.layerList.insert(0, 1)
        #     self.dimList.insert(0, 32)

        # self.make_conv_convert_list(out_dimList)
        
        self._freeze_stages()
                
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        for i in range(len(self.encoder.features)):
            if i in self.layerList:
                if i != 18:
                    for j in range(len(self.encoder.features[i].conv)):
                        feature = self.encoder.features[i].conv[j](feature)
                        if i == 1: 
                            if j == 0:
                                # if self.conv_convert_list is None:
                                out_featList.append(feature)
                                # else:
                                #     converted_feat = self.conv_convert_list[cnt](feature)
                                #     out_featList.append(converted_feat)
                                cnt = cnt + 1
                        else:
                            if j == 1:
                                # if self.conv_convert_list is None:
                                out_featList.append(feature)
                                # else:
                                #     converted_feat = self.conv_convert_list[cnt](feature)
                                #     out_featList.append(converted_feat)
                                cnt = cnt + 1
                else:
                    feature = self.encoder.features[i](feature)
                    # if self.conv_convert_list is None:
                    out_featList.append(feature)
                    # else:
                    #     converted_feat = self.conv_convert_list[cnt](feature)
                    #     out_featList.append(converted_feat)
            else:
                feature = self.encoder.features[i](feature)
            
        return out_featList
                    
class MobileNetV3(BaseEncoder):
    """MobileNetV3: https://arxiv.org/abs/1905.02244
    """    
    def __init__(self, architecture='mobilenetv3large', pretrained=True, finetune=False):#, out_dimList = [128, 256, 512, 1024]):
        super(MobileNetV3, self).__init__(finetune)
        #self.args = args
        # after passing 1st : H/4  x W/4
        # after passing 2nd : H/8  x W/8
        # after passing 3rd : H/16 x W/16
        # after passing 4th : H/32 x W/32
        if architecture.find('large') != -1:
            if pretrained:
                self.encoder = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            else:
                self.encoder = models.mobilenet_v3_large()
                
            self.layerList = [3, 6, 11, 16]
            self.dimList = [72, 120, 480, 960]
        else:
            if pretrained:
                self.encoder = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            else:
                self.encoder = models.mobilenet_v3_small()   
                
            self.layerList = [1, 3, 8, 12]
            self.dimList = [16, 88, 144, 576]         

        del self.encoder.classifier
        del self.encoder.avgpool

        # if len(out_dimList) == 0:
        #     self.conv_convert_list = None
        # if len(out_dimList) != 0:
        #     assert len(out_dimList) == 4

        #     norm = 'BN'
        #     act = 'ReLU'

        #     convert1 = myConv(self.dimList[0], out_dimList[0], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=self.dimList[0]//16)
        #     convert2 = myConv(self.dimList[1], out_dimList[1], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=self.dimList[1]//16)
        #     convert3 = myConv(self.dimList[2], out_dimList[2], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=self.dimList[2]//16)
        #     convert4 = myConv(self.dimList[3], out_dimList[3], kSize=1, stride=1, padding=0, bias=False, norm=norm, act=act, num_groups=self.dimList[3]//16)
        #     self.conv_convert_list = nn.ModuleList([convert1, convert2, convert3, convert4])#,convert5])
        
        self._freeze_stages()
    
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        for i in range(len(self.encoder.features)):            
            if i in self.layerList:
                if i != self.layerList[-1]:
                    for j in range(len(self.encoder.features[i].block)):
                        feature = self.encoder.features[i].block[j](feature)
                        if i == 1: 
                            if j == 0:
                                # if self.conv_convert_list is None:
                                out_featList.append(feature)
                                # else:
                                #     converted_feat = self.conv_convert_list[cnt](feature)
                                #     out_featList.append(converted_feat)
                                cnt = cnt + 1
                        else:
                            if j == 1:
                                # if self.conv_convert_list is None:
                                out_featList.append(feature)
                                # else:
                                #     converted_feat = self.conv_convert_list[cnt](feature)
                                #     out_featList.append(converted_feat)
                                cnt = cnt + 1
                else:
                    feature = self.encoder.features[i](feature)
                    # if self.conv_convert_list is None:
                    out_featList.append(feature)
                    # else:
                    #     converted_feat = self.conv_convert_list[cnt](feature)
                    #     out_featList.append(converted_feat)
            else:
                feature = self.encoder.features[i](feature)
            
        return out_featList
    
class TorchEfficientNet(BaseEncoder):
    """EfficientNet: https://arxiv.org/abs/1905.11946
    """    
    def __init__(self, architecture="torchefficientnet_b0", pretrained=True, finetune=False, replace_silu = False, use_customsilu = False):#, out_dimList = [128, 256, 512, 1024]
        super(TorchEfficientNet, self).__init__(finetune)
        
        if architecture.find("0") != -1:
            if not pretrained:
                self.encoder = models.efficientnet_b0()
            else:
                self.encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.dimList = [24, 40, 112, 1280]
        elif architecture.find("1") != -1:
            if not pretrained:
                 self.encoder = models.efficientnet_b1()
            else:
                self.encoder = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
            self.dimList = [24, 40, 112, 1280]
        elif architecture.find("2") != -1:
            if not pretrained:
                self.encoder = models.efficientnet_b2()
            else:
                self.encoder = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            self.dimList = [24, 48, 120, 1408]
        elif architecture.find("3") != -1:
            if not pretrained:
                self.encoder = models.efficientnet_b3()
            else:
                self.encoder = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            self.dimList = [32, 48, 136, 1536]
        elif architecture.find("4") != -1:
            if not pretrained:
                self.encoder = models.efficientnet_b4()
            else:
                self.encoder = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            self.dimList = [32, 56, 160, 1792]
        elif architecture.find("5") != -1:
            if not pretrained:
                self.encoder = models.efficientnet_b5()
            else:
                self.encoder = models.efficientnet_b5( weights=models.EfficientNet_B5_Weights.DEFAULT)
            self.dimList = [40, 64, 176, 2048]
        elif architecture.find("6") != -1:
            if not pretrained:
                self.encoder = models.efficientnet_b6()
            else:
                self.encoder = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.DEFAULT)
            self.dimList = [40, 72, 200, 2304]
        elif architecture.find("7") != -1:
            if not pretrained:
                self.encoder = models.efficientnet_b7()
            else:
                self.encoder = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
            self.dimList = [48, 80, 224, 2560]
        
        if replace_silu:
            if use_customsilu:
                replace_layers(self.encoder, nn.SiLU, CustomSiLU()) #CustomSiLU()
            else:
                replace_layers(self.encoder, nn.SiLU, nn.ReLU(inplace=True)) #CustomSiLU()

        del self.encoder.avgpool
        del self.encoder.classifier
                
        self.block_idx = [2, 3, 5, 8] #5th feature is extracted after bn2
        
        # if len(out_dimList) == 0:
        #     self.conv_convert_list = None
        # else:
        # if len(out_dimList) != 0:
        #     assert len(out_dimList) == 4
            
        #     norm = 'BN'
        #     act = 'ReLU'
        #     convert1 = myConv(self.dimList[0], out_dimList[0], kSize=1, stride=1, padding=0, bias=False, 
        #                     norm=norm, act=act, num_groups=self.dimList[0]//16)
        #     convert2 = myConv(self.dimList[1], out_dimList[1], kSize=1, stride=1, padding=0, bias=False, 
        #                     norm=norm, act=act, num_groups=self.dimList[1]//16)
        #     convert3 = myConv(self.dimList[2], out_dimList[2], kSize=1, stride=1, padding=0, bias=False, 
        #                     norm=norm, act=act, num_groups=self.dimList[2]//16)
        #     convert4 = myConv(self.dimList[3], out_dimList[3], kSize=1, stride=1, padding=0, bias=False, 
        #                     norm=norm, act=act, num_groups=self.dimList[3]//16)
        #     self.conv_convert_list = nn.ModuleList([convert1, convert2, convert3, convert4])
        
        self._freeze_stages()
        
    def forward(self, x):
        out_featList = []
        feature = x
        cnt = 0
        block_cnt = 0
        
        for k, v in self.encoder._modules.items():
            if k == 'features':
                for m, n in v._modules.items():
                    feature = n(feature)
                    if self.block_idx[block_cnt] == eval(m):
                        # if self.conv_convert_list is None:
                        out_featList.append(feature)
                        # else:
                        #     converted_feat = self.conv_convert_list[block_cnt](feature)
                        #     out_featList.append(converted_feat)
                        block_cnt += 1
                        if block_cnt == 4:
                            break
                       
        return out_featList
    
    # def train(self, mode=True):
    #     """Convert the model into training mode while keep layers freezed."""
    #     super(TorchEfficientNet, self).train(mode)

