import torch.nn as nn
import torch.nn.init as init

class ProxyEncoder(nn.Module):
    def __init__(self, encoder_name, pretrained=True, finetune=True, init_type='normal'):
        super().__init__()
        
        self.pretrained = pretrained
        self.init_type = init_type
        
        # standardize encoder_name: lowercase all characters
        encoder_name = encoder_name.lower()
        
        # timm encoders
        if encoder_name.startswith('timm_'): #timm encoders
            from encoders.timm_encoder import timmEncoder
            self.wrapper = timmEncoder(encoder_name, pretrained=self.pretrained, finetune=finetune)
        elif encoder_name.startswith('dinov2'):
            from encoders.dinov2_encoder import Dinov2
            self.wrapper = Dinov2(encoder_name, finetune=finetune, init_type=self.init_type)
        elif encoder_name.startswith('ultralytics_'):
            from encoders.ultralytics_encoder import UltralyticsEncoder
            self.wrapper = UltralyticsEncoder(encoder_name, pretrained=self.pretrained, finetune=finetune)
        elif encoder_name.find('convnextv2') != -1:
            from encoders.convnextv2_encoder import ConvNeXtV2
            self.wrapper = ConvNeXtV2(encoder_name, pretrained=self.pretrained, finetune=finetune)       
        elif encoder_name.find('convnext') != -1:
            from encoders.convnext_encoder import ConvNeXt
            self.wrapper = ConvNeXt(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_gelu=replace_silu)       
        elif encoder_name.find('resnet') != -1:
            from encoders.torchvision_encoder import ResNet
            self.wrapper = ResNet(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)  
        elif encoder_name.find('regnet') != -1:
            from encoders.torchvision_encoder import RegNet
            self.wrapper = RegNet(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)        
        elif encoder_name.find('resnext') != -1:
            from encoders.torchvision_encoder import ResNext
            self.wrapper = ResNext(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name == 'mobilenetv2':
            from encoders.torchvision_encoder import MobileNetV2
            self.wrapper = MobileNetV2(pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name.find('mobilenetv3') != -1:
            from encoders.torchvision_encoder import MobileNetV3
            self.wrapper = MobileNetV3(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name.find('mobileone') != -1:
            from encoders.mobileone_encoder import MobileOne
            self.wrapper = MobileOne(encoder_name, pretrained=self.pretrained, finetune=finetune)
        elif encoder_name.find('torchefficientnet') != -1:
            from encoders.torchvision_encoder import TorchEfficientNet
            self.wrapper = TorchEfficientNet(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu) 
        elif encoder_name.find('efficientnetap') != -1:
            from encoders.geffnet_encoder import EfficientNetAdvProp
            self.wrapper = EfficientNetAdvProp(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu) 
        elif encoder_name.find('efficientnetlite') != -1:
            from encoders.geffnet_encoder import EfficientNetLite
            self.wrapper = EfficientNetLite(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)#, replace_silu=replace_silu, use_customsilu=use_customsilu) 
        elif encoder_name.find('efficientnet') != -1:
            from encoders.geffnet_encoder import EfficientNet
            self.wrapper = EfficientNet(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu) 
        elif encoder_name.find('yolox') != -1:
            from encoders.yolox_encoder import YOLOX
            self.wrapper = YOLOX(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_name.find('yolov5') != -1:
            from encoders.nets.yolov5_encoder import YOLOv5                               
            self.wrapper = YOLOv5(encoder_name, pretrained=self.pretrained, finetune=finetune)#keepnum_maxpool=keepnum_maxpool, use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_name.find('yolov6') != -1:
            if encoder_name.find('yolov6lite') != -1:
                from encoders.yolov6lite_encoder import YOLOv6Lite
                self.wrapper = YOLOv6Lite(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
            else:
                from encoders.yolov6_encoder import YOLOv6
                self.wrapper = YOLOv6(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_name.find('yolov7') != -1:
            from encoders.yolov7_encoder import YOLOv7
            self.wrapper = YOLOv7(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_name.find('yolov8') != -1:
            from encoders.nets.yolov8_encoder import YOLOv8                    
            self.wrapper = YOLOv8(encoder_name, pretrained=self.pretrained, finetune=finetune)#keepnum_maxpool=keepnum_maxpool, use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_name.find('yolov9') != -1 or encoder_name.find('gelan') != -1:
            from encoders.yolov9_encoder import YOLOv9
            self.wrapper = YOLOv9(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_name.find('yolov10') != -1:
            from encoders.yolov10_encoder import YOLOv10                    
            self.wrapper = YOLOv10(encoder_name, pretrained=self.pretrained, finetune=finetune)#use_customsilu=use_customsilu, replace_silu=replace_silu)#, keepnum_maxpool=keepnum_maxpool)
        elif encoder_name.find('yolov11') != -1:
            from encoders.yolov11_encoder import YOLOv11
            self.wrapper = YOLOv11(encoder_name, pretrained=self.pretrained, finetune=finetune)#keepnum_maxpool=keepnum_maxpool, use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_name.find('yolov12') != -1:
            from encoders.yolov12_encoder import YOLOv12
            self.wrapper = YOLOv12(encoder_name, pretrained=self.pretrained, finetune=finetune)#keepnum_maxpool=keepnum_maxpool, use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_name.find('ppyoloe') != -1:#is True:
            from encoders.ppyoloe_encoder import PPYOLOE
            self.wrapper = PPYOLOE(encoder_name, pretrained=pretrained, finetune=finetune)#use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_name.find('damoyolo') != -1:
            from encoders.damoyolo_encoder import DAMOYOLO
            self.wrapper = DAMOYOLO(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name.find('rtdetr') != -1:
            from encoders.rtdetr_encoder import RTDetTR
            self.wrapper = RTDetTR(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name.find('hgnetv2') != -1:
            from encoders.hgnetv2_encoder import HGNetv2
            self.wrapper = HGNetv2(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name.find('dfine') != -1:
            from encoders.dfine_encoder import DFINE
            self.wrapper = DFINE(encoder_name, pretrained=self.pretrained, finetune=finetune)
        elif encoder_name.find('deim') != -1:
            from encoders.deim_encoder import DEIM
            self.wrapper = DEIM(encoder_name, pretrained=self.pretrained, finetune=finetune)
        else:
            raise Exception("Unknown encoder!")
        
        if not self.pretrained:
            self.init_params()
            
    def forward(self, input):
        return self.wrapper(input)
    
    def init_params(self):
        for m in self.wrapper.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                if self.init_type == 'normal':
                    init.normal_(m.weight, std=0.01)
                elif self.init_type == 'lecun':
                    init.normal_(m.weight, mean=0, std=1)
                elif self.init_type == 'uniform':
                    init.uniform_(m.weight, -0.02, 0.02)
                elif self.init_type == 'orthogonal':
                    init.orthogonal_(m.weight)
                elif self.init_type == 'constant':
                    init.constant_(m.weight, 0.1)
                elif self.init_type == 'kaiming':
                    init.kaiming_normal_(m.weight, mode='fan_out')
                elif self.init_type == 'he':
                    init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                elif self.init_type == 'kaiming_uniform':
                    init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                elif self.init_type == 'xavier':
                    init.xavier_normal_(m.weight)
                elif self.init_type == 'glorot_uniform':
                    init.xavier_uniform_(m.weight, gain=1)
                elif self.init_type == 'sparse':
                    init.sparse_(m.weight, sparsity=0.1, std=0.01)
                elif self.init_type == 'identity':
                    init.eye_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  # NN.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.bias, 0)
                init.constant_(m.weight, 1.0)
    @property 
    def dimList(self):
        return self.wrapper.dimList 