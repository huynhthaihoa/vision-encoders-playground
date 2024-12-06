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
            self.encoder = timmEncoder(encoder_name, pretrained=self.pretrained, finetune=finetune)
        elif encoder_name.startswith('ultralytics_'):
            from encoders.ultralytics_encoder import UltralyticsEncoder
            self.encoder = UltralyticsEncoder(encoder_name, pretrained=self.pretrained, finetune=finetune)
        elif encoder_name.find('convnextv2') != -1:
            from encoders.convnextv2_encoder import ConvNeXtV2
            self.encoder = ConvNeXtV2(encoder_name, pretrained=self.pretrained, finetune=finetune)       
        elif encoder_name.find('convnext') != -1:
            from encoders.convnext_encoder import ConvNeXt
            self.encoder = ConvNeXt(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_gelu=replace_silu)       
        elif encoder_name.find('resnet') != -1:
            from encoders.torchvision_encoder import ResNet
            self.encoder = ResNet(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)  
        elif encoder_name.find('regnet') != -1:
            from encoders.torchvision_encoder import RegNet
            self.encoder = RegNet(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)        
        elif encoder_name.find('resnext') != -1:
            from encoders.torchvision_encoder import ResNext
            self.encoder = ResNext(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name == 'mobilenetv2':
            from encoders.torchvision_encoder import MobileNetV2
            self.encoder = MobileNetV2(pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name.find('mobilenetv3') != -1:
            from encoders.torchvision_encoder import MobileNetV3
            self.encoder = MobileNetV3(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name.find('mobileone') != -1:
            from encoders.mobileone_encoder import MobileOne
            self.encoder = MobileOne(encoder_name, pretrained=self.pretrained, finetune=finetune)
        elif encoder_name.find('torchefficientnet') != -1:
            from encoders.torchvision_encoder import TorchEfficientNet
            self.encoder = TorchEfficientNet(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu) 
        elif encoder_name.find('efficientnetap') != -1:
            from encoders.geffnet_encoder import EfficientNetAdvProp
            self.encoder = EfficientNetAdvProp(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu) 
        elif encoder_name.find('efficientnetlite') != -1:
            from encoders.geffnet_encoder import EfficientNetLite
            self.encoder = EfficientNetLite(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)#, replace_silu=replace_silu, use_customsilu=use_customsilu) 
        elif encoder_name.find('efficientnet') != -1:
            from encoders.geffnet_encoder import EfficientNet
            self.encoder = EfficientNet(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu) 
        elif encoder_name.find('yolox') != -1:
            from encoders.yolox_encoder import YOLOX
            self.encoder = YOLOX(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_name.find('yolov5') != -1:
            from encoders.nets.yolov5_encoder import YOLOv5                               
            self.encoder = YOLOv5(encoder_name, pretrained=self.pretrained, finetune=finetune)#keepnum_maxpool=keepnum_maxpool, use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_name.find('yolov6') != -1:
            if encoder_name.find('yolov6lite') != -1:
                from encoders.yolov6lite_encoder import YOLOv6Lite
                self.encoder = YOLOv6Lite(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
            else:
                from encoders.yolov6_encoder import YOLOv6
                self.encoder = YOLOv6(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_name.startswith('yolov7'):
            from encoders.yolov7_encoder import YOLOv7
            self.encoder = YOLOv7(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_name.find('yolov8') != -1:
            from encoders.nets.yolov8_encoder import YOLOv8                    
            self.encoder = YOLOv8(encoder_name, pretrained=self.pretrained, finetune=finetune)#keepnum_maxpool=keepnum_maxpool, use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_name.startswith('yolov9') or encoder_name.startswith('gelan'):
            from encoders.yolov9_encoder import YOLOv9
            self.encoder = YOLOv9(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list, replace_silu=replace_silu, use_customsilu=use_customsilu)
        elif encoder_name.find('yolov10') != -1:
            from encoders.yolov10_encoder import YOLOv10                    
            self.encoder = YOLOv10(encoder_name, pretrained=self.pretrained, finetune=finetune)#use_customsilu=use_customsilu, replace_silu=replace_silu)#, keepnum_maxpool=keepnum_maxpool)
        elif encoder_name.find('yolov11') != -1:
            from encoders.yolov11_encoder import YOLOv11
            self.encoder = YOLOv11(encoder_name, pretrained=self.pretrained, finetune=finetune)#keepnum_maxpool=keepnum_maxpool, use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_name.startswith('ppyoloe') is True:
            from encoders.ppyoloe_encoder import PPYOLOE
            self.encoder = PPYOLOE(encoder_name, pretrained=pretrained, finetune=finetune)#use_customsilu=use_customsilu, replace_silu=replace_silu)
        elif encoder_name.startswith('rtdetr') is True:
            from encoders.rtdetr_encoder import RTDetTR
            self.encoder = RTDetTR(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name.find('hgnetv2') != -1:
            from encoders.hgnetv2_encoder import HGNetv2
            self.encoder = HGNetv2(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        elif encoder_name.find('damoyolo') != -1:
            from encoders.damoyolo_encoder import DAMOYOLO
            self.encoder = DAMOYOLO(encoder_name, pretrained=self.pretrained, finetune=finetune)#out_dimList=feature_list)
        
        if not self.pretrained:
            self.init_params()
            
    def forward(self, input):
        return self.encoder(input)
    
    def init_params(self):
        for m in self.encoder.modules():
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
        return self.encoder.dimList 