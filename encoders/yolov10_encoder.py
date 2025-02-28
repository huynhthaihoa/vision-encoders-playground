import os
import shutil
import torch


from .ultralytics.nn.tasks import FeatureExtraction
from .base_encoder import BaseEncoder
from function_utils import replace_layers
from class_utils import CustomSiLU

class YOLOv10(BaseEncoder):
    def __init__(self, architecture="yolov10n", 
                pretrained=False, 
                finetune=False, 
                # out_dimList = [],
                replace_silu=False, 
                use_customsilu=False,                
                 ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super(YOLOv10, self).__init__(finetune)
       
        cfg = f"encoders/ultralytics/cfg/models/v10/{architecture}.yaml"
        
        self.model = FeatureExtraction(cfg)

        if replace_silu:
            if use_customsilu:
                replace_layers(self.model, torch.nn.SiLU, CustomSiLU())
            else:
                replace_layers(self.model, torch.nn.SiLU, torch.nn.ReLU())
        
        if architecture.find('yolov10n') != -1:
            self.dimList = [64, 128, 256]
        elif architecture.find('yolov10s') != -1:
            self.dimList = [128, 256, 512]
        elif architecture.find('yolov10m') != -1:
            self.dimList = [192, 384, 576]
        elif architecture.find('yolov10b') != -1:
            self.dimList = [256, 512, 512]
        elif architecture.find('yolov10l') != -1:
            self.dimList = [256, 512, 512]
        elif architecture.find('yolov10x') != -1:
            self.dimList = [320, 640, 640]
            
        # self.make_conv_convert_list(out_dimList) 
        
        if pretrained:
            
            if architecture.startswith('ti') is True:
                architecture = architecture[2:]

            if os.path.exists('ultralytics'):
                shutil.rmtree('ultralytics')
            shutil.copytree('networks/encoders/ultralytics', 'ultralytics')
            ckpt_path = f"https://github.com/THU-MIG/yolov10/releases/download/v1.1/{architecture}.pt"
            ckpt_name = os.path.basename(ckpt_path)
            if not os.path.exists(ckpt_name):
                os.system(f"wget {ckpt_path}")
            dst = self.model.state_dict()
            src = torch.load(ckpt_name, 'cpu')['model'].state_dict()            
            ckpt = {}
            for k, v in src.items():
                if k in dst and v.shape == dst[k].shape:
                    print(k)
                    ckpt[k] = v
            self.model.load_state_dict(state_dict=ckpt)#, strict=False)  
            shutil.rmtree('ultralytics')          
        
        self._freeze_stages()
        
    def forward(self, x):
        feats = self.model.forward(x)
        # if self.conv_convert_list is not None:
        #     converted_feats = list()
        #     for i, feature in enumerate(feats):
        #         converted_feat = self.conv_convert_list[i](feature)
        #         converted_feats.append(converted_feat)
        #     return converted_feats 
        return feats       
