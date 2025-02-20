
import os
import torch

from .damo_yolo.base_models.backbones import build_backbone
from .damo_yolo.base_models.necks import build_neck

from .base_encoder import BaseEncoder
    
class DAMOYOLO(BaseEncoder):
        def __init__(
        self,
        architecture='damoyolo_tinynasl20_t', 
        pretrained=True, 
        finetune=False
        ):
            super(DAMOYOLO, self).__init__(finetune)
            
            if architecture == 'damoyolo_t':
                from .damo_yolo.configs.damoyolo_tinynasL20_T import Config
                # config_path = 'damo_yolo/configs/damoyolo_tinynasL20_T.py'
                pretrained_url = 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL20_T_436.pth'
            elif architecture == 'damoyolo_nm':
                from .damo_yolo.configs.damoyolo_tinynasL18_Nm import Config
                # config_path = 'damo_yolo/configs/damoyolo_tinynasL18_Nm.py'  
                pretrained_url = 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/ckpt/before_distill/damoyolo_nano_middle.pth'    
            elif architecture == 'damoyolo_ns':
                from .damo_yolo.configs.damoyolo_tinynasL18_Ns import Config
                # config_path = 'damo_yolo/configs/damoyolo_tinynasL18_Ns.py'
                pretrained_url = 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/ckpt/before_distill/damoyolo_nano_small.pth'
            elif architecture == 'damoyolo_nl':
                from .damo_yolo.configs.damoyolo_tinynasL20_Nl import Config
                # config_path = 'damo_yolo/configs/damoyolo_tinynasL20_Nl.py'
                pretrained_url = 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/ckpt/before_distill/damoyolo_nano_large.pth'
            elif architecture == 'damoyolo_s':
                from .damo_yolo.configs.damoyolo_tinynasL25_S import Config
                # config_path = 'damo_yolo/configs/damoyolo_tinynasL25_S.py'
                pretrained_url = 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL25_S_477.pth'
            elif architecture == 'damoyolo_m':
                from .damo_yolo.configs.damoyolo_tinynasL35_M import Config
                # config_path = 'damo_yolo/configs/damoyolo_tinynasL35_M.py'
                pretrained_url = 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL35_M_502.pth'
            elif architecture == 'damoyolo_l':
                from .damo_yolo.configs.damoyolo_tinynasL45_L import Config
                # config_path = 'damo_yolo/configs/damoyolo_tinynasL45_L.py'
                pretrained_url = 'https://idstcv.oss-cn-zhangjiakou.aliyuncs.com/DAMO-YOLO/release_model/clean_model_0317/damoyolo_tinynasL45_L_519.pth'
            config = Config()#parse_config(config_path)
            
            self.backbone = build_backbone(config.model.backbone)
            self.neck = build_neck(config.model.neck)
            
            self.dimList = self.neck.out_channels
            
            # self.make_conv_convert_list(out_dimList)

            if pretrained:
                ckpt = os.path.basename(pretrained_url)
                if not os.path.exists(ckpt):
                    os.system(f"wget {pretrained_url}")
                dst = self.state_dict()
                src = torch.load(ckpt, 'cpu')['model']
                ckpt = {}
                for k, v in src.items():
                    k = k.replace('module.', '')
                    if k in dst and v.shape == dst[k].shape:
                        print(k)
                        ckpt[k] = v
                self.load_state_dict(state_dict=ckpt, strict=True)
            self._freeze_stages()  
                      
        def forward(self, x):
            feature_outs = self.backbone(x)
            fpn_outs = self.neck(feature_outs)
            
            # if self.conv_convert_list is not None:
            #     out_featList = list()
            #     for i, feature in enumerate(fpn_outs):
            #         converted_feat = self.conv_convert_list[i](feature)
            #         out_featList.append(converted_feat)
            #     return out_featList
            return fpn_outs