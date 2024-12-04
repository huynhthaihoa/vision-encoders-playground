import os
import shutil
import torch

from copy import deepcopy

from .ultralytics.nn.tasks import yaml_model_load, parse_model
from .ultralytics.nn.modules.block import C2fAttn, ImagePoolingAttn
from .base_encoder import BaseEncoder
from function_utils import replace_layers
from class_utils import CustomSiLU

class UltralyticsEncoder(BaseEncoder):
    def __init__(self, architecture="ultralytics_yolov8n", 
                pretrained=False, 
                finetune=False, 
                out_dimList = [],
                replace_silu=False, 
                use_customsilu=False,                
                 ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super(UltralyticsEncoder, self).__init__(finetune)
        
        architecture = architecture[architecture.find('_') + 1:] #remove 'ultralytics_'
        cfg = architecture + ".yaml"
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            # LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save, feats = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        if replace_silu:
            if use_customsilu:
                replace_layers(self, torch.nn.SiLU, CustomSiLU())
            else:
                replace_layers(self, torch.nn.SiLU, torch.nn.ReLU())

        self.model = self.model[:-1] #remove head
        self.anchors = 3
        
        if architecture.find('world') != -1:
            self.txt_feats = torch.randn(1, nc or 80, 512)
        else:
            self.txt_feats = None
        
        if architecture.find('yolov8n') != -1:
            self.dimList = [64, 128, 256]
        elif architecture.find('yolov8s') != -1:
            self.dimList = [128, 256, 512]
        elif architecture.find('yolov8m') != -1:
            self.dimList = [192, 384, 576]
        elif architecture.find('yolov8l') != -1:
            self.dimList = [256, 512, 512]
        elif architecture.find('yolov8x') != -1:
            self.dimList = [320, 640, 640]
            
        self.make_conv_convert_list(out_dimList)  
        
        if pretrained:
            if os.path.exists('ultralytics'):
                shutil.rmtree('ultralytics')
            # os.makedirs('ultralytics', exist_ok=True)
            # src_files = os.listdir('networks/encoders/ultralytics/')
            # for file_name in src_files:
            #     src_file_name = os.path.join('networks/encoders/ultralytics/', file_name)
            #     # if os.path.isfile(src_file_name):
            #     dst_file_name = os.path.join('ultralytics', file_name)
            shutil.copytree('networks/encoders/ultralytics', 'ultralytics')
            ckpt_path = f"https://github.com/ultralytics/assets/releases/download/v8.1.0/{architecture}.pt"
            ckpt_name = os.path.basename(ckpt_path)
            if not os.path.exists(ckpt_name):
                os.system(f"wget {ckpt_path}")
            dst = self.state_dict()
            src = torch.load(ckpt_name, 'cpu')['model'].state_dict()            
            ckpt = {}
            for k, v in src.items():
                if k in dst and v.shape == dst[k].shape:
                    print(k)
                    ckpt[k] = v
            self.load_state_dict(state_dict=ckpt)#, strict=False)
            shutil.rmtree('ultralytics')
        self._freeze_stages() 
                       
    def forward(self, x):#, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if self.txt_feats is not None:
            txt_feats = self.txt_feats.to(device=x.device, dtype=x.dtype)
        y = [] # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        feats = []
        for feat in y:
            if feat is not None:
                feats.append(feat)
        feats = feats[-self.anchors:]
        if self.conv_convert_list is not None:
            converted_feats = list()
            for i, feature in enumerate(feats):
                converted_feat = self.conv_convert_list[i](feature)
                converted_feats.append(converted_feat)
            return converted_feats        
        return feats
