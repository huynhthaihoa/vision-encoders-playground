# # DINOv2
# dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
# dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
# dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# # DINOv2 with registers
# dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
# dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
# dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
# dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')

import torch
import torch.nn as nn

from .base_encoder import BaseEncoder

class Dinov2(BaseEncoder):
    def __init__(
        self,
        architecture='dinov2_vits14', 
        # pretrained=True, 
        finetune=False, 
        out_channels=[256, 512, 1024, 1024],
        init_type="normal",
        # use_clstoken=False,
    ):  
        super(Dinov2, self).__init__(finetune)
        
        if architecture.endswith('_clstoken'):# is True:
            # logger.info("use class token!")
            # exit(0)
            index = architecture.find('_clstoken')
            architecture = architecture[:index]
            self.use_clstoken = True
        else:
            self.use_clstoken = False
        
        self.dimList = out_channels
        
        self.init_type = init_type
        
        self._backbone = torch.hub.load('facebookresearch/dinov2', architecture)  
        
        if architecture == 'dinov2_vits14' or architecture == 'dinov2_vits14_reg':
            self.in_channel = 384
            self.intermediate_layer_idx = [2, 5, 8, 11]
        elif architecture == 'dinov2_vitb14' or architecture == 'dinov2_vitb14_reg':
            self.in_channel = 768
            self.intermediate_layer_idx = [2, 5, 8, 11]
        elif architecture == 'dinov2_vitl14' or architecture == 'dinov2_vitl14_reg':
            self.in_channel = 1024
            self.intermediate_layer_idx = [4, 11, 17, 23]
        elif architecture == 'dinov2_vitg14' or architecture == 'dinov2_vitg14_reg':
            self.in_channel = 1536
            self.intermediate_layer_idx = [9, 19, 29, 39]
        else:
            raise Exception(f"Cannot find DINOv2 variant {architecture}!")
           
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.in_channel,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in self.dimList
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])

        if self.use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * self.in_channel, self.in_channel),
                        nn.GELU()))
        
        # self.intermediate_layer_idx = {
        #     'vits': [2, 5, 8, 11],
        #     'vitb': [2, 5, 8, 11], 
        #     'vitl': [4, 11, 17, 23], 
        #     'vitg': [9, 19, 29, 39]
        # }
        
        self._freeze_stages() 
        
    def forward(self, x): 
        
        out = []
           
        _, _, h, w = x.shape
        out_h, out_w = self.get_out_size((h, w))
        
        print(out_h, out_w)
        
        features = self._backbone.get_intermediate_layers(x, self.intermediate_layer_idx, return_class_token=self.use_clstoken)
        for i, raw in enumerate(features):
            if self.use_clstoken:
                feat, cls_token = raw[0], raw[1]
                readout = cls_token.unsqueeze(1).expand_as(feat)
                feat = self.readout_projects[i](torch.cat((feat, readout), -1))
            else:
                #(B, H * W, C)
                feat = raw
            
            # convert into: (B, C, H * W) -> (B, C, H, W)   
            feat = feat.permute(0, 2, 1)
            feat = feat.reshape((feat.shape[0], feat.shape[1], out_h, out_w))
                        
            feat = self.projects[i](feat)
            feat = self.resize_layers[i](feat)
            
            out.append(feat)
        
        return out
    
    def get_out_size(self, in_size):
        h, w = in_size
        return (h // self._backbone.patch_size, w // self._backbone.patch_size)
    
    def _freeze_stages(self):
        for module in self._backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval() # always freeze BN
            else:
                for param in module.parameters():
                    param.requires_grad = self.finetune
                    
        for param in self.projects.parameters():
            param.requires_grad = True

        for param in self.resize_layers.parameters():
            param.requires_grad = True

        if self.use_clstoken:
            for param in self.readout_projects.parameters():
                param.requires_grad = True
