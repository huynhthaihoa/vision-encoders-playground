import torch
import torch.nn as nn

from class_utils import myConv

class BaseEncoder(nn.Module):
    """BaseEncoder: the base encoder that will be derived by the specific custom encoder

    Args:
        nn (_type_): _description_
    """    
    def __init__(self, finetune):
        super(BaseEncoder, self).__init__()
        self.finetune = finetune
        self.conv_convert_list = None
        self.dimList = None
      
    def forward(self, x):
        raise NotImplementedError

    def make_conv_convert_list(self, out_dimList):
        if len(out_dimList) in [4, 5]:
            norm = 'BN'
            act = 'ReLU'
            convertList = []
            for i, dim in enumerate(self.dimList):
                convert = myConv(dim, out_dimList[i], kSize=1, stride=1, padding=0, bias=False, 
                            norm=norm, act=act, num_groups=dim // 16)
                convertList.append(convert)
            self.conv_convert_list = torch.nn.Sequential(*convertList) #nn.ModuleList([convert for convert in convertList])
        
            
    def _freeze_stages(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval() # always freeze BN
            else:
                for param in module.parameters():
                    param.requires_grad = self.finetune
        
        if self.conv_convert_list is not None:
            for param in self.conv_convert_list.parameters():
                param.requires_grad = True
        
    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(BaseEncoder, self).train(mode)
