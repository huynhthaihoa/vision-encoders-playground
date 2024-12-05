import timm

from .base_encoder import BaseEncoder

class timmEncoder(BaseEncoder):
    """wrapper for every feature extractors from timm: https://huggingface.co/docs/timm/feature_extraction
    """    
    def __init__(self, architecture, pretrained=True, finetune=False):
        super(timmEncoder, self).__init__(finetune)
        
        if architecture.startswith('timm_'):
            architecture = architecture[architecture.find('_') + 1:]
            
        self.encoder = timm.create_model(architecture, features_only=True, pretrained=pretrained)
                    
        self._freeze_stages()
    
    def forward(self, x):
        features = self.encoder(x)
        if self.dimList is None:
            self.dimList = list()
            for feature in features:
                self.dimList.append(feature.shape[1])
        return features