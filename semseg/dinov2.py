import torch
from mmengine.model import BaseModule
from torch import nn

from mmseg.registry import MODELS



@MODELS.register_module()
class DINOv2(nn.Module):
    """Use DINOv2 pre-trained models
    """

    def __init__(self, version='large', freeze=False, load_from=None):
        super().__init__()
        
        if version == 'large':
            self.dinov2 = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_vitl14', source='local', pretrained=False)
        else:
            raise NotImplementedError

        if load_from is not None:
            d = torch.load(load_from, map_location='cpu')
            new_d = {}
            for key, value in d.items():
                if 'pretrained' in key:
                    new_d[key.replace('pretrained.', '')] = value
            self.dinov2.load_state_dict(new_d)
        
        self.freeze = freeze
        
    def forward(self, inputs):
        B, _, h, w = inputs.shape
        
        if self.freeze:
            with torch.no_grad():
                features = self.dinov2.get_intermediate_layers(inputs, 4)
        else:
            features = self.dinov2.get_intermediate_layers(inputs, 4)
        
        outs = []
        for feature in features:
            C = feature.shape[-1]
            feature = feature.permute(0, 2, 1).reshape(B, C, h // 14, w // 14).contiguous()
            outs.append(feature)
        
        return outs
