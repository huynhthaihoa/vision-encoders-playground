"""
https://github.com/abhyaambati/lora-from-scratch/blob/main/LoRA-From-Scratch/lora/injector.py
Inject LoRA into a model.
"""

from .lora_layer import LoRALinear
import torch.nn as nn

def inject_lora_into_model(model, r=8, target_module_name=None, alpha=1, dropout=0., replace=False, finetune_bias=False, verbose=False):#="attention"):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (target_module_name is None or (target_module_name is not None and target_module_name in name)):
            if verbose:
                print(f"Injecting LoRA into {name}")
            in_f, out_f = module.in_features, module.out_features
            lora = LoRALinear(in_f, out_f, r=r, alpha=alpha, dropout=dropout, replace=replace, finetune_bias=finetune_bias)
            if replace is False:
                lora.weight.data = module.weight.data.clone()
            lora.bias.data = module.bias.data.clone() if module.bias is not None else None
            parent = _get_parent_module(model, name)
            setattr(parent, name.split('.')[-1], lora)
    return model

def _get_parent_module(model, module_name):
    parts = module_name.split('.')
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    return obj