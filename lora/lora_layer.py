"""
LoRA Layer Implementation: https://github.com/abhyaambati/lora-from-scratch/blob/main/LoRA-From-Scratch/lora/lora_layer.py
This code implements a LoRA (Low-Rank Adaptation) layer for PyTorch.
LoRA is a technique for adapting pre-trained models to new tasks with low-rank updates.
This implementation is based on the original paper "LoRA: Low-Rank Adaptation of Large Language Models" by Edward Hu et al.
"""
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=1.0, dropout=0.0, replace=False):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = alpha / r

        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        
        self.bias = nn.Parameter(torch.randn(out_features), requires_grad=False)
        self.replace = replace
        
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.randn(out_features, r) * 0.01)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        result = nn.functional.linear(x, self.weight)
        lora_adjustment = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        if self.replace:
            return lora_adjustment + self.bias
        return result + lora_adjustment + self.bias