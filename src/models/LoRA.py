import torch
import torch.nn as nn
import math
import numpy as np
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim,rank,alpha):
        super().__init__()
        self.A=nn.Parameter(torch.empty(in_dim,rank))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
        self.B=nn.Parameter(torch.zeros(rank,out_dim))
        self.alpha=alpha/rank


    def forward(self,x):
        x=self.alpha*(x @ self.A @ self.B)
        return x

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear=linear
        self.lora=LoRALayer(self.linear.in_features,linear.out_features,rank,alpha)


    def forward(self,x):
        return self.linear(x) + self.lora(x)

def replace_linear_with_lora(model,rank,alpha):
    for name,module in model.named_children():
        if isinstance(module,nn.Linear):
            setattr(model,name,LinearWithLoRA(module,rank,alpha))
        else:
            replace_linear_with_lora(module,rank,alpha)

def FreezeModel(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for param in model.parameters():
        param.requires_grad = False
    return f"Total trainable parameters before: {total_params:,}"



