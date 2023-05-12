import torch
import torch.nn as nn
import numpy as np
class DynamicLinear(nn.Module):
    def __init__(self, out_features=None, bias=True, same_dim=False):
        super().__init__()
        if not same_dim and out_features is None:
            raise ValueError("out_features must be specified if same_dim=False")
        self.out_features = out_features
        self.same_dim = same_dim
        self.bias = bias
        self.weight = None
        self.bias_term = None

    def forward(self, x):
        device = x.device
        if self.weight is None:
            in_features = x.size(-1)
            if self.same_dim:
                self.out_features = in_features
            self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features)).to(device)
            nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
            
            if self.bias:
                self.bias_term = nn.Parameter(torch.Tensor(self.out_features)).to(device)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / np.sqrt(fan_in)
                nn.init.uniform_(self.bias_term, -bound, bound)

        return torch.nn.functional.linear(x, self.weight, self.bias_term)
