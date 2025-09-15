import math

import torch
import torch.nn as nn
from einops import rearrange, einsum

'''
torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None)
    tensor (Tensor) – an n-dimensional torch.Tensor
    mean (float) – the mean of the normal distribution
    std (float) – the standard deviation of the normal distribution
    a (float) – the minimum cutoff value
    b (float) – the maximum cutoff value
    generator (Optional[Generator]) – the torch Generator to sample from (default: None)
'''

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        '''
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
            Linear weights: N(µ = 0, σ2 = 2 / (din+dout) ) truncated at [−3σ, 3σ]
        '''
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self._init_weight()

    def _init_weight(self) -> None:
        std = math.sqrt(2 / (self.in_features + self.out_features))
        a = -3 * std
        b = 3 * std
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用einops代替@运算符
        return einsum(x, self.W, "... in_feat, out_feat in_feat -> ... out_feat")


if __name__ == '__main__':
    module = Linear(3, 4)
    x = torch.randn(2, 3)
    y = module(x)
    print(y.shape)