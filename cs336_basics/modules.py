import math

import torch
import torch.nn as nn
from einops import rearrange, einsum


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
        '''
        torch.nn.init.trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None)
            tensor (Tensor) – an n-dimensional torch.Tensor
            mean (float) – the mean of the normal distribution
            std (float) – the standard deviation of the normal distribution
            a (float) – the minimum cutoff value
            b (float) – the maximum cutoff value
            generator (Optional[Generator]) – the torch Generator to sample from (default: None)
        '''
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用einops代替@运算符
        return einsum(x, self.W, "... in_feat, out_feat in_feat -> ... out_feat")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        '''
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
            Embedding不涉及矩阵计算，类似查表；num_embeddings为行数，查找某行，返回embedding_dim维度的向量
        '''
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.W = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

    def _init_weight(self) -> None:
        nn.init.trunc_normal_(self.W, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
            RMSNorm数学公式参考"03. Architectures, hyperparameters"的笔记
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        '''
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
     
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        ret = x * rms * self.gamma

        return ret.to(in_type)


if __name__ == '__main__':
    module = Linear(3, 4)
    x = torch.randn(2, 3)
    y = module(x)
    print(y.shape)