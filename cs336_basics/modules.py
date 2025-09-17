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
    

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        if d_ff is None:
            d_hidden = 8 / 3 * d_model
            self.d_ff = (d_hidden + 63) // 64 * 64 # 确保d_ff近似d_hidden的同时又是64的倍数
        else:
            self.d_ff = d_ff
        self.linear1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.linear2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.linear3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_for_gate = self.linear1(x)
        silu = x_for_gate * torch.sigmoid(x_for_gate)
        content = self.linear3(x)
        hidden_state = silu * content
        output = self.linear2(hidden_state)

        return output


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k:int, max_seq_len:int, device=None) -> None:
        '''
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        '''
        super(RotaryPositionalEmbedding, self).__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even!")
        # 构建旋转角矩阵，shape=(max_seq_len, d_k//2)
        m = torch.arange(max_seq_len, device=device)
        ks = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        n = 1.0 / torch.pow(theta, ks / d_k)
        mn = torch.outer(m, n) # shape=(max_seq_len, d_k//2)
        cos_cache = torch.cos(mn) # shape=(max_seq_len, d_k//2)
        sin_cache = torch.sin(mn) # shape=(max_seq_len, d_k//2)
        # 注册Buffer，不会被优化器（Optimizer）在反向传播过程中更新
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)-> torch.Tensor:
        # x.shape = (batch_size, seq_len, d_k)
        # 重要：按照最后一维现将x分成even和odd两部分
        x_even, x_odd = x[..., 0::2], x[..., 1::2] # shape=(batch_size, seq_len, d_k//2)
        cos = self.cos_cache[token_positions] # shape=(d_k//2,)
        sin = self.sin_cache[token_positions] # shape=(d_k//2,)
        # a_ = cos * a - sin * b
        y_even = cos * x_even - sin * x_odd # shape=(batch_size, seq_len, d_k//2)
        # b_ = sin * a + cos * b
        y_odd = sin * x_even + cos * x_odd # shape=(batch_size, seq_len, d_k//2)

        embed = torch.empty_like(x)
        embed[..., 0::2] = y_even
        embed[..., 1::2] = y_odd

        return embed


if __name__ == '__main__':
    module = Linear(3, 4)
    x = torch.randn(2, 3)
    y = module(x)
    print(y.shape)