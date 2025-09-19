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
            embedding_dim: int Dimension of the embedding vectors, i.e., d_model
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

    def _rectify_token_positions_shape(self, token_positions: torch.Tensor, x_ndim: int) -> torch.Tensor:
        # 该函数的核心功能是补全token_positions不足的维度。
        if token_positions.ndim == x_ndim - 1:
            return token_positions
        shapes = [1] * (x_ndim - 1 - token_positions.ndim) + [-1] * (token_positions.ndim)
        token_positions = token_positions.view(*shapes)
        return token_positions

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor)-> torch.Tensor:
        '''
            x.shape = (batch_size, seq_len, d_k)
            token_positions.shape = (..., seq_len)
        '''
        # 重要：按照最后一维现将x分成even和odd两部分！
        x_even, x_odd = x[..., 0::2], x[..., 1::2] # shape=(batch_size, ..., seq_len, d_k//2)
        '''
        关于A[B]这个索引方法的比喻说明：可以想象成A是货物仓库，A[0]是货物种类数量，货物编号从0~A.size(0)-1，A的其他维度都是该货物的属性，比如货物的长宽高。
        B是一个B.shape形状的货架，比如B.shape=(2, 3)，可以当成2行3列的货架。B中的每个元素可以想象成货架每个格子有个标签，用来提示该格子应该放A中的哪个货物。
        因此B中标签的取值范围应该为[0, A.size(0)-1]，不然就无法正确获取A中的货物。
        A[B]就是一个取货摆放的过程。完成取货摆放后，货架形状还是(2, 3)与B保持一致，同时增加了货物自己的长宽高，即A.shape[1:]的参数。
        最后完成的结果C=A[B]，其中C.shape = B.shape + A.shape[1:]
        '''
        # 这里加入token_positions的shape矫正函数
        assert torch.max(token_positions) <= self.cos_cache.shape[0], 'IndexError, token_positions value exceeds max_seq_len'
        token_positions = self._rectify_token_positions_shape(token_positions, x.ndim)
        cos = self.cos_cache[token_positions] # shape=(..., d_k//2)
        sin = self.sin_cache[token_positions] # shape=(..., d_k//2)
        # a_ = cos * a - sin * b
        y_even = cos * x_even - sin * x_odd # shape=(batch_size, ..., seq_len, d_k//2)
        # b_ = sin * a + cos * b
        y_odd = sin * x_even + cos * x_odd # shape=(batch_size, ..., seq_len, d_k//2)

        embed = torch.empty_like(x)
        embed[..., 0::2] = y_even
        embed[..., 1::2] = y_odd

        return embed


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True)[0] # 减去最大值，确保数值稳定
    x = torch.exp(x)
    x = x / x.sum(dim=dim, keepdim=True)
    return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1.0 / math.sqrt(d_k)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        '''
            Q: query tensor, shape=(batch_size, ..., seq_len, d_k)
            K: key tensor, shape=(batch_size, ..., seq_len, d_k)    
            V: value tensor, shape=(batch_size, ..., seq_len, d_v)
            mask: mask tensor, shape=(seq_len, seq_len)
            Returns: context tensor, shape=(batch_size, ..., seq_len, d_v)
        '''
        # 注意标准的Attention公式中，是Q @ K.T
        QKt = einsum(Q, K, "... q_seq d, ... k_seq d -> ... q_seq k_seq") # shape=(batch_size, ..., seq_len, seq_len)
        scores = QKt * self.scale # shape=(batch_size, ..., seq_len, seq_len)
        if mask is not None:
            scores.masked_fill_(~mask, float('-inf'))
        attn = softmax(scores, dim=-1) # shape=(batch_size, ..., seq_len, seq_len)
        output = einsum(attn, V, "... q_seq k_seq, ... k_seq d -> ... q_seq d") # shape = (batch_size, ..., seq_len, d_v)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False, max_seq_len: int = None, theta: float = 10000.0, device=None, dtype=None):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads!")
        self.num_heads = num_heads
        d = d_model // num_heads
        self.use_rope = use_rope
        params = {'device': device, 'dtype': dtype}

        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, d, max_seq_len, device)
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = [Linear(d_model, d_model, **params)
                                                              for _ in range(4)]
        # 下面的mask用得是下三角矩阵！！！行0只能看到列0；行1只能看到列0和1...
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
        self.attn = ScaledDotProductAttention(d)

    def forward(self, x: torch.Tensor, token_positions):
        _, seq_len, _ = x.shape
        # 修改内容：token_positions不能为None，防止后面出现逻辑性错误难以排查
        # Q, K, V shape = (batch_size, seq_len, num_heads, d) -> (batch_size, num_heads, seq_len, d)
        Q, K, V = [rearrange(proj(x), "b s (h d) -> b h s d", h=self.num_heads) for proj in \
                   [self.q_proj, self.k_proj, self.v_proj]]
        if self.use_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = self.mask[:seq_len, :seq_len]
        attn_output = self.attn(Q, K, V, mask) # shape = (batch_size, num_heads, seq_len, d)
        output = self.o_proj(rearrange(attn_output, "b h s d -> b s (h d)")) # shape = (batch_size, ..., d_model)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_rope: bool = False, max_seq_len: int = None, theta: float = 10000.0, device=None, dtype=None):
        super(TransformerBlock, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads!")
        params = {'device': device, 'dtype': dtype}
        self.rmsnorm1 = RMSNorm(d_model, **params)
        self.mha = MultiHeadAttention(d_model, num_heads, use_rope=True, max_seq_len=max_seq_len, theta=theta, **params)
        self.rmsnorm2 = RMSNorm(d_model, **params)
        self.ffn = SwiGLU(d_model, d_ff, **params)

    def forward(self, in_features: torch.Tensor, token_positions: torch.Tensor = None):
        # part1: x + MultiHeadSelfAttention(RMSNorm(x))
        # token_positions这个参数非常重要！
        x = self.rmsnorm1(in_features)
        x = self.mha(x, token_positions)
        x = x + in_features
        residual = x

        # part2: x + SwiGLU(RMSNorm(x))
        x = self.rmsnorm2(x)
        x = self.ffn(x)
        output = x + residual

        return output


class TransformerLM(nn.Module):
    def __init__(self,
             vocab_size: int,
             max_seq_len: int,
             d_model: int,
             num_layers: int,
             num_heads: int,
             d_ff: int,
             rope_theta: float,
             device,
             dtype) -> None:
        super(TransformerLM, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads!")
        params = {'device': device, 'dtype': dtype}
        self.embedding = Embedding(vocab_size, d_model, **params)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, use_rope=True, max_seq_len=max_seq_len, theta=rope_theta, **params) \
                for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, **params)
        # lm_head虽然叫output embedding，然而其实是个Linear层
        self.lm_head = Linear(d_model, vocab_size, **params)

    def forward(self, in_features: torch.Tensor):
        _, seq_len = in_features.shape
        device = in_features.device
        token_positions = torch.arange(seq_len, device=device)
        x = self.embedding(in_features)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        output = self.lm_head(x)

        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    x = torch.randn(2, 4, 16, device=device)
    model = MultiHeadAttention(d_model=16, num_heads=4, use_rope=True, max_seq_len=4, theta=10000.0, device=device)
    token_positions = torch.arange(4).unsqueeze(0).to(device)
    output = model(x, token_positions=token_positions)
    print(output.shape)
    print('Done!')