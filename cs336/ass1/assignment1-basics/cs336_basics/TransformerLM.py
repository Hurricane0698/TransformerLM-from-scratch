import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, 
                 in_features:int, 
                 out_features:int, 
                 device:torch.device | None = None,
                 dtype:torch.dtype | None = None) -> None:
        super().__init__()
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = (2 / (in_features + out_features))**0.5
        #构建矩阵,保存的不变量是反映当前学习状态的w，不带偏置
        self.weight = nn.Parameter(nn.init.trunc_normal_(w, mean=0, std=std, a=-3*std, b=3*std))
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return x @ self.weight.transpose(0,1)
    
class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings:int, 
                 embedding_dim:int, 
                 device:torch.device|None = None, 
                 dtype:torch.dtype|None = None) -> None:
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = nn.Parameter(nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3))
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return self.weight[x]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x:torch.Tensor) -> torch.Tensor :
        input_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (torch.sum(x**2, -1, keepdim=True)/ self.d_model + self.eps)**0.5#[b, s, 1]
        norm_x = x / rms *self.gain
        return norm_x.to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int|None = None) -> None:
        super().__init__()
        #设置3个可学习的参数矩阵，利用SiLU(𝑥) = 𝑥 ⋅ 𝜎(𝑥) = 𝑥/(1 + 𝑒−𝑥)激活当作GLU门控元素逐一相乘，fnn内部隐藏层设计为8/3的d_model,取相邻的64倍整数
        if d_ff is None:
            self.d_ff = 8/3*d_model
            if self.d_ff % 64 != 0:
                r = round(self.d_ff/64)#取最近的64整数因子
                self.d_ff = 64*r
            else:
                self.d_ff = int(self.d_ff)
        else:
            self.d_ff = d_ff
        self.w1 = Linear(d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, d_model)
        self.w3 = Linear(d_model, self.d_ff)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        activate = self.w1(x)
        gate = (activate)*torch.sigmoid(activate) #[..., dff]
        gated_x = gate * (self.w3(x))
        return self.w2(gated_x)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_length:int, device:torch.device|None = None) -> None:
        super().__init__()
        #先写cos,sin和theta组成的通用表达式
        seq_index = torch.arange(max_seq_length, device=device).unsqueeze(1)#[s, 1]
        d_k_index = torch.arange(d_k//2, device=device).unsqueeze(0)#[1, d_k/2]
        freq = 1 / theta**(2*d_k_index/d_k)
        #用广播而不是矩阵乘法，广播时从右往左对齐，其中一个维度要么是1要么两个维度相同，维度为1的部分复制乘上其他部分
        angle = seq_index * freq
        self.register_buffer("cos_table", angle.cos())
        self.register_buffer("sin_table", angle.sin())
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        cos = self.cos_table[token_positions]#[b, s, d/2]
        sin = self.sin_table[token_positions]
        #把x[...,d]拆成偶数、奇数数列：[...,d/2]
        x_even = x[..., 0::2]#[...,start::step]
        x_odd = x[..., 1::2]
        new_even = cos*x_even - sin*x_odd
        new_odd = sin*x_even + cos*x_odd
        out = torch.empty_like(x) #创建空矩阵，把算好的放进来
        out[..., 0::2] = new_even
        out[..., 1::2] = new_odd
        return out

def softmax(x:torch.Tensor, dim:int):
    max_x = x.max(dim,keepdim=True).values#max return tuple style values、indices
    norm_x = x - max_x
    exp_x = norm_x.exp()
    return exp_x / exp_x.sum(dim, keepdim=True)

def scaled_dot_product_attention(queries:torch.Tensor, 
                                 keys:torch.Tensor, 
                                 values:torch.Tensor, 
                                 mask:torch.Tensor|None = None)->torch.Tensor:
    d_k = keys.shape[-1]
    attn_w = queries @ keys.transpose(-1,-2) /d_k**0.5 #[b,..., n, d_k],[b,..., d_k, m]->[b,..., n, m]
    if mask is not None:
        attn_w.masked_fill_(~mask, -torch.inf)#True保留，先反转
    attn_s = softmax(attn_w, -1)
    context_vec = attn_s @ values #[b,..., n, d_v]
    return context_vec