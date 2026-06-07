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
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x:torch.Tensor) -> torch.Tensor :
        input_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (torch.sum(x**2, -1, keepdim=True)/ self.d_model + self.eps)**0.5#[b, s, 1]
        norm_x = x / rms *self.weight
        return norm_x.to(input_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model:int, d_ff:int|None = None) -> None:
        super().__init__()
        #设置3个可学习的参数矩阵，利用SiLU(𝑥) = 𝑥 ⋅ 𝜎(𝑥) = 𝑥/(1 + 𝑒−𝑥)激活当作GLU门控元素逐一相乘，fnn内部隐藏层设计为8/3的d_model,取相邻的64倍整数
        if d_ff is None:
            self.d_ff = 4*d_model
            if self.d_ff % 64 != 0:
                r = round(self.d_ff/64)#取最近的64整数因子
                self.d_ff = 64*r
            else:
                self.d_ff = int(self.d_ff)
        else:
            self.d_ff = d_ff
        self.w1 = Linear(d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, d_model)
        #self.w3 = Linear(d_model, self.d_ff)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        activate = self.w1(x)
        gate = (activate)*torch.sigmoid(activate) #[..., dff]
        #gated_x = gate * (self.w3(x))
        return self.w2(gate)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_length:int, device:torch.device|None = None) -> None:
        super().__init__()
        #先写cos,sin和theta组成的通用表达式
        seq_index = torch.arange(max_seq_length, device=device)[:, None]#[s, 1]
        d_k_index = torch.arange(d_k//2, device=device)[None, :]#[1, d_k/2]，这里其实不用unsqueeze也可以，因为右对齐，[d_k/2]会自动补一个[1]
        freq = 1 / theta**(2*d_k_index/d_k)
        #用广播而不是矩阵乘法，广播时从右往左对齐，其中一个维度要么是1要么两个维度相同，维度为1的部分复制乘上其他部分
        angle = seq_index * freq
        self.register_buffer("cos_table", angle.cos(), persistent=False)
        self.register_buffer("sin_table", angle.sin(), persistent=False)
    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        cos = self.cos_table[token_positions]#[b, s, d/2]，查表更通用，因为之后像是 KV Cache 的推理阶段，输入可能不是从 0 开始，就不能用 cos_table[：seq_len]
        sin = self.sin_table[token_positions]
        #把x[...,d]拆成偶数、奇数数列：[...,d/2]
        x_even = x[..., 0::2]#[...,start::step]
        x_odd = x[..., 1::2]
        new_even = cos*x_even - sin*x_odd #先不要想怎么代码写整体计算，先从最小单元想
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
    context_vec = attn_s @ values #[b,..., n, d_v]，不要假设q和kv倒数第二个序列一样，q可能只有1（kv cache），但是kv得保存整个序列信息
    return context_vec

class MutiHeadCausalAttention(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_context_length:int, theta:float|None = None)->None:
        super().__init__()
        #先保存参数，同时计算head dim
        self.d_model = d_model
        self.theta = theta
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_out 必须被 num_heads 整除"
        self.head_dim = d_model // num_heads
        #初始化q_proj,k_proj,v_proj
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        #初始化mask
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(max_context_length, max_context_length,dtype=torch.bool),diagonal=1
            ), persistent=False
        )
        #初始化RoPE,需要theta常数、d, max_length.计算attention之前嵌入相对位置信息
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_context_length)
        self.output_proj = Linear(d_model, d_model)#这一层之前忘记写了

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor|None = None)->torch.Tensor:
        #计算三个向量，对q和k拆分。如果rope存在，应用rope
        *before_dim, s, d= x.shape #前面的维度解包，表达为[b,k] s, d
        assert d == self.d_model, "d_model must be same as d"#这里之前忘记写了，检验一下更好报错清晰
        #Wx:[..., s, d_model]->[..., s, num_heads, head_dim]->[..., num_heads, s, head_dim]
        queires = self.q_proj(x).view(*before_dim, s, self.num_heads, self.head_dim).transpose(-2, -3)
        keys = self.k_proj(x).view(*before_dim, s, self.num_heads, self.head_dim).transpose(-2, -3)
        values = self.v_proj(x).view(*before_dim, s, self.num_heads, self.head_dim).transpose(-2, -3)
        if self.theta is not None:
            #显式检验，报错清晰
            assert token_positions is not None, "RoPE MHA must have token_positions"
            assert token_positions.shape[-1] == s, "token_positions last dim must equal x sequence length"
            queires = self.rope(queires, token_positions)
            keys = self.rope(keys, token_positions)
        context_vec = scaled_dot_product_attention(queires, keys, values, ~self.mask[:s, :s])
        s = context_vec.shape[-2]
        return self.output_proj(context_vec.transpose(-2, -3).contiguous().view(*before_dim, s, self.d_model))
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, max_seq_length:int, theta:float) -> None:
        super().__init__()
        self.attn = MutiHeadCausalAttention(d_model, num_heads, max_seq_length, theta)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        seq_length = x.shape[-2]
        token_positions = torch.arange(seq_length)
        short_cut = x
        x = self.ln1(x)
        x = self.attn(x, token_positions)
        x += short_cut

        short_cut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x += short_cut
        return x

class TransformerLM(nn.Module):
    def __init__(self, 
                vocab_size: int,
                context_length: int,
                d_model: int,
                num_layers: int,
                num_heads: int,
                d_ff: int,
                rope_theta: float) -> None:
        super().__init__() 
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(
            *[TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
              for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        emb_x = self.token_embeddings(x)
        logits = self.layers(emb_x)
        norm_logits = self.ln_final(logits)
        return self.lm_head(norm_logits)
