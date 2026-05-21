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