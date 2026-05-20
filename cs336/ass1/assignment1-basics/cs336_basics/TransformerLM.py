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