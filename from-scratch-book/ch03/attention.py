import torch.nn as nn
import torch
class selfattentionv1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_q = nn.Parameter(torch.randn(d_in, d_out))
        self.w_k = nn.Parameter(torch.randn(d_in, d_out))
        self.w_v = nn.Parameter(torch.randn(d_in, d_out))
    def forward(self, x):
        q = x @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v
        attn_scores = q @ k.T
        attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ v
        return context_vec
torch.manual_seed(123)
sav1 = selfattentionv1(3, 2)

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec