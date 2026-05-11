import torch.nn as nn 
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        assert self.d_out % self.num_heads == 0, "d_out must be divisible by num_heads"
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_outproject = nn.Linear(d_out, d_out)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        batch_size, num_tokens, self.d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(batch_size, num_tokens, 
                         self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(batch_size, num_tokens, 
                               self.num_heads, self.head_dim).transpose(1, 2)
        att_scores = queries @ keys.transpose(2, 3)
        att_scores.masked_fill_(self.mask[:num_tokens, :num_tokens], -torch.inf)
        att_weights = torch.softmax(att_scores / self.head_dim**0.5, dim=-1)
        att_weights = self.dropout(att_weights)
        values = values.view(batch_size, num_tokens, 
                              self.num_heads, self.head_dim).transpose(1, 2)
        context_vec = att_weights @ values
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch_size, 
                                                                    num_tokens, self.d_out)
        context_vec = self.W_outproject(context_vec)
        return context_vec
