import torch
import torch.nn as nn


# 书里的 6 个输入词元嵌入，每个词元是 3 维向量
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your
        [0.55, 0.87, 0.66],  # journey
        [0.57, 0.85, 0.64],  # starts
        [0.22, 0.58, 0.33],  # with
        [0.77, 0.25, 0.10],  # one
        [0.05, 0.80, 0.55],  # step
    ]
)

d_in = inputs.shape[1]  # 输入维度：3
d_out = 2               # 输出维度：2


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        # v1：手动创建权重矩阵
        # 形状是 (d_in, d_out)
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        # v2：用 nn.Linear 做同样的矩阵变换
        # 注意：nn.Linear 内部权重形状是 (d_out, d_in)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5,
            dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec


# 创建 v1
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)

# 创建 v2
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)


print("====== 复制权重之前 ======")
print("SelfAttention_v1 输出：")
print(sa_v1(inputs))

print("\nSelfAttention_v2 输出：")
print(sa_v2(inputs))

print("\n两者是否相同？")
print(torch.allclose(sa_v1(inputs), sa_v2(inputs)))


# 核心：把 v2 的权重复制到 v1
# 因为 nn.Linear 的 weight 是 (d_out, d_in)
# 而 v1 需要的是 (d_in, d_out)
# 所以必须转置 .T
with torch.no_grad():
    sa_v1.W_query.copy_(sa_v2.W_query.weight.T)
    sa_v1.W_key.copy_(sa_v2.W_key.weight.T)
    sa_v1.W_value.copy_(sa_v2.W_value.weight.T)


print("\n\n====== 复制权重之后 ======")
print("SelfAttention_v1 输出：")
print(sa_v1(inputs))

print("\nSelfAttention_v2 输出：")
print(sa_v2(inputs))

print("\n两者是否相同？")
print(torch.allclose(sa_v1(inputs), sa_v2(inputs)))