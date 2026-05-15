import torch
import torch.nn as nn
from ch03.mutiheadattention import MultiHeadAttention
import tiktoken
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate_att": 0.1,
    "drop_rate_emb": 0.1,
    "drop_rate_shortcut": 0.1,
    "qkv_bias": False
}
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"])
        self.trf_blocks = nn.Sequential(
        *[TransformerBlock(cfg)
        for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
        cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):

        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
        torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        context_length=cfg["context_length"],
        num_heads=cfg["n_heads"],
        dropout=cfg["drop_rate_att"],
        qkv_bias=cfg["qkv_bias"]
        )
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.ffn = FeedForward(cfg)
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate_shortcut"])
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = x + shortcut
        x = self.drop_shortcut(x)

        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + shortcut
        x = self.drop_shortcut(x)
        return x
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.episilon = 1e-5
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.episilon)
        return self.scale * norm + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        next_token_logits = logits[:, -1, :]
        if top_k is not None:
            top_logits,  _ = torch.topk(next_token_logits, top_k)
            min_top_logits = top_logits[:, -1]
            next_token_logits = torch.where(
                next_token_logits < min_top_logits,
                torch.tensor(-torch.inf),
                next_token_logits
            )
        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
            prob = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(prob, num_samples=1)
        else:
            prob = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        if next_token.item() == eos_id:
            break
        idx = torch.cat((idx, next_token), dim=1)
    return idx
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)
start_context_1 = "every effort moves"
start_context_2 = "I really like"
encoded1 = tokenizer.encode(start_context_1)
encoded2 = tokenizer.encode(start_context_2)
encoded_tensor1 = torch.tensor(encoded1).unsqueeze(0)
encoded_tensor2 = torch.tensor(encoded2).unsqueeze(0)
inputs = torch.cat(
    (encoded_tensor1, encoded_tensor2),
    dim=0
)
print("inputs shape:", inputs.shape)
model.eval()

