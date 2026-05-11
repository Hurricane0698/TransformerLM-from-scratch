from ch05.gpt2 import GPT_CONFIG_124M, GPTModel, generate
from ch02.dataloader import create_dataloader_v1
import tiktoken
import torch
from ch04.train import text_to_idx, idx_to_text
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = GPTModel(GPT_CONFIG_124M)
model2.load_state_dict(torch.load("model.pth", map_location=device))