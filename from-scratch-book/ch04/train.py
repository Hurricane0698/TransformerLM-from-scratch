from ch05.gpt2 import GPT_CONFIG_124M, GPTModel, generate
from ch02.dataloader import create_dataloader_v1
import tiktoken
import torch
def text_to_idx(text, tokenizer):
    encode_tensor = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    return encode_tensor
def idx_to_text(idx, tokenizer):
    text = tokenizer.decode(idx.squeeze(0).tolist())
    return text
tokenizer = tiktoken.get_encoding("gpt2")
torch.manual_seed(123)
with open("ch02/the-verdict.txt", "r", encoding="utf-8") as f:
    verdict = f.read()
train_ratio = 0.9
split_idx = int(train_ratio * len(verdict))
train_text = verdict[:split_idx]
val_text = verdict[split_idx:]

train_loader = create_dataloader_v1(
    train_text,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_text,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

def cal_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1),
            target_batch.flatten()
)    
    return loss

def cal_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = cal_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches

def train_model_simple(model,train_loader, val_loader, device,
                       optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_loss, val_loss, track_token_seed = [], [], []
    tokens_seed, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = cal_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seed += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_losses, val_losses = eval_model(model, train_loader, val_loader, device, eval_iter)
                train_loss.append(train_losses)
                val_loss.append(val_losses)
                track_token_seed.append(tokens_seed)
                print(f"epoch {epoch}, step {global_step}, train loss: {train_losses}, val loss: {val_losses}")
            generate_sample(model, start_context, tokenizer, device)
    return track_token_seed, train_loss, val_loss
def eval_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = cal_loss_loader(train_loader, model, device, eval_iter)
        val_loss = cal_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss
def generate_sample(model, start_context, tokenizer, device):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encode = text_to_idx(start_context, tokenizer).to(device)
    with torch.no_grad():
        generated_idx = generate(model, encode, max_new_tokens=50, context_size=context_size)
    generated_text = idx_to_text(generated_idx, tokenizer)
    print(generated_text.replace("\n", " "))
    model.train()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)
num_epochs = 1
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, device, optimizer,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
torch.save({
"model_state_dict": model.state_dict(),
"optimizer_state_dict": optimizer.state_dict(),
},
"model_and_optimizer.pth"
)