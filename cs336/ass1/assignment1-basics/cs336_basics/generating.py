import torch
def generation(prompts:str, 
               model, device, tokenizer, context_size:int, max_new_tokens:int, temperature:float,top_p:float, eos_id=None): 
    assert temperature > 0., "温度必须大于0"
    token_ids = torch.tensor(tokenizer.encode(prompts), device=device)
    with torch.no_grad(): 
        for _ in range(max_new_tokens):
            context = token_ids[-context_size:]
            logits = model(context) #[T, V]
            last_logits = logits[-1, :]
            last_logits = last_logits / temperature
            sorted_probs, sorted_indices = torch.softmax(last_logits, dim=-1).sort(-1, True)
            cumprobs = torch.cumsum(sorted_probs, -1)
            reached = (cumprobs >= top_p)
            delete_mask = torch.zeros_like(reached)
            delete_mask[1:] = reached[:-1]
            sorted_probs.masked_fill_(delete_mask, 0)
            sorted_token = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices[sorted_token]
             
            if eos_id is not None and (next_token == eos_id).all().item(): #如果想有一个eosid就停止可以用any()
                break
            token_ids = torch.cat((token_ids, next_token), dim=0)
    return tokenizer.decode(token_ids.tolist())