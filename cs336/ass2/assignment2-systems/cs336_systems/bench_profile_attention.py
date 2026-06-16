import argparse
import cs336_basics.model as model_module
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import timeit
import torch
import torch.cuda.nvtx as nvtx
import math
from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
     Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        with nvtx.range("mask"):
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
        
    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with nvtx.range("final matmul"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")

def build_argparser():
    parser = argparse.ArgumentParser(description="测量模型前向、反向传播及训练耗时脚本")
    #确定模型超参数配置
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=3072)

    #再确定实验参数
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--time-step", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    return parser

def main(args):
    '''对象：模型，优化器，计时器，随机生成的数据、loss、步数。不变量：时间必须反映warmup后真实训练耗时，步数必须反映实际步数，
    测量方式必须和模式相对应。
    先得到初始化数据，再初始化模型和优化器，再判断模式，根据模式进行warmup，然后计时，最后结束。
    检查device和shape,type'''
    #初始化参数
    context_length = args.context_length
    mode = args.mode.strip()
    device = args.device
    w = args.warmup_steps
    t = args.time_step
    #生成随机数据
    data = torch.randint(0, args.vocab_size,(args.batch_size, context_length+1), device=device)
    #初始化模型和优化器以及全局变量, 在创建 BasicsTransformerLM 之前，把模块里的函数临时替换掉
    model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    model = BasicsTransformerLM(vocab_size=args.vocab_size, 
                                context_length=context_length, 
                                d_model=args.d_model, 
                                num_layers=args.num_layers,
                                num_heads=args.num_heads, 
                                d_ff=args.d_ff)
    model = model.to(device)
    inputs = data[:, :context_length]
    time_list = []
    #判断模式，热身，开始计时
    assert mode in ["forward_only", "forward-and-backward", "full-training-steps"]
    if mode == "forward_only":
        model.eval()
        with torch.no_grad():
            for _ in range(t+w):
                if _ <= w-1:
                    model(inputs)
                    torch.cuda.synchronize()#这里也必须同步，因为同步是等所有cuda stream内队列完成，如果这里不同步很可能导致下面计时偏大
                
                elif _ >= w:
                    if _ == w:
                        nvtx.range_push("benchmark_measure")
                    start = timeit.default_timer()
                    model(inputs)
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    time_list.append((end - start))
            nvtx.range_pop()
    elif mode == "forward-and-backward":
        model.train()
        targets = data[:, 1:context_length+1]
        for _ in range(t+w):
            if _ <= w-1:
                model.zero_grad()
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
                loss.backward()
                torch.cuda.synchronize()
            elif _ >= w:
                if _ == w:
                    nvtx.range_push("benchmark_measure")
                model.zero_grad()
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
                torch.cuda.synchronize()
                #测backward
                start = timeit.default_timer()
                loss.backward()
                torch.cuda.synchronize()
                end = timeit.default_timer()
                time_list.append((end - start))
        nvtx.range_pop()
    elif mode == "full-training-steps":
        model.train()
        targets = data[:, 1:context_length+1]
        optimizer = AdamW(model.parameters())
        for _ in range(t+w):
            if _ <= w-1:
                model.zero_grad()
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
            elif _ >= w:
                if _ == w:
                    nvtx.range_push("benchmark_measure")
                model.zero_grad()
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
                loss.backward()
                torch.cuda.synchronize()
                #测optimizer step
                start = timeit.default_timer()
                optimizer.step()
                torch.cuda.synchronize()
                end = timeit.default_timer()
                time_list.append((end - start))
        nvtx.range_pop()
    time_tensor = torch.tensor(time_list)
    avg_time = time_tensor.mean(-1,keepdim=False)
    std = time_tensor.std(-1, unbiased=False, keepdim=False)
    return f"Average time: {avg_time}, Standard Deviation: {std}, Mode:{mode}"
    
if __name__ == "__main__":
    args = build_argparser().parse_args()
    print(main(args))