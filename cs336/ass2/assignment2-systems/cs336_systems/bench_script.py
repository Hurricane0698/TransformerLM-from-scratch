import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import timeit
import torch
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
    parser.add_argument("--warmup-steps", type=int, default=10)
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
    #初始化模型和优化器以及全局变量
    model = BasicsTransformerLM(vocab_size=args.vocab_size, 
                                context_length=context_length, 
                                d_model=args.d_model, 
                                num_layers=args.num_layers,
                                num_heads=args.num_heads, 
                                d_ff=args.d_ff)
    model = model.to(device)
    inputs = data[:, :context_length]
    total_time = 0
    #判断模式，热身，开始计时
    assert mode in ["forward_only", "forward-and-backward", "full-training-steps"]
    if mode == "forward_only":
        model.eval()
        with torch.no_grad():
            for _ in range(t+w):
                if _ <= w-1:
                    model(inputs)
                    torch.cuda.synchronize()#这里也必须同步，因为同步是等所有cuda stream内队列完成，如果这里不同步很可能导致下面计时偏大
                elif _ >= w-1:
                    start = timeit.default_timer()
                    model(inputs)
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    total_time += (end - start)
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
            elif _ >= w-1:
                model.zero_grad()
                start = timeit.default_timer()
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
                loss.backward()
                torch.cuda.synchronize()
                end = timeit.default_timer()
                total_time += (end - start)
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
            elif _ >= w-1:
                model.zero_grad()
                start = timeit.default_timer()
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                end = timeit.default_timer()
                total_time += (end - start)
    return f"Average time: {total_time / t}, Mode:{mode}"
    
if __name__ == "__main__":
    args = build_argparser().parse_args()
    print(main(args))