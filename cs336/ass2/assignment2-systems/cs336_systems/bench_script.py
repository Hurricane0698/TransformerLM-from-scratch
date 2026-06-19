import argparse
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from contextlib import nullcontext
import timeit
import torch
import torch.cuda.nvtx as nvtx
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
    #出现flag开启混合精度
    parser.add_argument("--use-bf16", action="store_true")
    #or: parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    #...("--use-bf16",type=bool, default=True),eg:--use-bf16 False，命令行收到非空字符串"False"，任何非空字符串都是 True
    
    parser.add_argument("--bench-memory", action="store_true")
    parser.add_argument("--profile-autograd-nvtx", action="store_true")
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
    time_list = []
    #判断模式，热身，开始计时
    assert mode in ["forward_only", "forward-and-backward", "full-training-steps"]
    #确定是否使用混合精度
    if args.use_bf16 :
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()

    if mode == "forward_only":
        model.eval()
        with torch.no_grad():
            #memory记录分支
            if args.bench_memory:
                for i in range(w+1):
                    if i <= w-1:
                        with ctx:
                            model(inputs)
                        torch.cuda.synchronize()
                    else:
                        with ctx:
                            torch.cuda.memory._record_memory_history(max_entries=1000000)
                            model(inputs)
                            pt = f"memory_ctx{context_length}_forward.pickle"
                            torch.cuda.synchronize()
                            torch.cuda.memory._dump_snapshot(f"memory_ctx{context_length}_forward.pickle")
                            torch.cuda.memory._record_memory_history(enabled=None)
                return f"save to {pt}"

            for _ in range(t+w):
                if _ <= w-1:
                    #不同精度kernel，cache不一样，所以也需要ctx包住
                    with ctx:
                        model(inputs)
                    torch.cuda.synchronize()#这里也必须同步，因为同步是等所有cuda stream内队列完成，如果这里不同步很可能导致下面计时偏大
                
                elif _ >= w:
                    if _ == w:
                        nvtx.range_push("benchmark_measure")
                    start = timeit.default_timer()
                    with ctx:
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
                with ctx:
                    logits = model(inputs)
                    loss = cross_entropy(logits, targets)
                loss.backward()
                torch.cuda.synchronize()
            elif _ >= w:
                if _ == w:
                    nvtx.range_push("benchmark_measure")
                model.zero_grad()
                #如果是混合精度，生成混合精度的前向图供backward使用(backward就不需要包在ctx下面了)
                with ctx:
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
        #memory记录分支
        if args.bench_memory:
            for i in range(w+1):
                if i <= w-1:
                    model.zero_grad()
                    with ctx:
                        logits = model(inputs)
                        loss = cross_entropy(logits, targets)
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                else:
                    model.zero_grad()
                    torch.cuda.memory._record_memory_history(max_entries=1000000)
                    with ctx:
                        logits = model(inputs)
                        loss = cross_entropy(logits, targets)
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    pt = f"memory_ctx{context_length}_train.pickle"
                    torch.cuda.memory._dump_snapshot(pt)
                    torch.cuda.memory._record_memory_history(enabled=None)
            return f"save to {pt}"
        
        #autograd分支：
        if args.profile_autograd_nvtx:
            for i in range(w + 1):
                if i < w:
                    model.zero_grad()
                    logits = model(inputs)
                    loss = cross_entropy(logits, targets)
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                else:
                    with nvtx.range("benchmark_measure"):
                        with torch.autograd.profiler.emit_nvtx():
                            model.zero_grad()
                            with ctx:
                                logits = model(inputs)
                                loss = cross_entropy(logits, targets)
                            loss.backward()
                            optimizer.step()
                            torch.cuda.synchronize()
            return "profile done"
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
