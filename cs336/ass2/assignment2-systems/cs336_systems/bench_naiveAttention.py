from cs336_basics.model import scaled_dot_product_attention
import torch
import csv
import timeit
from pathlib import Path
#[8, sequence_length, d_model] 
# sequence_length in [256, 1024, 4096, 8192, 16384] , d_model in [16, 32, 64, 128]
#对每个d_model跑所有sequence_length, 生成q, k, v 向量后warmup5次，跑100次前向，100次反向
#分别测前向时间，前向后memory消耗，反向时间，csv持久化保存

#对象: q,k,v,config, forward time, backwardtime, saved memory for backward，step, device
batch_size = 8
device = "cuda"
step = 100
warmup = 5
#编译版
scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)
output_path = Path("compiled_attention_benchmark.csv")
with output_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "d_model",
            "sequence_length",
            "batch_size",
            "stage",
            "mean_s",
            "memory_after_forward_mib",
            "status"
        ]
    )
    writer.writeheader()

    for d_model in [16, 32, 64, 128]:
        #如果小配置oom,跳过
        skip_larger = False
        for sequence_length in [256, 1024, 4096, 8192, 16384]:
            #先创建占位，防止后面finally出错
            q = k = v = out = loss = None
            if skip_larger:
                continue
            total_forward = 0
            total_backward = 0
            total_memory_f = 0
            try:
                q = torch.randn(batch_size, sequence_length, d_model, device=device,requires_grad=True)
                k = torch.randn(batch_size, sequence_length, d_model, device=device, requires_grad=True)
                v = torch.randn(batch_size, sequence_length, d_model, device=device, requires_grad=True)
                for i in range(step+warmup):
                    #防止梯度累计污染行为
                    q.grad = None
                    k.grad = None
                    v.grad = None
                    #热身
                    if i <= warmup - 1:
                        out = scaled_dot_product_attention(q,k,v)
                        loss = out.sum()
                        loss.backward()
                        torch.cuda.synchronize()
                    else:
                        #前向
                        start = timeit.default_timer()
                        out = scaled_dot_product_attention(q, k, v)
                        torch.cuda.synchronize()
                        end = timeit.default_timer()
                        total_memory_f += torch.cuda.memory_allocated() / 1024**2
                        total_forward += end - start
                        #反向
                        loss = out.sum()
                        torch.cuda.synchronize()
                        start = timeit.default_timer()
                        loss.backward()
                        torch.cuda.synchronize()
                        end = timeit.default_timer()
                        total_backward += end - start
            #写数据
                writer.writerow({
                    "d_model": d_model,
                    "sequence_length": sequence_length,
                    "batch_size": batch_size,
                    "stage": "forward",
                    "mean_s": total_forward / step,
                    "memory_after_forward_mib": total_memory_f / step,
                    "status": "ok",
                })
                writer.writerow({
                    "d_model": d_model,
                    "sequence_length": sequence_length,
                    "batch_size": batch_size,
                    "stage": "backward",
                    "memory_after_forward_mib": "",
                    "mean_s": total_backward / step,
                    "status": "ok",
                })
                f.flush()
            #oom处理
            except torch.OutOfMemoryError:
                writer.writerow({
                    "d_model": d_model,
                    "sequence_length": sequence_length,
                    "batch_size": batch_size,
                    "stage": "forward_backward",
                    "memory_after_forward_mib": "",
                    "mean_s": "",
                    "status": "oom",
                })
                f.flush()
                skip_larger = True
                
            #清理
            finally:
                #删缓存
                del q, k, v, out, loss
                #清空非缓存占用防止污染后续
                torch.cuda.empty_cache()