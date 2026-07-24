import os
import torch
import timeit
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.naive_ddp import NaiveDDP
from cs336_basics.nn_utils import cross_entropy

# 先创建模型XL参数config
# worker内，完成forward数据创建，初始化以及通过naive ddp进行同步
# worker计算自己loss，通过naiveddp同步梯度再optimizer更新，记录平均的总耗时和grad同步耗时，同时计算衍生指标占比
# 打印结果
model_config = {"vocab_size": 10000, "context_length": 512, "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}


# rank初始化，选gpu，初始化设置cuda device
def set_up(rank: int, world_size: int):
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1145"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


# rank工作层,负责除了spawn外所有
def benchmark_worker(rank: int, world_size: int, global_batch_size: int, model_config: dict, warmups=5, measured_steps=10):
    try:
        set_up(rank, world_size)
        local_batch_size = global_batch_size // world_size
        inp = torch.randint(0, model_config["vocab_size"], (local_batch_size, model_config["context_length"]), device="cuda")
        tar = torch.randint(0, model_config["vocab_size"], (local_batch_size, model_config["context_length"]), device="cuda")
        xl_model = BasicsTransformerLM(
            model_config["vocab_size"], model_config["context_length"], model_config["d_model"], model_config["num_layers"], model_config["num_heads"], model_config["d_ff"]
        )
        # 先把模型移动到gpu
        xl_model.to("cuda")
        xl_model = NaiveDDP(xl_model)
        optimizer = AdamW(xl_model.parameters())
        for _ in range(warmups):
            optimizer.zero_grad()
            logits = xl_model(inp)
            loss = cross_entropy(logits, tar)
            loss.backward()
            xl_model.synchronize_gradients()
            optimizer.step()
            torch.cuda.synchronize()

        global_time_list = []
        synchronize_time_list = []
        for _ in range(measured_steps):
            start1 = timeit.default_timer()
            optimizer.zero_grad()
            logits = xl_model(inp)
            loss = cross_entropy(logits, tar)
            loss.backward()
            # 等backward kernel执行完再计时
            torch.cuda.synchronize()
            start2 = timeit.default_timer()
            xl_model.synchronize_gradients()
            torch.cuda.synchronize()
            end2 = timeit.default_timer()
            optimizer.step()
            torch.cuda.synchronize()
            end1 = timeit.default_timer()

            global_time_list.append(end1 - start1)
            synchronize_time_list.append(end2 - start2)

        gather_size_global = [None for _ in range(world_size)]
        gather_size_local = [None for _ in range(world_size)]
        dist.all_gather_object(gather_size_global, global_time_list)
        dist.all_gather_object(gather_size_local, synchronize_time_list)
        if rank == 0:
            # 实际时间由最慢worker决定，因此对rank维度取最大值
            true_global_time = torch.tensor(gather_size_global).amax(dim=0)
            true_syn_time = torch.tensor(gather_size_local).amax(dim=0)
            mean_total_ms = true_global_time.mean().item() * 1000
            mean_syn_ms = true_syn_time.mean().item() * 1000
            prop = mean_syn_ms / mean_total_ms
            print(f"Mean_total_ms:{mean_total_ms}, Mean_syn_ms:{mean_syn_ms}, Proportion:{prop * 100}%")
    # 无论成功或者失败都进行资源清理
    finally:
        dist.destroy_process_group()


# 编排层，spawn
def run_benchmark(world_size: int, model_config: dict, global_batch_size: int, warmups=5, measured_steps=10):
    assert global_batch_size % world_size == 0, "world_size must divide global_batch_size"
    mp.spawn(fn=benchmark_worker, args=(world_size, global_batch_size, model_config, warmups, measured_steps), join=True, nprocs=world_size)


if __name__ == "__main__":
    global_batch_size = 4
    world_size = 2
    run_benchmark(world_size, model_config, global_batch_size)
