import argparse
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import timeit
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp.overlap import OverlapDDP
from cs336_basics.nn_utils import cross_entropy

# overlap版从cpu计时器无法看overhead来源，需要使用nsight trace才能看
model_config = {"vocab_size": 10000, "context_length": 512, "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}


@contextmanager
def annotated_range(name: str, profiler_enabled: bool):
    with nvtx.range(name):
        with torch.profiler.record_function(name) if profiler_enabled else nullcontext():
            yield


# rank初始化，选gpu，初始化设置cuda device
def set_up(rank: int, world_size: int):
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1145"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


# rank工作层,负责除了spawn外所有
def benchmark_worker(
    rank: int,
    world_size: int,
    global_batch_size: int,
    model_config: dict,
    warmups=5,
    measured_steps=10,
    profile_trace_dir: str | None = None,
):
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
        xl_model = OverlapDDP(xl_model)
        optimizer = AdamW(xl_model.parameters())
        for _ in range(warmups):
            optimizer.zero_grad()
            logits = xl_model(inp)
            loss = cross_entropy(logits, tar)
            loss.backward()
            xl_model.finish_gradient_synchronization()
            optimizer.step()
            torch.cuda.synchronize()
        # 仅记录端到端耗时
        global_time_list = []
        for step in range(measured_steps):
            profiler_enabled = profile_trace_dir is not None and step == 0
            profiler_context = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                )
                if profiler_enabled
                else nullcontext()
            )
            with profiler_context as profiler:
                with annotated_range("profile_step", profiler_enabled):
                    start = timeit.default_timer()
                    optimizer.zero_grad()
                    with annotated_range("forward", profiler_enabled):
                        logits = xl_model(inp)
                        loss = cross_entropy(logits, tar)
                    with annotated_range("backward", profiler_enabled):
                        loss.backward()
                    with annotated_range("finish_gradient_sync", profiler_enabled):
                        xl_model.finish_gradient_synchronization()
                    with annotated_range("optimizer_step", profiler_enabled):
                        optimizer.step()
                    torch.cuda.synchronize()
                    end = timeit.default_timer()
                    global_time_list.append(end - start)

            if profiler_enabled:
                trace_dir = Path(profile_trace_dir)
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / f"overlap_ddp_rank{rank}.json"
                profiler.export_chrome_trace(str(trace_path))
                print(f"rank {rank} wrote PyTorch Profiler trace to {trace_path}", flush=True)

        gather_size_global = [None for _ in range(world_size)]
        dist.all_gather_object(gather_size_global, global_time_list)

        if rank == 0:
            # 实际时间由最慢worker决定，因此对rank维度取最大值
            true_global_time = torch.tensor(gather_size_global).amax(dim=0)
            mean_total_ms = true_global_time.mean().item() * 1000
            print(f"Mean_total_ms:{mean_total_ms}")
    # 无论成功或者失败都进行资源清理
    finally:
        dist.destroy_process_group()


# 编排层，spawn
def run_benchmark(
    world_size: int,
    model_config: dict,
    global_batch_size: int,
    warmups=5,
    measured_steps=10,
    profile_trace_dir: str | None = None,
):
    assert global_batch_size % world_size == 0, "world_size must divide global_batch_size"
    mp.spawn(
        fn=benchmark_worker,
        args=(world_size, global_batch_size, model_config, warmups, measured_steps, profile_trace_dir),
        join=True,
        nprocs=world_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument("--profile-trace-dir")
    args = parser.parse_args()
    run_benchmark(
        args.world_size,
        model_config,
        args.global_batch_size,
        warmups=args.warmups,
        measured_steps=args.measured_steps,
        profile_trace_dir=args.profile_trace_dir,
    )
