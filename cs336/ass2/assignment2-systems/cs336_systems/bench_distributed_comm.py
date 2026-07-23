# 目标：all reduce，2，4，6个GPU上，数据量1MB,10MB,100MB,1GB,dtype为float32

import argparse
import os
import torch
import timeit
import torch.multiprocessing as mp
import torch.distributed as dist
import csv
from pathlib import Path


def build_argparse():
    parser = argparse.ArgumentParser(description="distributed_communication_single_node")

    parser.add_argument("--experiment-name", type=str, required=True)

    parser.add_argument("--world-size", type=int, default=2)

    parser.add_argument("--backend", type=str, default="gloo")

    parser.add_argument("--data-size", type=int, default=1, help="how many of MB on a single gpu")
    parser.add_argument("--warm-ups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    return parser


def set_up(rank: int, world_size: int, backend: str):
    if backend == "nccl":
        # 不同rank用不同gpu
        torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1145"
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)


def bench_distributed(rank: int, world_size: int, data_size: int, backend: str, warm_ups: int, repeats: int, experiment_name, run_dir: Path):
    # 先set_up，再生成对应大小tensor,再time起始，计算，time结尾
    set_up(rank, world_size, backend)
    assert data_size > 0, "data_size must be positive"
    num_elements = data_size * 1024 * 1024 // 4
    # 由backend选dtype
    data = torch.randn(num_elements, device="cuda" if backend == "nccl" else "cpu")
    # 5次warmup
    for _ in range(warm_ups):
        dist.all_reduce(data, async_op=False)
    time_list = []
    for _ in range(repeats):
        if backend == "nccl":
            torch.cuda.synchronize()
        start = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        if backend == "nccl":
            torch.cuda.synchronize()
        end = timeit.default_timer()
        time = end - start
        time_list.append(time)
    gather_size = [None for _ in range(world_size)]
    dist.all_gather_object(gather_size, time_list)
    gather_time = torch.tensor(gather_size)  # [world_size, repeat]

    # rank0计算mean/max并写csv
    if rank == 0:
        mean = gather_time.mean().item() * 1000
        max = gather_time.max().item() * 1000
        csv_path = run_dir / "table.csv"
        b = csv_path.exists()
        with open(run_dir / "table.csv", "a", newline="", encoding="utf-8") as f:
            Filednames = ["experiment_name", "backend", "world_size", "data_size_mib", "warmups", "repeats", "mean_ms", "max_ms"]
            writer = csv.DictWriter(f, fieldnames=Filednames)
            if not b:
                writer.writeheader()
            writer.writerow(
                {
                    "experiment_name": experiment_name,
                    "backend": backend,
                    "world_size": world_size,
                    "data_size_mib": data_size,
                    "warmups": warm_ups,
                    "repeats": repeats,
                    "mean_ms": mean,
                    "max_ms": max,
                }
            )

        # 进程清理
    dist.destroy_process_group()


if __name__ == "__main__":
    args = build_argparse().parse_args()
    world_size = args.world_size
    backend = args.backend
    data_size = args.data_size
    warmups = args.warm_ups
    repeats = args.repeats
    experiment_name = args.experiment_name
    run_dir = Path("cs336/experiments/distributed_communication_single_node")
    run_dir.mkdir(parents=True, exist_ok=True)
    mp.spawn(fn=bench_distributed, args=(world_size, data_size, backend, warmups, repeats, experiment_name, run_dir), nprocs=world_size, join=True)
