from __future__ import annotations

import argparse
import csv
import gc
import itertools
from functools import partial
from pathlib import Path

import torch
import triton.testing

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attention import (
    compiled_backward,
    triton_flash_attention_backward,
    triton_flash_attention_forward,
)

#确定基本不变常量和数据写入title
BATCH_SIZE = 1
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
FIELDS = (
    "gpu", "batch_size", "causal", "sequence_length", "embedding_dim", "dtype", "provider",
    "q_tile_size", "k_tile_size", "num_warps", "stage", "latency_ms", "status", "error",
)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def parse_csv_dtypes(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(","))
    return values


def parse_tile_configs(value: str) -> tuple[tuple[int, int, int], ...]:
    #q_tile x k_tile x num_warps，16x32x4
    configs = []
    for item in value.split(","):
        config = tuple(int(number) for number in item.split("x"))
        if len(config) != 3:
            raise argparse.ArgumentTypeError("tile格式必须是q_tile x k_tile x num_warps")
        configs.append(config)
    return tuple(configs)


class BenchmarkFlashAttention(torch.autograd.Function):
    #给benchmark传tile

    @staticmethod
    def forward(ctx, Q, K, V, q_tile_size, k_tile_size, num_warps):
        output, logsumexp = triton_flash_attention_forward(Q, K, V, q_tile_size, k_tile_size, num_warps, True)
        ctx.q_tile_size = q_tile_size
        ctx.k_tile_size = k_tile_size
        ctx.num_warps = num_warps
        ctx.save_for_backward(logsumexp, Q, K, V, output)
        return output

    @staticmethod
    def backward(ctx, dO):
        logsumexp, Q, K, V, output = ctx.saved_tensors
        dQ, dK, dV = triton_flash_attention_backward(
            Q, K, V, output, dO, logsumexp, True,
            ctx.q_tile_size, ctx.k_tile_size, ctx.num_warps,
        )
        return dQ, dK, dV, None, None, None


class BenchmarkCompiledBackwardFlashAttention(torch.autograd.Function):
    """Historical comparator: Triton forward plus the previous compiled PyTorch backward."""

    @staticmethod
    def forward(ctx, Q, K, V, q_tile_size, k_tile_size, num_warps):
        output, logsumexp = triton_flash_attention_forward(Q, K, V, q_tile_size, k_tile_size, num_warps, True)
        ctx.save_for_backward(logsumexp, Q, K, V, output)
        return output

    @staticmethod
    def backward(ctx, dO):
        logsumexp, Q, K, V, output = ctx.saved_tensors
        dQ, dK, dV = compiled_backward(Q, K, V, output, dO, logsumexp, True)
        return dQ, dK, dV, None, None, None


def make_inputs(sequence_length: int, embedding_dim: int, dtype: torch.dtype, seed: int):
    #对象: Q/K/V/dO都是[B,N,D]；每个provider重新设seed，保证两边输入相同
    torch.manual_seed(seed)
    Q, K, V, dO = (
        torch.randn(BATCH_SIZE, sequence_length, embedding_dim, device="cuda", dtype=dtype)
        for _ in range(4)
    )
    return Q.requires_grad_(), K.requires_grad_(), V.requires_grad_(), dO


def benchmark_three_stages(forward, Q, K, V, dO, warmup_ms: int, rep_ms: int):
    #不变量: forward只测forward；backward复用同一张graph；forward_backward每次新建graph
    forward_ms = triton.testing.do_bench(forward, warmup=warmup_ms, rep=rep_ms, return_mode="mean")

    output = forward()

    def backward():
        output.backward(dO, retain_graph=True)

    backward_ms = triton.testing.do_bench(
        backward, warmup=warmup_ms, rep=rep_ms, grad_to_none=[Q, K, V], return_mode="mean"
    )

    def forward_backward():
        forward().backward(dO)

    forward_backward_ms = triton.testing.do_bench(
        forward_backward, warmup=warmup_ms, rep=rep_ms, grad_to_none=[Q, K, V], return_mode="mean"
    )
    return {"forward": forward_ms, "backward": backward_ms, "forward_backward": forward_backward_ms}


def write_rows(writer, base: dict, timings: dict):
    for stage, latency_ms in timings.items():
        writer.writerow({**base, "stage": stage, "latency_ms": float(latency_ms), "status": "ok", "error": ""})


def clear_case():
    #每个provider单独释放显存，避免PyTorch的N*N mask影响后面的Triton结果
    gc.collect()
    torch.cuda.empty_cache()


def run_sweep(
    sequence_lengths: tuple[int, ...], embedding_dims: tuple[int, ...], dtype_names: tuple[str, ...],
    tile_configs: tuple[tuple[int, int, int], ...], warmup_ms: int, rep_ms: int, seed: int,
    output_path: Path, expected_gpu: str = "",
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("需要CUDA GPU")
    gpu = torch.cuda.get_device_name()
    if expected_gpu and expected_gpu.lower() not in gpu.lower():
        raise RuntimeError(f"需要{expected_gpu}, 当前是{gpu}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        #sweep顺序: N -> D -> dtype；PyTorch只跑一次，Triton再遍历不同tile
        for N, D, dtype_name in itertools.product(sequence_lengths, embedding_dims, dtype_names):
            Q, K, V, dO = make_inputs(N, D, DTYPES[dtype_name], seed)
            index = torch.arange(N, device="cuda")
            causal_mask = index[:, None] >= index[None, :]

            pytorch_forward = partial(scaled_dot_product_attention, Q, K, V, mask=causal_mask)
            timings = benchmark_three_stages(pytorch_forward, Q, K, V, dO, warmup_ms, rep_ms)
            base = {
                "gpu": gpu, "batch_size": BATCH_SIZE, "causal": True, "sequence_length": N,
                "embedding_dim": D, "dtype": dtype_name, "provider": "pytorch",
                "q_tile_size": "", "k_tile_size": "", "num_warps": "",
            }
            write_rows(writer, base, timings)
            file.flush()
            del Q, K, V, dO, index, causal_mask, pytorch_forward, timings
            clear_case()

            providers = (
                ("triton_compiled_backward", BenchmarkCompiledBackwardFlashAttention),
                ("triton", BenchmarkFlashAttention),
            )
            for q_tile, k_tile, num_warps in tile_configs:
                for provider, implementation in providers:
                    Q, K, V, dO = make_inputs(N, D, DTYPES[dtype_name], seed)

                    triton_forward = partial(implementation.apply, Q, K, V, q_tile, k_tile, num_warps)
                    timings = benchmark_three_stages(triton_forward, Q, K, V, dO, warmup_ms, rep_ms)
                    base = {
                        "gpu": gpu, "batch_size": BATCH_SIZE, "causal": True, "sequence_length": N,
                        "embedding_dim": D, "dtype": dtype_name, "provider": provider,
                        "q_tile_size": q_tile, "k_tile_size": k_tile, "num_warps": num_warps,
                    }
                    write_rows(writer, base, timings)
                    file.flush()
                    del Q, K, V, dO, triton_forward, timings
                    clear_case()
    return output_path


def build_parser():
    parser = argparse.ArgumentParser(description="FlashAttention-2 B200 benchmark")
    parser.add_argument("--sequence-lengths", type=parse_csv_ints, default=parse_csv_ints("128,256,512,1024,2048,4096,8192,16384,32768,65536"))
    parser.add_argument("--embedding-dims", type=parse_csv_ints, default=parse_csv_ints("16,32,64,128"))
    parser.add_argument("--dtypes", type=parse_csv_dtypes, default=parse_csv_dtypes("bfloat16,float32"))
    parser.add_argument("--tile-configs", type=parse_tile_configs, default=parse_tile_configs("16x32x4"))
    parser.add_argument("--warmup-ms", type=int, default=25)
    parser.add_argument("--rep-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("flash_attention_benchmark_b200.csv"))
    return parser


def main():
    args = build_parser().parse_args()
    path = run_sweep(args.sequence_lengths, args.embedding_dims, args.dtypes, args.tile_configs,
                     args.warmup_ms, args.rep_ms, args.seed, args.output)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
