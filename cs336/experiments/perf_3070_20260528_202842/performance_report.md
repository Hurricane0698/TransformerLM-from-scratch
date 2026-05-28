# TinyStories 17M Performance Note: RTX 3070 Laptop

## Scope

The goal is to measure a small local training run, summarize GPU behavior, and extrapolate runtime for a larger token budget under explicit assumptions.

## Run

- Run directory: `cs336/experiments/perf_3070_20260528_202842`
- Model: 17M-style Transformer config, d_model=512, layers=4, heads=16
- Batch/context: 16 x 256 = 4,096 input tokens per optimizer step
- Steps: 10,000
- Observed training tokens: 40,960,000
- Wall-clock training time: 31.2 min
- End-to-end throughput: 21,891 tokens/sec

## GPU Observations

- GPU samples: 1,760 rows over 29.5 min
- Active-period median GPU utilization: 96.0%
- Active-period mean GPU utilization: 95.1%
- Active-period mean power: 101.8 W
- Median active temperature: 86.0 C; max temperature: 87.0 C
- Median active memory used: 3.25 GiB; max memory used: 4.49 GiB

Interpretation: the run is compute-active rather than idle. The high utilization plus sustained high temperature suggests the laptop GPU is near its steady thermal/power operating regime. This is useful context when extrapolating from a laptop 3070 to datacenter GPUs.

## Scaling Projection

Target token budget:

```text
327,680,000 = 128 batch x 10,000 steps x 256 context length
```

Using the measured 3070 end-to-end throughput linearly:

```text
327,680,000 / 21891 tokens/sec = 249.5 min = 4.16 h
```

Reference B200 window from the assignment statement:

- 30 min target: 182,044 tokens/sec, 8.3x measured 3070 throughput
- 20 min target: 273,067 tokens/sec, 12.5x measured 3070 throughput

It assumes the same model, context length, implementation style, and similar non-training overhead accounting.

## Figures

- `gpu_telemetry.svg`: GPU utilization, power, temperature, and memory over wall time
- `throughput_profile.svg`: cumulative and interval training throughput
- `scaling_projection.svg`: runtime projection for the 327.68M-token budget
