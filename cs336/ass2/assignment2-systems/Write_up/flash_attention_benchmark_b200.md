# FlashAttention-2 Benchmark on NVIDIA B200

Batch size is 1 and causal masking is enabled. Latencies are mean milliseconds reported by `triton.testing.do_bench` with 25 ms warmup and 100 ms measurement. The Triton configuration is `Q_TILE_SIZE=16`, `K_TILE_SIZE=32`, `num_warps=4`.

> The complete tables below preserve the original benchmark. In those tables, Triton backward means Triton forward plus the `torch.compile` PyTorch recomputation backward.

## Optional Triton backward follow-up

| N | PyTorch B | compiled B | Triton B | compiled / Triton | PyTorch / Triton | compiled F+B | Triton F+B | F+B improvement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.187 | 0.119 | 0.062 | 1.91x | 3.01x | 0.179 | 0.091 | 1.97x |
| 1024 | 0.184 | 0.112 | 0.062 | 1.81x | 2.96x | 0.166 | 0.087 | 1.91x |
| 4096 | 0.391 | 0.138 | 0.169 | 0.82x | 2.32x | 0.224 | 0.258 | 0.87x |
| 16384 | 5.288 | 1.677 | 1.615 | 1.04x | 3.27x | 2.450 | 2.389 | 1.03x |
| 65536 | 81.549 | 34.552 | 23.370 | 1.48x | 3.49x | 46.564 | 35.367 | 1.32x |

This is a deliberately small follow-up rather than a tile sweep. The strongest
large-sequence result is at `N=65536`: the tiled backward is `1.48x` faster than
the previous compiled backward and `3.49x` faster than the PyTorch baseline;
forward plus backward improves by `1.32x` over the previous implementation.
At `N=4096`, however, compiled backward is faster. The data therefore does not
support claiming a uniform speedup across sequence lengths. The earlier
three-stage run was executed separately, so this table does not use cross-run
ratios to claim a fusion-only speedup.

## Triton backward tile and warp sweep

The tuning objective is backward latency, not a single global tile claim. The
search used three stages:

1. Compile smoke for all 27 combinations of `Q_TILE_SIZE`, `K_TILE_SIZE` in
   `{16,32,64}` and `num_warps` in `{2,4,8}`.
2. Screen all 27 configurations at BF16, `d=64`, and
   `N={1024,4096,16384,65536}`.
3. Re-test seven candidates across 12 `(N,d)` shapes, then confirm the default,
   `64x64x4`, and `64x64x8` on the six long-sequence shapes with 100 ms warmup
   and 500 ms measurement.


`64x64x8` is the robust long-sequence winner. It won every confirmed shape for
`N={16384,65536}` and `d={32,64,128}`. The default column is the original
`16x32x4` Triton backward. Compiled latency is the median repeated measurement
of Triton forward plus compiled PyTorch backward; naive is the course
`scaled_dot_product_attention`, which materializes the full attention matrix.

| N | d | best config | Triton B | default B | tuning gain | compiled B | vs compiled | naive B | vs naive |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 32 | 64x64x8 | 0.546 | 1.570 | 2.88x | 2.180 | 3.99x | 5.265 | 9.65x |
| 16384 | 64 | 64x64x8 | 0.605 | 1.616 | 2.67x | 2.200 | 3.63x | 5.264 | 8.69x |
| 16384 | 128 | 64x64x8 | 1.413 | 2.650 | 1.88x | 2.224 | 1.57x | 5.294 | 3.75x |
| 65536 | 32 | 64x64x8 | 6.582 | 22.149 | 3.36x | 34.395 | 5.23x | 81.532 | 12.39x |
| 65536 | 64 | 64x64x8 | 8.936 | 23.360 | 2.61x | 34.454 | 3.86x | 81.528 | 9.12x |
| 65536 | 128 | 64x64x8 | 19.458 | 38.764 | 1.99x | 35.257 | 1.81x | 82.024 | 4.22x |

The same configuration also wins the confirmed forward-plus-backward cases:

| N | d | Triton F+B | best compiled F+B | speedup | naive F+B | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 16384 | 32 | 0.881 | 2.468 | 2.80x | 8.168 | 9.28x |
| 16384 | 64 | 0.977 | 2.569 | 2.63x | 8.185 | 8.38x |
| 16384 | 128 | 1.871 | 2.684 | 1.43x | 8.218 | 4.39x |
| 65536 | 32 | 10.227 | 37.543 | 3.67x | 126.065 | 12.33x |
| 65536 | 64 | 13.405 | 38.925 | 2.90x | 126.073 | 9.40x |
| 65536 | 128 | 26.013 | 41.804 | 1.61x | 126.943 | 4.88x |

The mechanism matches the tile ownership in the kernels. Increasing the query
tile reduces the number of times each key/value tile is reread and shortens the
query loop in the key-owned `dK/dV` kernel. Increasing the key tile shortens the
key loop in the query-owned `dQ` kernel. Eight warps expose enough parallelism
for the larger `64x64` matrix products. The benefit shrinks at `d=128` because
the persistent gradient accumulators and temporary score tiles consume more
registers, reducing the resource advantage of larger tiles.

This is not a universal default for short sequences. In the 12-shape re-test,
the sub-millisecond `N<=4096` winners varied with `d`; across all 12 shapes,
`64x64x8` had the best geometric-mean normalized backward latency but could be
up to `2.15x` slower than the per-shape winner. Use `64x64x8` as the established
long-sequence B200 configuration, not as an architecture-independent optimum.

## bfloat16

| N | d | PyTorch F | Triton F | F speedup | PyTorch B | Triton B | B speedup | PyTorch F+B | Triton F+B | F+B speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 0.048 | 0.008 | 5.79x | 0.167 | 0.111 | 1.50x | 0.312 | 0.170 | 1.84x |
| 128 | 32 | 0.048 | 0.008 | 5.83x | 0.146 | 0.101 | 1.45x | 0.290 | 0.160 | 1.81x |
| 128 | 64 | 0.049 | 0.008 | 5.89x | 0.149 | 0.103 | 1.45x | 0.303 | 0.178 | 1.71x |
| 128 | 128 | 0.049 | 0.010 | 4.82x | 0.150 | 0.095 | 1.57x | 0.284 | 0.155 | 1.84x |
| 256 | 16 | 0.048 | 0.010 | 4.77x | 0.148 | 0.094 | 1.58x | 0.287 | 0.155 | 1.85x |
| 256 | 32 | 0.047 | 0.010 | 4.61x | 0.148 | 0.095 | 1.56x | 0.285 | 0.154 | 1.85x |
| 256 | 64 | 0.048 | 0.010 | 4.55x | 0.154 | 0.093 | 1.66x | 0.286 | 0.152 | 1.89x |
| 256 | 128 | 0.047 | 0.012 | 3.82x | 0.183 | 0.115 | 1.59x | 0.329 | 0.157 | 2.09x |
| 512 | 16 | 0.049 | 0.012 | 4.00x | 0.162 | 0.118 | 1.37x | 0.322 | 0.165 | 1.96x |
| 512 | 32 | 0.049 | 0.014 | 3.42x | 0.168 | 0.115 | 1.47x | 0.356 | 0.185 | 1.92x |
| 512 | 64 | 0.048 | 0.016 | 2.94x | 0.156 | 0.107 | 1.45x | 0.294 | 0.174 | 1.69x |
| 512 | 128 | 0.048 | 0.018 | 2.58x | 0.166 | 0.104 | 1.60x | 0.311 | 0.173 | 1.80x |
| 1024 | 16 | 0.058 | 0.020 | 2.84x | 0.152 | 0.108 | 1.40x | 0.283 | 0.173 | 1.64x |
| 1024 | 32 | 0.057 | 0.023 | 2.50x | 0.153 | 0.095 | 1.60x | 0.302 | 0.158 | 1.90x |
| 1024 | 64 | 0.056 | 0.025 | 2.27x | 0.154 | 0.105 | 1.46x | 0.291 | 0.158 | 1.84x |
| 1024 | 128 | 0.057 | 0.031 | 1.86x | 0.151 | 0.094 | 1.61x | 0.294 | 0.157 | 1.87x |
| 2048 | 16 | 0.093 | 0.034 | 2.72x | 0.168 | 0.094 | 1.79x | 0.288 | 0.156 | 1.84x |
| 2048 | 32 | 0.093 | 0.040 | 2.34x | 0.168 | 0.112 | 1.50x | 0.306 | 0.157 | 1.94x |
| 2048 | 64 | 0.093 | 0.043 | 2.15x | 0.165 | 0.096 | 1.73x | 0.291 | 0.278 | 1.05x |
| 2048 | 128 | 0.093 | 0.053 | 1.75x | 0.170 | 0.120 | 1.42x | 0.298 | 0.181 | 1.64x |
| 4096 | 16 | 0.202 | 0.068 | 2.99x | 0.392 | 0.170 | 2.30x | 0.585 | 0.232 | 2.52x |
| 4096 | 32 | 0.202 | 0.079 | 2.58x | 0.391 | 0.171 | 2.29x | 0.586 | 0.244 | 2.40x |
| 4096 | 64 | 0.202 | 0.092 | 2.19x | 0.391 | 0.171 | 2.29x | 0.585 | 0.257 | 2.27x |
| 4096 | 128 | 0.204 | 0.127 | 1.61x | 0.394 | 0.180 | 2.19x | 0.591 | 0.299 | 1.98x |
| 8192 | 16 | 0.811 | 0.229 | 3.54x | 1.433 | 0.576 | 2.49x | 2.229 | 0.798 | 2.79x |
| 8192 | 32 | 0.812 | 0.201 | 4.04x | 1.433 | 0.572 | 2.50x | 2.232 | 0.768 | 2.91x |
| 8192 | 64 | 0.811 | 0.249 | 3.26x | 1.432 | 0.571 | 2.51x | 2.228 | 0.814 | 2.74x |
| 8192 | 128 | 0.818 | 0.347 | 2.35x | 1.442 | 0.586 | 2.46x | 2.244 | 0.927 | 2.42x |
| 16384 | 16 | 2.933 | 0.738 | 3.98x | 5.284 | 2.160 | 2.45x | 8.195 | 2.895 | 2.83x |
| 16384 | 32 | 2.932 | 0.621 | 4.72x | 5.273 | 2.160 | 2.44x | 8.185 | 2.774 | 2.95x |
| 16384 | 64 | 2.934 | 0.781 | 3.76x | 5.292 | 2.181 | 2.43x | 8.213 | 2.957 | 2.78x |
| 16384 | 128 | 2.951 | 1.351 | 2.18x | 5.316 | 2.201 | 2.42x | 8.249 | 3.476 | 2.37x |
| 32768 | 16 | 11.251 | 2.816 | 4.00x | 20.527 | 8.519 | 2.41x | 31.755 | 11.332 | 2.80x |
| 32768 | 32 | 11.250 | 2.428 | 4.63x | 20.522 | 8.519 | 2.41x | 31.738 | 10.944 | 2.90x |
| 32768 | 64 | 11.264 | 3.056 | 3.69x | 20.508 | 8.598 | 2.39x | 31.747 | 11.653 | 2.72x |
| 32768 | 128 | 11.331 | 4.569 | 2.48x | 20.609 | 8.672 | 2.38x | 31.915 | 13.226 | 2.41x |
| 65536 | 16 | 44.637 | 11.007 | 4.06x | 81.774 | 34.515 | 2.37x | 126.444 | 45.503 | 2.78x |
| 65536 | 32 | 44.640 | 9.514 | 4.69x | 81.659 | 34.487 | 2.37x | 126.337 | 43.986 | 2.87x |
| 65536 | 64 | 44.476 | 12.025 | 3.70x | 81.761 | 34.627 | 2.36x | 126.292 | 46.641 | 2.71x |
| 65536 | 128 | 45.080 | 17.357 | 2.60x | 82.220 | 35.443 | 2.32x | 127.305 | 52.770 | 2.41x |

## float32

| N | d | PyTorch F | Triton F | F speedup | PyTorch B | Triton B | B speedup | PyTorch F+B | Triton F+B | F+B speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 16 | 0.055 | 0.008 | 6.62x | 0.175 | 0.108 | 1.62x | 0.340 | 0.169 | 2.01x |
| 128 | 32 | 0.053 | 0.008 | 6.43x | 0.165 | 0.113 | 1.46x | 0.324 | 0.177 | 1.83x |
| 128 | 64 | 0.053 | 0.010 | 5.13x | 0.166 | 0.112 | 1.48x | 0.330 | 0.177 | 1.86x |
| 128 | 128 | 0.047 | 0.010 | 4.54x | 0.145 | 0.090 | 1.61x | 0.269 | 0.150 | 1.80x |
| 256 | 16 | 0.052 | 0.010 | 5.07x | 0.159 | 0.105 | 1.51x | 0.310 | 0.170 | 1.82x |
| 256 | 32 | 0.052 | 0.011 | 4.86x | 0.152 | 0.102 | 1.49x | 0.328 | 0.164 | 1.99x |
| 256 | 64 | 0.046 | 0.012 | 3.74x | 0.145 | 0.107 | 1.36x | 0.277 | 0.169 | 1.64x |
| 256 | 128 | 0.054 | 0.014 | 3.75x | 0.167 | 0.104 | 1.61x | 0.295 | 0.172 | 1.71x |
| 512 | 16 | 0.053 | 0.014 | 3.70x | 0.184 | 0.139 | 1.33x | 0.376 | 0.207 | 1.82x |
| 512 | 32 | 0.053 | 0.016 | 3.21x | 0.169 | 0.113 | 1.49x | 0.320 | 0.180 | 1.77x |
| 512 | 64 | 0.052 | 0.018 | 2.79x | 0.176 | 0.114 | 1.54x | 0.331 | 0.176 | 1.88x |
| 512 | 128 | 0.051 | 0.023 | 2.24x | 0.173 | 0.107 | 1.62x | 0.333 | 0.176 | 1.90x |
| 1024 | 16 | 0.061 | 0.023 | 2.71x | 0.177 | 0.111 | 1.59x | 0.330 | 0.173 | 1.91x |
| 1024 | 32 | 0.061 | 0.025 | 2.48x | 0.160 | 0.108 | 1.48x | 0.305 | 0.172 | 1.77x |
| 1024 | 64 | 0.062 | 0.031 | 2.02x | 0.166 | 0.106 | 1.56x | 0.311 | 0.174 | 1.79x |
| 1024 | 128 | 0.062 | 0.037 | 1.67x | 0.158 | 0.108 | 1.47x | 0.308 | 0.171 | 1.80x |
| 2048 | 16 | 0.099 | 0.039 | 2.53x | 0.208 | 0.106 | 1.96x | 0.313 | 0.170 | 1.85x |
| 2048 | 32 | 0.099 | 0.043 | 2.30x | 0.193 | 0.109 | 1.78x | 0.308 | 0.179 | 1.72x |
| 2048 | 64 | 0.099 | 0.054 | 1.84x | 0.195 | 0.115 | 1.70x | 0.317 | 0.170 | 1.86x |
| 2048 | 128 | 0.099 | 0.068 | 1.47x | 0.196 | 0.109 | 1.80x | 0.310 | 0.175 | 1.78x |
| 4096 | 16 | 0.271 | 0.090 | 3.01x | 0.571 | 0.211 | 2.71x | 0.834 | 0.298 | 2.80x |
| 4096 | 32 | 0.270 | 0.090 | 3.00x | 0.551 | 0.211 | 2.61x | 0.813 | 0.299 | 2.72x |
| 4096 | 64 | 0.272 | 0.125 | 2.18x | 0.555 | 0.214 | 2.59x | 0.817 | 0.335 | 2.44x |
| 4096 | 128 | 0.274 | 0.174 | 1.57x | 0.562 | 0.219 | 2.57x | 0.824 | 0.390 | 2.11x |
| 8192 | 16 | 1.008 | 0.303 | 3.33x | 1.973 | 0.668 | 2.95x | 2.962 | 0.967 | 3.06x |
| 8192 | 32 | 1.008 | 0.266 | 3.78x | 1.956 | 0.667 | 2.93x | 2.948 | 0.929 | 3.17x |
| 8192 | 64 | 1.010 | 0.365 | 2.77x | 1.958 | 0.670 | 2.92x | 2.952 | 1.031 | 2.86x |
| 8192 | 128 | 1.024 | 0.792 | 1.29x | 1.991 | 0.715 | 2.78x | 2.998 | 1.505 | 1.99x |
| 16384 | 16 | 3.699 | 0.979 | 3.78x | 7.375 | 2.512 | 2.94x | 11.055 | 3.494 | 3.16x |
| 16384 | 32 | 3.702 | 0.862 | 4.29x | 7.322 | 2.510 | 2.92x | 10.999 | 3.371 | 3.26x |
| 16384 | 64 | 3.708 | 1.300 | 2.85x | 7.318 | 2.505 | 2.92x | 11.007 | 3.801 | 2.90x |
| 16384 | 128 | 3.722 | 2.343 | 1.59x | 7.338 | 2.573 | 2.85x | 11.047 | 4.888 | 2.26x |
| 32768 | 16 | 14.425 | 3.740 | 3.86x | 28.778 | 9.847 | 2.92x | 43.178 | 13.594 | 3.18x |
| 32768 | 32 | 14.444 | 3.365 | 4.29x | 28.801 | 9.898 | 2.91x | 43.233 | 13.259 | 3.26x |
| 32768 | 64 | 14.494 | 4.773 | 3.04x | 28.865 | 9.996 | 2.89x | 43.343 | 14.772 | 2.93x |
| 32768 | 128 | 14.554 | 7.786 | 1.87x | 28.947 | 10.210 | 2.84x | 43.479 | 17.912 | 2.43x |
| 65536 | 16 | 57.249 | 14.615 | 3.92x | 114.853 | 40.668 | 2.82x | 172.192 | 54.735 | 3.15x |
| 65536 | 32 | 57.319 | 13.289 | 4.31x | 114.923 | 40.201 | 2.86x | 172.163 | 53.450 | 3.22x |
| 65536 | 64 | 57.642 | 18.967 | 3.04x | 115.320 | 40.688 | 2.83x | 172.663 | 59.613 | 2.90x |
| 65536 | 128 | 57.723 | 31.059 | 1.86x | 116.362 | 43.098 | 2.70x | 172.943 | 71.903 | 2.41x |
