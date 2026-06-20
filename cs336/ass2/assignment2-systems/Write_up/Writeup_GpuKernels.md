Problem (pytorch_attention): 

| d_model | sequence length | forward time (ms) | backward time (ms) | memory after forward (MiB) | status |
|---:|---:|---:|---:|---:|---|
| 16 | 256 | 0.395 | 0.824 | 20.77 | ok |
| 16 | 1024 | 1.413 | 3.488 | 82.34 | ok |
| 16 | 4096 | 19.214 | 48.015 | 1048.63 | ok |
| 16 | 8192 | 5275.477 | 7313.109 | 4129.00 | ok |
| 16 | 16384 | OOM | OOM | - | OOM |
| 32 | 256 | 0.869 | 1.899 | 21.27 | ok |
| 32 | 1024 | 1.857 | 4.490 | 84.34 | ok |
| 32 | 4096 | 18.958 | 47.361 | 1056.63 | ok |
| 32 | 8192 | 1244.844 | 2457.527 | 4145.00 | ok |
| 32 | 16384 | OOM | OOM | - | OOM |
| 64 | 256 | 0.852 | 2.128 | 22.27 | ok |
| 64 | 1024 | 2.110 | 4.900 | 88.34 | ok |
| 64 | 4096 | 19.170 | 46.954 | 1072.63 | ok |
| 64 | 8192 | 1492.355 | 5442.819 | 4177.00 | ok |
| 64 | 16384 | OOM | OOM | - | OOM |
| 128 | 256 | 1.036 | 2.630 | 24.27 | ok |
| 128 | 1024 | 2.223 | 4.950 | 96.34 | ok |
| 128 | 4096 | 21.028 | 49.097 | 1104.63 | ok |
| 128 | 8192 | 1413.842 | 5191.645 | 4241.00 | ok |
| 128 | 16384 | OOM | OOM | - | OOM |

Based on my calculations, the minimum oom configuration is d_model 16, sequence length 16,384, and batch size 8.

According to my answer from assignment 1, the total attention memory consumption is approximately:
2 × batch size × sequence length × sequence_length × head dimension
Given that the head dimension here equals 1, the final value is approximately 16 GB. Additionally, the activation value for QKV itself is about 24 MB, which is quite small compared to atten_scores.

The memory saved for backward grows quadratically with sequence length in the large-sequence regime, because naive attention saves tensors proportional to the attention matrix of shape B x C x C. In my measurements, for fixed D=16, memory after forward increases from about 1048.75 MiB at C=4096 to 4129.25 MiB at C=8192, which is close to the expected 4x increase when sequence length doubles. At smaller sequence lengths the growth looks closer to linear because fixed overheads and O(BCD) tensors are still visible. To eliminate this memory cost, I would use FlashAttention / memory-efficient attention: tile the attention computation, save only small softmax normalization statistics, and recompute score/probability tiles during backward instead of materializing or saving the full C x C attention matrix.

Problem(torch_compile):
(a):

| d_model | sequence length | vanilla forward (ms) | compiled forward (ms) | vanilla backward (ms) | compiled backward (ms) |
|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 0.395 | 0.450 | 0.824 | 0.911 |
| 16 | 1024 | 1.413 | 1.278 | 3.488 | 2.836 |
| 16 | 4096 | 19.214 | 12.340 | 48.015 | 30.693 |
| 16 | 8192 | 5275.477 | 37.402 | 7313.109 | 910.695 |
| 16 | 16384 | OOM | OOM | OOM | OOM |
| 32 | 256 | 0.869 | 0.419 | 1.899 | 0.753 |
| 32 | 1024 | 1.857 | 1.341 | 4.490 | 3.012 |
| 32 | 4096 | 18.958 | 13.494 | 47.361 | 30.758 |
| 32 | 8192 | 1244.844 | 39.790 | 2457.527 | 895.037 |
| 32 | 16384 | OOM | OOM | OOM | OOM |
| 64 | 256 | 0.852 | 0.819 | 2.128 | 0.943 |
| 64 | 1024 | 2.110 | 2.773 | 4.900 | 3.788 |
| 64 | 4096 | 19.170 | 16.585 | 46.954 | 31.306 |
| 64 | 8192 | 1492.355 | 36.592 | 5442.819 | 552.604 |
| 64 | 16384 | OOM | OOM | OOM | OOM |
| 128 | 256 | 1.036 | 1.423 | 2.630 | 1.918 |
| 128 | 1024 | 2.223 | 2.884 | 4.950 | 3.917 |
| 128 | 4096 | 21.028 | 17.956 | 49.097 | 32.347 |
| 128 | 8192 | 1413.842 | 44.197 | 5191.645 | 657.166 |
| 128 | 16384 | OOM | OOM | OOM | OOM |

(b):
| size | vanilla forward (s) | compiled forward (s) | forward speedup | vanilla backward (s) | compiled backward (s) | backward speedup | vanilla optimizer (s) | compiled optimizer (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 0.016283 | 0.012310 | 1.32x | 0.032120 | 0.025477 | 1.26x | 0.008263 | 0.008284 |
| medium | 0.046885 | 0.037334 | 1.26x | 0.092778 | 0.074819 | 1.24x | 0.017291 | 0.017029 |
| large | 0.105994 | 0.088927 | 1.19x | 0.208353 | 0.174812 | 1.19x | 0.030528 | 0.030300 |
| xl | 0.292266 | 0.276859 | 1.06x | 0.569051 | 0.524198 | 1.09x | 0.080121 | 0.079914 |
| 10B | 0.943759 | 0.898867 | 1.05x | 1.866899 | 1.777618 | 1.05x | OOM | OOM |
