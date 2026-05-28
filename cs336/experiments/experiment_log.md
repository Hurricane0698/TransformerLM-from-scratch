Experiment: TinyStories baseline training

Goal:
Validate an end-to-end TransformerLM training pipeline implemented from scratch.

Setup:
- Dataset: TinyStories token ids
- Tokenizer vocab size: 10,000
- Model: 4 layers, d_model 512, 16 heads
- Context length: 256
- Batch size: 16
- Training steps: 10,000
- Tokens processed: 40.96M
- Optimizer: AdamW
- LR schedule: warmup + cosine decay
- Checkpointing and CSV logging enabled

Results:
- Train loss decreased from ~4.40 to ~1.84
- Valid loss decreased from ~4.41 to ~1.84
- Best valid loss around step 8200: ~1.77
- Training time: ~30.5 minutes

Systems / Performance:
- Hardware: RTX 3070 Laptop GPU, 8GB VRAM
- End-to-end throughput: ~21,891 tokens/sec
- Wall-clock time: 31.2 min for 40.96M training tokens
- Active GPU utilization: median 96.0%, mean 95.1%
- GPU memory: median 3.25 GiB, max 4.49 GiB
- Temperature: median 86 C, max 87 C

Observations:
1. The training loop was stable and validation loss tracked train loss closely.
2. Preprocessing/encoding cost exceeded training cost for this short baseline.
3. uint16 token-id storage reduced encoded dataset size relative to raw text.
4. GPU telemetry showed the run was compute-active rather than idle: utilization stayed around 95-96% during active training.
5. Throughput-based projection suggests that the 327.68M-token budget would take ~4.16 hours on this measured 3070 setup, while a 20-30 min B200 target implies roughly 8.3x-12.5x higher end-to-end throughput.

Limitations:
- Only one run, no hyperparameter sweep.
- eval_iters=10 gives noisy validation estimates.
- No qualitative generation analysis yet.
- No comparison against larger/smaller models.
- Throughput is end-to-end wall-clock throughput, including evaluation and checkpoint overhead, not pure GPU kernel throughput.
- The 327.68M-token estimate is a linear runtime projection
- GPU results are from one laptop 3070 run and may be affected by thermal/power limits.

Next steps:
- Increase eval_iters for smoother validation.
- Generate samples from checkpoints.
- Compare model sizes or context lengths.
- Add WandB only when running multiple experiments.

Artifacts:
- Loss curves: cs336/experiments/tinystories_baseline_20260528_175419/loss_vs_step.svg
- GPU telemetry: cs336/experiments/perf_3070_20260528_202842/gpu_telemetry.svg
- Throughput profile: cs336/experiments/perf_3070_20260528_202842/throughput_profile.svg
- Runtime projection: cs336/experiments/perf_3070_20260528_202842/scaling_projection.svg