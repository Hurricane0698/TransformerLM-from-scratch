# CS336 TinyStories Track

This directory contains the current main line of the repository: a from-scratch
TransformerLM implementation and TinyStories training experiments.

## Contents

| Path | Purpose |
| --- | --- |
| `ass1/assignment1-basics/cs336_basics/` | core tokenizer, model, optimizer, training, and generation modules |
| `modal_tinystories.py` | Modal launcher for running the same training path on a B200 GPU |
| `scripts/generate_tinystories.py` | sampling script for trained TinyStories checkpoints |
| `experiments/` | run configs, metrics, GPU telemetry, and experiment artifacts |
| `figures/` | report-ready plots and summary tables |
| `notes/` | lecture notes and implementation notes |

## Main Artifacts

- `experiments/b200_lr1p25e-3_bs128_10k/`: final B200 training run.
- `experiments/lr_sweep_comparison_20260529/`: learning-rate sweep comparison.
- `experiments/perf_3070_20260528_202842/`: local RTX 3070 performance run.
- `figures/tinystories_b200_scaling_loss.svg`: B200 loss curve.
- `figures/lr_sweep_research_summary.png`: learning-rate sweep figure.

Large model checkpoints are intentionally ignored by Git. The committed
experiment artifacts are the configs, metrics, telemetry, and plots needed to
inspect the runs.
