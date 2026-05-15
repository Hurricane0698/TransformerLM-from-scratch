# LLM

This repository documents my path toward understanding and researching large
language models from first principles.

The current focus is a from-scratch GPT-style implementation based on
*Build a Large Language Model From Scratch*. The next stage will extend this
foundation toward CS336-level topics: training systems, scaling behavior, data,
evaluation, and research-oriented experiments.

## Current Milestone

`from-scratch-book/scratch_again.ipynb` is the current stage result. It includes
a compact implementation path from basic text processing to a GPT-style language
model:

- tokenizer and dataset pipeline
- token and positional embeddings
- causal self-attention
- multi-head attention
- layer normalization
- feed-forward network
- transformer block
- GPT-style language model
- basic generation experiments

## Repository Structure

```text
llm/
├── from-scratch-book/
│   ├── ch02/
│   ├── ch03/
│   ├── ch04/
│   ├── ch05/
│   ├── scratch_again.ipynb
│   ├── llm-from-scratch.ipynb
│   ├── figures/
│   ├── artifacts/
│   └── notes.md
├── cs336/
│   └── README.md
├── references/
└── README.md
```

`from-scratch-book/` contains the current book-based learning stage. 

## Research Direction

This project is meant to grow from implementation practice into small research
experiments. Future README updates will include experiment figures, such as loss
curves, attention visualizations, sampling comparisons, and training-dynamics
plots produced with Python and Matplotlib.
