# Environment

This repository uses a project-local Python environment for LLM research work.
The goal is to keep the WSL system clean while still avoiding unnecessary
duplication of large packages such as PyTorch.

## Layers

```text
WSL system layer
  git, uv, shell tools, build tools, editor integration

Research workspace layer
  project repos, papers, datasets, checkpoints, experiment logs

Project environment layer
  .venv, pyproject.toml, uv.lock, notebook kernel
```

## Practical Policy

- Keep Python packages out of the global system Python.
- Use one long-lived `.venv` for this LLM research repository.
- Do not create a new PyTorch environment for every small experiment.
- Create a separate environment only when a project needs incompatible versions,
  such as a different PyTorch, CUDA, Triton, or FlashAttention stack.
- Keep datasets, checkpoints, logs, and raw course materials out of git.
- Keep research projects and software engineering projects in separate
  workspaces when possible, for example `~/Research/` and `~/Projects/`.

## Disk Strategy

PyTorch is large. With a 512G disk, the right tradeoff is not one environment per
small idea. Instead:

- keep this repository's `.venv` as the main LLM research environment;
- let `uv` reuse its global package cache when environments are recreated;
- split a new environment only for a real dependency conflict;
- store large artifacts under ignored directories such as `artifacts/`,
  `checkpoints/`, `datasets/`, or `runs/`.


## Common Commands

Create or update the project environment:

```bash
uv sync
```

Run Python inside the project environment:

```bash
uv run python
```

Register the notebook kernel:

```bash
uv run python -m ipykernel install --user --name llm-research --display-name "Python (llm research)"
```

Start Jupyter:

```bash
uv run jupyter notebook
```

## CS336 Materials

The official CS336 lecture repository is external course material. It can be
cloned locally for study, but it should not be committed into this repository.

Recommended local path:

```text
cs336/lectures/
```

The repository should track original notes, experiments, figures, and writeups
under:

```text
cs336/notes/
cs336/experiments/
cs336/figures/
```
