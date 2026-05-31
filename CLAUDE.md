
## What this repo is
Open-source replication of "Attention Residuals" (Kimi Team, arXiv:2603.15031) by wdlctc. Implements baseline, Block AttnRes, and Full AttnRes variants for language model pretraining.

## What we are doing
Testing Ziming Liu's hypothesis (https://kindxiaoming.github.io/blog/2026/attention-residual-2/) that AttnRes's gains come primarily from hidden state rescaling, not from learned depth-wise routing.

We are adding a fourth variant: **Rescaled Residual**, using the deterministic formula:
This keeps hidden state magnitude bounded (like AttnRes) but without any learned parameters or attention mechanism.

## The experiment
- Train three variants at 100M scale (d=512, L=12) on FineWeb-Edu: baseline, Block AttnRes, rescaled residual
- Use identical hyperparameters across all three (with 2-3 learning rate sweeps, since Liu showed scale differences affect LR fairness)
- Evaluate on WikiText-2 PPL, LAMBADA accuracy, HellaSwag accuracy
- Compare against wdlctc's published results as sanity check

## Key files
- `modeling_attnres.py` — model architecture, where AttnRes forward pass lives
- `train.py` — training script (supports torchrun multi-GPU and single GPU)
- `eval.py` — evaluation script
- `CLAUDE.md` — this file

## Constraints
- Single GPU training (Colab T4/A100 or similar), no multi-GPU
- Need to adapt train.py for single-GPU with gradient accumulation
- Keep changes minimal — only add the rescaled residual variant, don't refactor existing code
EOF

# Project Context

## What this repo is
Open-source replication of "Attention Residuals" (Kimi Team, arXiv:2603.15031) by wdlctc. Implements baseline, Block AttnRes, and Full AttnRes variants for language model pretraining.

## What we are doing
Testing Ziming Liu's hypothesis that AttnRes gains come primarily from hidden state rescaling, not learned depth-wise routing.

Adding a Rescaled Residual variant using: h = (l/(l+1)) * h + (1/(l+1)) * f(h)

This bounds hidden state magnitude (like AttnRes) but with no learned parameters.

## The experiment
- Train baseline, Block AttnRes, and rescaled residual at 100M scale on FineWeb-Edu
- Identical hyperparameters, sweep 2-3 learning rates
- Evaluate on WikiText-2 PPL, LAMBADA, HellaSwag
- Sanity check against wdlctc's published numbers

## Key files
- modeling_attnres.py — model architecture, AttnRes forward pass
- train.py — training script
- eval.py — evaluation script

## Constrai
- Single GPU training (Colab T4/A100)
- Keep changes minimal — only add rescaled residual variant
