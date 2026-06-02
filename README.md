# Is Attention All You Need (For Residuals)?

**Testing Ziming Liu's hypothesis that Attention Residuals' gains come from hidden state rescaling, not learned depth-wise routing.**

Built on [wdlctc/open-attention-residuals](https://github.com/wdlctc/open-attention-residuals), which open-sourced the Kimi Team's [Attention Residuals](https://arxiv.org/abs/2603.15031) (arXiv:2603.15031).

---

## Background

Standard transformers use additive residual connections:
```
h = h + sublayer(h)
```

The **Attention Residuals** paper (Kimi Team, 2025) replaces this with learned softmax attention over previous layer representations, showing improved performance over vanilla transformers.

[Ziming Liu's blog post](https://kindxiaoming.github.io/blog/2026/attention-residual-2/) challenged this, arguing the gains come primarily from **hidden state magnitude bounding** — a side effect of the softmax weighted average — rather than the learned routing itself. He tested this on toy models and called for empirical validation at LLM scale:

> *"Kimi's results would be more convincing if they also evaluated a rescaled residual baseline."*

This repo provides that baseline.

---

## The Experiment

We train three variants of a 115M parameter transformer from scratch on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) for 20,000 steps with identical hyperparameters. The only difference is the residual connection.

### Variants

**Baseline** — standard additive residual:
```python
h = h + sublayer(h)
```

**Block AttnRes** — learned cross-block attention (original paper):
```python
V = stack([block_0, block_1, ..., partial_block])
weights = softmax(learned_query @ normalize(V))
h = weighted_sum(weights, V)
h = h + sublayer(h)
```
Learns *which* previous layer's output to route through. Adds ~24K parameters.

**Rescaled Residual** — deterministic per-layer damping (our new variant, testing Liu's hypothesis):
```python
alpha = 1.0 / (layer_idx + 1)
h = h + alpha * sublayer(h)
```
Layer 0: `alpha=1.0` (standard). Layer 11: `alpha=0.083` (heavily damped). Bounds hidden state magnitude without any learned parameters or attention mechanism. **Zero extra parameters.**

### Model Config
```
Architecture:  Qwen3 (from scratch, no pretrained weights)
Parameters:    115.6M
Hidden size:   512
Layers:        12
Attention heads: 8 (4 KV heads)
FFN size:      1536
Sequence len:  1024
Training data: FineWeb-Edu (streaming)
Steps:         20,000
Batch size:    2 per GPU × 2 GPUs × 4 grad accum = 16,384 tokens/step
Learning rate: 6e-4 with cosine decay, 1000 step warmup
```

---

## Results

| Metric | Baseline | Rescaled Residual | Block AttnRes |
|--------|----------|-------------------|---------------|
| WikiText-2 PPL ↓ | 90.31 | 94.08 | **88.89** |
| LAMBADA Acc ↑ | 0.088 | 0.088 | **0.090** |
| HellaSwag Acc ↑ | 0.315 | 0.320 | **0.335** |
| Final Train Loss ↓ | 3.674 | 3.688 | **3.640** |

All three variants trained with identical hyperparameters on identical data. Evaluated on WikiText-2 test set (PPL), LAMBADA (500 samples), and HellaSwag (200 samples).

### Training Curves

All three variants show nearly identical loss trajectories during training — the curves overlap throughout the 20k steps. The differences emerge in downstream evaluation, not training loss.

---

## Findings

**1. Block AttnRes consistently outperforms baseline.**
Across all four metrics, Block AttnRes beats the standard residual baseline. This replicates wdlctc's published direction of results at 100M scale.

**2. Rescaled Residual partially supports Liu's hypothesis.**
Rescaled residual beats or matches baseline on HellaSwag (0.320 vs 0.315) and LAMBADA (0.088 tie), suggesting that magnitude bounding does provide some benefit. However it underperforms on PPL (94.08 vs 90.31), indicating the hypothesis is only partially correct.

**3. Learned routing adds value beyond rescaling.**
The gap between rescaled (0.320) and block AttnRes (0.335) on HellaSwag — the most reliable benchmark here — shows the learned depth-wise attention mechanism contributes meaningfully over and above simple magnitude control.

**4. The ranking is clear:**
```
Block AttnRes > Rescaled Residual ≈ Baseline
```

**5. Both effects matter.**
Rescaling alone gets part of the way to AttnRes's gains. Learned routing adds more on top. The answer to Liu's question is "both" — not one or the other.

---

## Limitations

- **Single scale (100M).** Results at 300M+ may differ. wdlctc's clearest results are at 600M.
- **Short context (seq_len=1024).** Original paper used 2048. Longer contexts likely amplify the benefits of cross-layer routing.
- **Single LR.** No per-variant LR sweep. Different architectures may have different optimal LRs.
- **Small eval samples.** 200 HellaSwag / 500 LAMBADA samples have high variance. Full benchmark eval needed.
- **Single seed.** Results not averaged over multiple runs.

---

## What's Next

To make these results publication-ready:
- Run at 300M and 600M scale (matching wdlctc's setup)
- Sweep learning rates (3e-4, 6e-4, 1e-3) per variant
- Test `1/sqrt(l+1)` and other rescaling formulas
- Full benchmark eval (10k+ samples)
- Multiple seeds

---

## How to Run

### Setup
```bash
git clone https://github.com/rishie17/is-atention-all-you-need.git
cd is-atention-all-you-need
pip install -r requirements.txt
```

### Train
```bash
# Baseline
torchrun --nproc_per_node=1 train.py --mode baseline \
    --steps 20000 --batch_size 4 --grad_accum 4 \
    --seq_len 1024 --dtype fp16 --lr 6e-4

# Block AttnRes
torchrun --nproc_per_node=1 train.py --mode block --num_blocks 4 \
    --steps 20000 --batch_size 4 --grad_accum 4 \
    --seq_len 1024 --dtype fp16 --lr 6e-4

# Rescaled Residual
torchrun --nproc_per_node=1 train.py --mode rescaled \
    --steps 20000 --batch_size 4 --grad_accum 4 \
    --seq_len 1024 --dtype fp16 --lr 6e-4
```

Use `--dtype bf16` on A100. Use `--nproc_per_node=2` for 2xT4 (Kaggle).

### Evaluate
```bash
python eval.py --model_path output/scratch-baseline-d512-L12-20k/final --mode baseline
python eval.py --model_path output/scratch-block-d512-L12-20k/final --mode block
python eval.py --model_path output/scratch-rescaled-d512-L12-20k/final --mode rescaled
```

---

## Acknowledgments

- [Attention Residuals](https://arxiv.org/abs/2603.15031) — Kimi Team (original paper)
- [wdlctc/open-attention-residuals](https://github.com/wdlctc/open-attention-residuals) — open-source implementation we built on
- [Ziming Liu's blog post](https://kindxiaoming.github.io/blog/2026/attention-residual-2/) — hypothesis we tested
- [Qwen3](https://arxiv.org/abs/2505.09388) — base architecture
