# Is Attention All You Need (For Residuals)?

Testing [Ziming Liu's hypothesis](https://kindxiaoming.github.io/blog/2026/attention-residual-2/) that the gains from [Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, arXiv:2603.15031) come primarily from **hidden state rescaling**, not from learned depth-wise routing.

Built on top of [wdlctc/open-attention-residuals](https://github.com/wdlctc/open-attention-residuals).

---

## The Hypothesis

The original AttnRes paper replaces standard additive residual connections with softmax attention over previous block representations:

```
h_l = Σ α_{i→l} · s_i        (AttnRes: learned routing via softmax)
```

Ziming Liu's blog argues this is doing two things at once:
1. **Rescaling** hidden state magnitude (bounded like a weighted average)
2. **Routing** — attending to specific earlier layers

His hypothesis: the rescaling alone explains most of the gains. You can test this by replacing AttnRes with a simple deterministic formula that rescales but doesn't route:

```
h = (l/(l+1)) * h + (1/(l+1)) * f(h)        (Rescaled Residual)
```

This keeps hidden state magnitude bounded without any learned parameters or attention mechanism.

---

## The Experiment

Train four variants at 100M scale (`d=512, L=12`) on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu):

| Variant | Description |
|---------|-------------|
| `baseline` | Standard additive residuals (vanilla Qwen3) |
| `block` | Block AttnRes — cross-block attention (N=4 blocks) |
| `full` | Full AttnRes — per-sublayer cross-layer attention |
| `rescaled` | Rescaled Residual — deterministic per-layer scaling, no learned params |

Evaluate on: **WikiText-2 PPL**, **LAMBADA accuracy**, **HellaSwag accuracy**

If `rescaled` ≈ `block`/`full`, Liu's hypothesis is supported.  
If `rescaled` < `block`/`full`, learned routing matters.

---

## Results

> Training in progress. Will update with numbers.

| Model | WikiText-2 PPL | LAMBADA Acc | HellaSwag Acc |
|-------|----------------|-------------|---------------|
| Baseline | — | — | — |
| Block AttnRes | — | — | — |
| Rescaled Residual | — | — | — |

For reference, the original wdlctc results at 100M scale (20k steps):

| Model | Train Loss | WikiText-2 PPL | LAMBADA Acc | HellaSwag Acc |
|-------|-----------|----------------|-------------|---------------|
| Baseline | 3.303 | 60.21 | 0.082 | 0.325 |
| Block AttnRes | **3.350** | **55.69** | **0.114** | **0.340** |

---

## Setup

```bash
git clone https://github.com/rishie17/is-atention-all-you-need.git
cd is-atention-all-you-need
pip install -r requirements.txt
```

---

## Training

### Single GPU (Colab / local)

```bash
# Baseline
python train.py --mode baseline --grad_accum 8

# Block AttnRes (4 blocks)
python train.py --mode block --num_blocks 4 --grad_accum 8

# Full AttnRes
python train.py --mode full --grad_accum 8

# Rescaled Residual (our new variant)
python train.py --mode rescaled --grad_accum 8
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `baseline` | `baseline`, `block`, `full`, `rescaled` |
| `--steps` | `20000` | Training steps |
| `--batch_size` | `4` | Per-GPU batch size |
| `--grad_accum` | `2` | Gradient accumulation steps |
| `--lr` | `6e-4` | Peak learning rate |
| `--hidden_size` | `512` | Model hidden size |
| `--num_layers` | `12` | Number of transformer layers |
| `--num_blocks` | `4` | AttnRes blocks (block mode only) |
| `--out_dir` | auto | Checkpoint output directory |
| `--wandb_project` | `residual` | W&B project name |

**Effective batch size** = `batch_size × grad_accum × num_GPUs`. With defaults on a single GPU: `4 × 2 = 8` sequences of 2048 tokens = ~16k tokens/step.

For a fair comparison across all variants, use **identical** `--lr`, `--steps`, and `--grad_accum`. We recommend sweeping `--lr` over `[3e-4, 6e-4, 1e-3]` since hidden state scale differences can affect optimal LR.

### Multi-GPU (if available)

```bash
torchrun --nproc_per_node=8 train.py --mode block --num_blocks 4
```

---

## Evaluation

After training, run all three benchmarks in one shot:

```bash
python eval.py \
    --model_path output/scratch-rescaled-d512-L12-20k/final \
    --mode rescaled
```

This evaluates:
1. **WikiText-2 perplexity** — held-out language modeling
2. **LAMBADA** — last-word prediction accuracy (500 samples)
3. **HellaSwag** — commonsense completion (200 samples)

`--mode` must match what you trained: `baseline`, `block`, `full`, or `rescaled`.

---

## Architecture

```
100M config: d=512, L=12, heads=8, kv_heads=4, ff=1536
```

### How Block AttnRes works

Instead of standard residuals:
```
partial = partial + sublayer_output(partial)
```

Block AttnRes attends over all completed block representations:
```python
def block_attn_res(blocks, partial_block, proj, norm, recency_bias):
    V = torch.stack(blocks + [partial_block])       # (N+1, B, T, D)
    K = norm(V)                                      # RMSNorm keys
    query = proj.weight.view(-1)                     # learned query (D,)
    logits = einsum("d, n b t d -> n b t", query, K)
    logits[-1] += recency_bias                       # large init → stable start
    weights = softmax(logits, dim=0)
    return einsum("n b t, n b t d -> b t d", weights, V)
```

### How Rescaled Residual works

Per-layer deterministic scaling — no learned params, no attention:
```python
# In forward pass, layer_idx = 0, 1, ..., L-1
alpha = 1.0 / (layer_idx + 1)
partial_block = partial_block + alpha * attn_out
partial_block = partial_block + alpha * mlp_out
```

Layer 0: `alpha=1.0` (standard residual), Layer 11: `alpha=0.09` (heavily damped).

AttnRes adds per layer: 2× projection vectors + 2× RMSNorm = **~0.03% extra parameters**.  
Rescaled Residual adds: **zero extra parameters**.

---

## Key Files

```
modeling_attnres.py   — all four model variants (Qwen3AttnResForCausalLM)
train.py              — training loop, single-GPU and multi-GPU
eval.py               — WikiText-2 PPL, LAMBADA, HellaSwag
requirements.txt      — pip dependencies
```

---

## Acknowledgments

- [Attention Residuals](https://arxiv.org/abs/2603.15031) — Kimi Team (original paper)
- [wdlctc/open-attention-residuals](https://github.com/wdlctc/open-attention-residuals) — open-source implementation we built on
- [Ziming Liu's blog post](https://kindxiaoming.github.io/blog/2026/attention-residual-2/) — hypothesis we're testing
- [Qwen3](https://arxiv.org/abs/2505.09388) — base architecture
