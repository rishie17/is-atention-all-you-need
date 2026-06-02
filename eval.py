"""
Evaluate from-scratch models on held-out data and simple benchmarks.

Usage:
    python eval.py --model_path output/scratch-baseline-d512-L12-20k/final --mode baseline
    python eval.py --model_path output/scratch-block-d512-L12-20k/final --mode block
    python eval.py --model_path output/scratch-full-d512-L12-20k/final --mode full
"""

import argparse
import sys
import os
import math

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Local import
from modeling_attnres import Qwen3AttnResForCausalLM, Qwen3AttnResConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--mode", required=True, choices=["baseline", "block", "full", "rescaled"])
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--num_samples", type=int, default=200,
                   help="Number of evaluation samples")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def load_model(model_path, mode, device):
    import os
    from safetensors.torch import load_file as load_safetensors
    from transformers import AutoConfig

    # Load config from local path
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)

    # Build model from config
    if mode == "baseline":
        model = Qwen3ForCausalLM(config)
    else:
        model = Qwen3AttnResForCausalLM(config)

    # Load weights manually — bypass from_pretrained's HF repo validation
    weights_path = os.path.join(model_path, "model.safetensors")
    state_dict = load_safetensors(weights_path, device=device)
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


def eval_perplexity(model, tokenizer, seq_len, num_samples, device, dataset_name="wikitext",
                    dataset_config="wikitext-2-raw-v1", split="test"):
    """Compute perplexity on a dataset."""
    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)

    nlls = []
    total_tokens = 0

    # Sliding window evaluation
    max_pos = min(input_ids.size(1), num_samples * seq_len)
    for begin in tqdm(range(0, max_pos, seq_len), desc="Evaluating"):
        end = min(begin + seq_len, input_ids.size(1))
        chunk = input_ids[:, begin:end]

        with torch.no_grad():
            outputs = model(input_ids=chunk)
            logits = outputs.logits

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = chunk[:, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction="sum")
        nll = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                       shift_labels.view(-1))
        nlls.append(nll.item())
        total_tokens += shift_labels.numel()

        if total_tokens >= num_samples * seq_len:
            break

    avg_nll = sum(nlls) / total_tokens
    ppl = math.exp(avg_nll)
    return avg_nll, ppl, total_tokens


def eval_lambada(model, tokenizer, device, max_samples=500):
    """Evaluate on LAMBADA (last word prediction accuracy)."""
    ds = load_dataset("lambada", split="test")
    correct = 0
    total = 0

    for sample in tqdm(ds.select(range(min(max_samples, len(ds)))), desc="LAMBADA"):
        text = sample["text"]
        # Split into context and last word
        words = text.strip().split()
        if len(words) < 2:
            continue
        last_word = words[-1]
        context = " ".join(words[:-1])

        input_ids = tokenizer(context, return_tensors="pt")["input_ids"].to(device)
        target_ids = tokenizer(" " + last_word, add_special_tokens=False)["input_ids"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            next_token_logits = outputs.logits[0, -1, :]
            predicted_id = next_token_logits.argmax().item()

        if len(target_ids) > 0 and predicted_id == target_ids[0]:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    return acc, correct, total


def eval_hellaswag(model, tokenizer, device, max_samples=200):
    """Evaluate on HellaSwag (commonsense completion)."""
    ds = load_dataset("Rowan/hellaswag", split="validation")

    correct = 0
    total = 0

    for sample in tqdm(ds.select(range(min(max_samples, len(ds)))), desc="HellaSwag"):
        ctx = sample["ctx"]
        endings = sample["endings"]
        label = int(sample["label"])

        scores = []
        for ending in endings:
            text = ctx + " " + ending
            input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=512)["input_ids"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits

            # Score = avg log-prob of the ending tokens
            ctx_ids = tokenizer(ctx, return_tensors="pt")["input_ids"]
            ctx_len = ctx_ids.size(1)

            shift_logits = logits[:, ctx_len-1:-1, :]
            shift_labels = input_ids[:, ctx_len:]

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            score = token_log_probs.mean().item()
            scores.append(score)

        if scores.index(max(scores)) == label:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    return acc, correct, total


def main():
    args = parse_args()

    print(f"Loading {args.mode} model from {args.model_path}...")
    model = load_model(args.model_path, args.mode, args.device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M params | mode={args.mode}")
    print()

    # 1. Perplexity on WikiText-2
    print("=" * 50)
    print("WikiText-2 Perplexity")
    print("=" * 50)
    nll, ppl, n_tokens = eval_perplexity(
        model, tokenizer, args.seq_len, args.num_samples, args.device)
    print(f"  Loss: {nll:.4f} | PPL: {ppl:.2f} | Tokens: {n_tokens}")
    print()

    # 2. LAMBADA accuracy
    print("=" * 50)
    print("LAMBADA (last word prediction)")
    print("=" * 50)
    acc, correct, total = eval_lambada(model, tokenizer, args.device)
    print(f"  Accuracy: {acc:.4f} ({correct}/{total})")
    print()

    # 3. HellaSwag
    print("=" * 50)
    print("HellaSwag (commonsense)")
    print("=" * 50)
    acc_hs, correct_hs, total_hs = eval_hellaswag(model, tokenizer, args.device)
    print(f"  Accuracy: {acc_hs:.4f} ({correct_hs}/{total_hs})")
    print()

    # Summary
    print("=" * 50)
    print(f"SUMMARY ({args.mode})")
    print("=" * 50)
    print(f"  WikiText-2 PPL: {ppl:.2f}")
    print(f"  LAMBADA Acc:    {acc:.4f}")
    print(f"  HellaSwag Acc:  {acc_hs:.4f}")


if __name__ == "__main__":
    main()
