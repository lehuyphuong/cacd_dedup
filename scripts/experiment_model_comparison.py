"""
scripts/experiment_model_comparison.py

Compares 37 cross-encoder models on 5 chunk pairs with varying similarity.
Each model runs independently; outputs a heatmap and score table per model.

Run from the project root:
  python scripts/experiment_model_comparison.py

Output:
  results/model_comparison/{model_slug}/case1_identical.png
  results/model_comparison/{model_slug}/case2_paraphrase_full.png
  ...
  results/model_comparison/summary.csv   -- aggregated table across all models
"""

from __future__ import annotations

import csv
import math
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("results/model_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "cross-encoder/ettin-reranker-1b-v1",
    "cross-encoder/ettin-reranker-400m-v1",
    "cross-encoder/ettin-reranker-150m-v1",
    "cross-encoder/ettin-reranker-68m-v1",
    "cross-encoder/ettin-reranker-32m-v1",
    "cross-encoder/ettin-reranker-17m-v1",
    "cross-encoder/ms-marco-TinyBERT-L2-v2",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
    "cross-encoder/ms-marco-MiniLM-L2-v2",
    "cross-encoder/ms-marco-MiniLM-L4-v2",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-TinyBERT-L2",
    "cross-encoder/ms-marco-TinyBERT-L4",
    "cross-encoder/ms-marco-TinyBERT-L6",
    "cross-encoder/ms-marco-electra-base",
    "cross-encoder/stsb-roberta-large",
    "cross-encoder/stsb-distilroberta-base",
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/monoelectra-large",
    "cross-encoder/monoelectra-base",
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    "cross-encoder/msmarco-MiniLM-L6-en-de-v1",
    "cross-encoder/nli-MiniLM2-L6-H768",
    "cross-encoder/nli-deberta-base",
    "cross-encoder/msmarco-MiniLM-L12-en-de-v1",
    "cross-encoder/nli-deberta-v3-large",
    "cross-encoder/nli-deberta-v3-base",
    "cross-encoder/nli-deberta-v3-xsmall",
    "cross-encoder/nli-distilroberta-base",
    "cross-encoder/nli-roberta-base",
    "cross-encoder/qnli-distilroberta-base",
    "cross-encoder/qnli-electra-base",
    "cross-encoder/quora-distilroberta-base",
    "cross-encoder/quora-roberta-base",
    "cross-encoder/quora-roberta-large",
    "cross-encoder/stsb-TinyBERT-L4",
    "cross-encoder/stsb-roberta-base",
]

# ── 5 experimental chunk pairs ────────────────────────────────────────────────

PAIRS = {
    "case1_identical": {
        "label": "Case 1 — identical sentences (100% identical)",
        "a": "The Eiffel Tower was built in 1889 in Paris, France.",
        "b": "The Eiffel Tower was built in 1889 in Paris, France.",
        "expected": "DROP",
    },
    "case2_paraphrase_full": {
        "label": "Case 2 — same meaning, different wording (100% paraphrase)",
        "a": "The Eiffel Tower was built in 1889 in Paris, France.",
        "b": "France's iconic Eiffel Tower was constructed in the year 1889 in the city of Paris.",
        "expected": "DROP",
    },
    "case3_identical_50pct": {
        "label": "Case 3 — 50% identical (shared first half, different second half)",
        "a": "The Eiffel Tower was built in 1889 in Paris, France. It attracts millions of tourists every year.",
        "b": "The Eiffel Tower was built in 1889 in Paris, France. It was designed by the engineer Gustave Eiffel.",
        "expected": "KEEP",
    },
    "case4_paraphrase_50pct": {
        "label": "Case 4 — ~50% overlap with paraphrase + new information",
        "a": "The Eiffel Tower was built in 1889 in Paris, France. It attracts millions of tourists every year.",
        "b": "France's iconic tower was constructed in 1889 in Paris. It was designed by the engineer Gustave Eiffel.",
        "expected": "KEEP",
    },
    "case5_different": {
        "label": "Case 5 — completely different sentences",
        "a": "The Eiffel Tower was built in 1889 in Paris, France.",
        "b": "The Amazon rainforest covers over 5.5 million square kilometers in South America.",
        "expected": "KEEP",
    },
}

# Ground-truth: case1 & case2 => DROP, case3-5 => KEEP
EXPECTED = {k: v["expected"] for k, v in PAIRS.items()}

# ── Scoring ───────────────────────────────────────────────────────────────────

def score_pair(text_a: str, text_b: str, tokenizer, model) -> dict:
    inputs = tokenizer(
        text_a, text_b,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    raw_logit = outputs.logits.squeeze()
    # Handle both binary models (scalar logit) and multi-class models
    # (NLI: 3 classes — contradiction / neutral / entailment)
    if raw_logit.dim() == 0:
        scalar_logit = raw_logit.item()
        prob_dup = float(torch.sigmoid(raw_logit).item())
    else:
        # Last class is typically entailment / match / duplicate
        scalar_logit = raw_logit[-1].item()
        probs = torch.softmax(raw_logit, dim=-1)
        prob_dup = float(probs[-1].item())

    cov_a2b, cov_b2a, entropy_val = 0.0, 0.0, 1.0
    attn_matrix = None
    tokens_a_list, tokens_b_list = [], []
    sep_idx_val, n_tokens_val = 0, 0

    if outputs.attentions is not None:
        last_attn = outputs.attentions[-1][0]
        avg_attn  = last_attn.mean(dim=0)

        input_ids    = inputs["input_ids"][0]
        all_tokens   = tokenizer.convert_ids_to_tokens(input_ids)
        n_tokens_val = int(inputs["attention_mask"][0].sum().item())

        sep_id        = tokenizer.sep_token_id
        sep_positions = (input_ids == sep_id).nonzero(as_tuple=True)[0]
        sep_idx_val   = int(sep_positions[0].item()) if len(sep_positions) > 0 else n_tokens_val // 2

        a_range = slice(1, sep_idx_val)
        b_range = slice(sep_idx_val + 1, n_tokens_val - 1)
        sub_a2b = avg_attn[a_range, b_range]
        sub_b2a = avg_attn[b_range, a_range]

        if sub_a2b.numel() > 0:
            cov_a2b = sub_a2b.max(dim=1).values.mean().item()
        if sub_b2a.numel() > 0:
            cov_b2a = sub_b2a.max(dim=1).values.mean().item()
        if sub_a2b.numel() > 0:
            probs_ent   = sub_a2b / (sub_a2b.sum(dim=1, keepdim=True) + 1e-9)
            ent         = -(probs_ent * torch.log(probs_ent + 1e-9)).sum(dim=1)
            max_ent     = math.log(max(sub_a2b.shape[1], 2))
            entropy_val = float((ent / max_ent).mean().item())

        attn_matrix   = avg_attn[:n_tokens_val, :n_tokens_val].cpu().numpy()
        tokens_a_list = all_tokens[1:sep_idx_val]
        tokens_b_list = all_tokens[sep_idx_val + 1:n_tokens_val - 1]

    return {
        "raw_logit":         round(scalar_logit, 4),
        "prob_duplicate":    round(prob_dup, 4),
        "coverage_a_to_b":   round(cov_a2b, 4),
        "coverage_b_to_a":   round(cov_b2a, 4),
        "redundancy_signal": round(min(cov_a2b, cov_b2a), 4),
        "attn_entropy":      round(entropy_val, 4),
        "attention_matrix":  attn_matrix,
        "tokens_a":          tokens_a_list,
        "tokens_b":          tokens_b_list,
        "sep_idx":           sep_idx_val,
        "n_tokens":          n_tokens_val,
    }


def decide(result: dict) -> str:
    """
    Decision based on prob_duplicate with natural threshold at 0.5.
    Binary models (quora, stsb): sigmoid(logit) > 0.5 => DROP.
    NLI 3-class models: softmax[-1] > 0.5 => DROP.
    """
    if result["prob_duplicate"] > 0.5:
        return "DROP"
    return "KEEP"


# ── Heatmap ───────────────────────────────────────────────────────────────────

def plot_heatmap(result: dict, case_key: str, case_label: str,
                 decision: str, expected: str, model_slug: str,
                 model_dir: Path) -> str:

    attn     = result["attention_matrix"]
    sep_idx  = result["sep_idx"]
    n_tokens = result["n_tokens"]

    if attn is None or sep_idx == 0:
        return ""

    a_range  = slice(1, sep_idx)
    b_range  = slice(sep_idx + 1, n_tokens - 1)
    sub_attn = attn[a_range, b_range]
    tokens_a = result["tokens_a"]
    tokens_b = result["tokens_b"]

    if sub_attn.size == 0 or len(tokens_a) == 0 or len(tokens_b) == 0:
        return ""

    fig_w = max(6, min(0.45 * len(tokens_b), 22))
    fig_h = max(4, min(0.45 * len(tokens_a), 16))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(sub_attn, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=sub_attn.max())
    ax.set_xticks(range(len(tokens_b)))
    ax.set_xticklabels(tokens_b, rotation=90, fontsize=7)
    ax.set_yticks(range(len(tokens_a)))
    ax.set_yticklabels(tokens_a, fontsize=7)
    ax.set_xlabel("Candidate (B)", fontsize=9)
    ax.set_ylabel("New chunk (A)", fontsize=9)

    correct = "pass" if decision == expected else "fail"
    title = (
        f"{case_label}\n"
        f"model={model_slug}\n"
        f"logit={result['raw_logit']:.3f}  prob_dup={result['prob_duplicate']:.3f}  "
        f"cov(A=>B)={result['coverage_a_to_b']:.4f}  cov(B=>A)={result['coverage_b_to_a']:.4f}\n"
        f"decision={decision}  expected={expected}  {correct}"
    )
    ax.set_title(title, fontsize=7, pad=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="attention weight")
    fig.tight_layout()

    out_path = model_dir / f"{case_key}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return str(out_path)


# ── Per-model runner ──────────────────────────────────────────────────────────

def run_model(model_name: str, summary_rows: list) -> None:
    slug = model_name.replace("/", "__").replace("-", "_")
    model_dir = OUT_DIR / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    try:
        t_load    = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_attentions=True
        )
        model.to(DEVICE)
        model.eval()
        load_time = round(time.perf_counter() - t_load, 1)
        n_params  = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Loaded in {load_time}s | params={n_params:.0f}M | device={DEVICE}")
    except Exception as e:
        print(f"  [SKIP] Failed to load: {e}")
        for case_key in PAIRS:
            summary_rows.append({
                "model": model_name, "case": case_key,
                "status": "LOAD_ERROR", "raw_logit": "", "prob_dup": "",
                "cov_a2b": "", "cov_b2a": "", "redundancy": "",
                "decision": "", "expected": EXPECTED[case_key], "correct": "",
            })
        return

    n_correct = 0
    for case_key, pair in PAIRS.items():
        try:
            result   = score_pair(pair["a"], pair["b"], tokenizer, model)
            decision = decide(result)
            expected = EXPECTED[case_key]
            correct  = decision == expected

            if correct:
                n_correct += 1

            print(
                f"  {case_key:<30} logit={result['raw_logit']:>8.3f}  "
                f"prob_dup={result['prob_duplicate']:.3f}  "
                f"redund={result['redundancy_signal']:.4f}  "
                f"=> {decision:<4}  ({'pass' if correct else 'fail'} expected {expected})"
            )

            plot_heatmap(result, case_key, pair["label"],
                         decision, expected, slug, model_dir)

            summary_rows.append({
                "model":      model_name,
                "case":       case_key,
                "status":     "OK",
                "raw_logit":  result["raw_logit"],
                "prob_dup":   result["prob_duplicate"],
                "cov_a2b":    result["coverage_a_to_b"],
                "cov_b2a":    result["coverage_b_to_a"],
                "redundancy": result["redundancy_signal"],
                "decision":   decision,
                "expected":   expected,
                "correct":    "1" if correct else "0",
            })
        except Exception as e:
            print(f"  [ERROR] {case_key}: {e}")
            summary_rows.append({
                "model": model_name, "case": case_key,
                "status": "SCORE_ERROR", "raw_logit": "", "prob_dup": "",
                "cov_a2b": "", "cov_b2a": "", "redundancy": "",
                "decision": "", "expected": EXPECTED[case_key], "correct": "0",
            })

    print(f"  => Accuracy: {n_correct}/5 correct")

    # Release VRAM/RAM before loading the next model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("CACD — Model Comparison Experiment")
    print(f"Device: {DEVICE} | Models: {len(MODELS)} | Cases: {len(PAIRS)}")
    print(f"Output: {OUT_DIR.resolve()}")

    summary_rows: list[dict] = []

    for i, model_name in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}]", end="")
        run_model(model_name, summary_rows)

    # Write summary CSV
    csv_path = OUT_DIR / "summary.csv"
    fields = ["model", "case", "status", "raw_logit", "prob_dup",
              "cov_a2b", "cov_b2a", "redundancy", "decision", "expected", "correct"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n{'='*70}")
    print("FINAL SUMMARY — Accuracy per model (5 cases, expected: case1&2=DROP, case3-5=KEEP)")
    print(f"{'='*70}")
    print(f"{'Model':<55} {'Acc':>5}")
    print("-" * 62)

    model_acc: dict[str, list] = {}
    for row in summary_rows:
        model_acc.setdefault(row["model"], [])
        if row["correct"] in ("0", "1"):
            model_acc[row["model"]].append(int(row["correct"]))

    for model_name, scores in model_acc.items():
        bar = "X" * sum(scores) + "." * (5 - sum(scores))
        print(f"  {model_name:<53} {bar}  {sum(scores)}/5")

    print(f"\nSummary CSV: {csv_path}")
    print(f"Heatmaps:    {OUT_DIR}/")


if __name__ == "__main__":
    main()
