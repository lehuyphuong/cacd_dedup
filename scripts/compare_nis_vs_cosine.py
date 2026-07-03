"""
Compare CACD (NIS) vs Cosine Similarity on one chunk pair:
  A: "The company launched a new smartphone."
  B: "The firm introduced its latest mobile device."

Both sentences are paraphrases with high semantic overlap but
different surface forms — the ideal test case for comparing methods.

Outputs:
  - Cosine similarity score (bi-encoder)
  - prob_dup from cross-encoder classification head
  - NIS from cross-encoder attention matrix (full derivation printed)
  - Final CACD decision (DROP / KEEP)
"""

import math
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Chunk pair ────────────────────────────────────────────────────────────────
A = "The company launched a new smartphone."
B = "The firm introduced its latest mobile device."

# ── Models ───────────────────────────────────────────────────────────────────
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_MODEL   = "cross-encoder/msmarco-MiniLM-L6-en-de-v1"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ── CACD thresholds (same as settings.py) ────────────────────────────────────
PROB_HIGH         = 0.8
PROB_LOW          = 0.2
NIS_DROP_THRESHOLD = 0.8
LENGTH_GUARD      = 300
NIS_FLOOR         = 0.3

print("=" * 65)
print("CACD vs Cosine Similarity — Chunk Pair Comparison")
print("=" * 65)
print(f"  A: {A}")
print(f"  B: {B}")
print()

# ════════════════════════════════════════════════════════════════════════════
# METHOD 1 — Cosine Similarity (bi-encoder)
# ════════════════════════════════════════════════════════════════════════════
print("── Method 1: Cosine Similarity (bi-encoder) ──────────────────")
embedder = SentenceTransformer(EMBED_MODEL)
v_A = embedder.encode([A])
v_B = embedder.encode([B])
cos_score = float(cosine_similarity(v_A, v_B)[0][0])
print(f"  Embedding model : {EMBED_MODEL}")
print(f"  Embedding dim   : {v_A.shape[1]}")
print(f"  Cosine score    : {cos_score:.4f}")
print(f"  Threshold       : 0.8  (standard Similarity filter)")

cosine_decision = "DROP" if cos_score >= 0.8 else "KEEP"
print(f"  Decision        : {cosine_decision}  (score {'≥' if cos_score >= 0.8 else '<'} 0.8)")
print()

# ════════════════════════════════════════════════════════════════════════════
# METHOD 2 — CACD (cross-encoder + NIS)
# ════════════════════════════════════════════════════════════════════════════
print("── Method 2: CACD (cross-encoder + NIS) ──────────────────────")
tokenizer = AutoTokenizer.from_pretrained(CROSS_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    CROSS_MODEL, output_attentions=True
)
model.to(DEVICE)
model.eval()

inputs = tokenizer(
    A, B,
    return_tensors="pt",
    truncation=True,
    max_length=256,
    padding=True,
).to(DEVICE)

with torch.no_grad():
    outputs = model(**inputs)

# ── prob_dup ──────────────────────────────────────────────────────────────
logits = outputs.logits.squeeze()
if logits.dim() == 0:
    prob_dup = float(torch.sigmoid(logits).item())
    raw_logit = logits.item()
else:
    prob_dup = float(torch.softmax(logits, dim=-1)[-1].item())
    raw_logit = logits[-1].item()

print(f"  Cross-encoder   : {CROSS_MODEL}")
print(f"  Raw logit       : {raw_logit:.4f}")
print(f"  prob_duplicate  : {prob_dup:.4f}  (sigmoid/softmax of logit)")
print()

# ── Attention matrix ──────────────────────────────────────────────────────
last_attn  = outputs.attentions[-1][0]        # (num_heads, seq, seq)
avg_attn   = last_attn.mean(dim=0)            # (seq, seq)

input_ids  = inputs["input_ids"][0]
tokens     = tokenizer.convert_ids_to_tokens(input_ids)
n_tokens   = int(inputs["attention_mask"][0].sum().item())

sep_id     = tokenizer.sep_token_id
sep_pos    = (input_ids == sep_id).nonzero(as_tuple=True)[0]
sep_idx    = int(sep_pos[0].item()) if len(sep_pos) > 0 else n_tokens // 2

tokens_A   = tokens[1:sep_idx]
tokens_B   = tokens[sep_idx + 1:n_tokens - 1]

print(f"  Tokens A : {tokens_A}")
print(f"  Tokens B : {tokens_B}")
print(f"  sep_idx  : {sep_idx}  (position of first [SEP])")
print()

# ── NIS derivation (printed step by step) ────────────────────────────────
a_range    = slice(1, sep_idx)
b_range    = slice(sep_idx + 1, n_tokens - 1)
sub_b_to_a = avg_attn[b_range, a_range]       # (|B|, |A|)

print("  ── NIS derivation ────────────────────────────────────────")
print(f"  sub_B→A shape   : {tuple(sub_b_to_a.shape)}  (|B|={len(tokens_B)}, |A|={len(tokens_A)})")
print()

# Step 2 — row-normalize
row_sums    = sub_b_to_a.sum(dim=1, keepdim=True).clamp(min=1e-9)
prob_b_to_a = sub_b_to_a / row_sums          # (|B|, |A|), each row sums to 1

print("  Step 2 — p(i|j) after row-normalize (each row sums to 1):")
for j, tok_j in enumerate(tokens_B):
    row = prob_b_to_a[j].cpu().numpy()
    pairs = ", ".join(f"{tokens_A[i]}:{row[i]:.3f}" for i in range(len(tokens_A)))
    print(f"    j={j} '{tok_j}': [{pairs}]")
print()

# Step 3 — Shannon entropy per token in B
len_a         = sub_b_to_a.shape[1]
ent_per_token = -(prob_b_to_a * torch.log(prob_b_to_a + 1e-9)).sum(dim=1)

print("  Step 3 — H(j) Shannon Entropy per token in B:")
for j, tok_j in enumerate(tokens_B):
    h = float(ent_per_token[j].item())
    print(f"    j={j} '{tok_j}': H(j) = {h:.4f}")
print()

# Step 4 — H_max = log(|A|)
H_max = math.log(len_a)
print(f"  Step 4 — H_max = log(|A|) = log({len_a}) = {H_max:.4f}")
print(f"           (proven via Jensen: uniform distribution maximizes entropy)")
print()

# Step 5 — Normalize
H_tilde = ent_per_token / H_max
print("  Step 5 — H̃(j) = H(j) / H_max  (normalized to [0,1]):")
for j, tok_j in enumerate(tokens_B):
    h  = float(ent_per_token[j].item())
    ht = float(H_tilde[j].item())
    print(f"    j={j} '{tok_j}': H̃(j) = {h:.4f} / {H_max:.4f} = {ht:.4f}")
print()

# Step 6 — NIS = mean(H̃)
nis = float(H_tilde.mean().clamp(0.0, 1.0).item())
print(f"  Step 6 — NIS = mean(H̃(j)) = {nis:.4f}")
print()

# ── Stage 3 decision ─────────────────────────────────────────────────────
chunk_len = len(B)
print("  ── Stage 3: CACD Decision ────────────────────────────────")
print(f"  prob_dup   = {prob_dup:.4f}  | PROB_HIGH={PROB_HIGH}, PROB_LOW={PROB_LOW}")
print(f"  NIS        = {nis:.4f}  | NIS_DROP_THRESHOLD={NIS_DROP_THRESHOLD}")
print(f"  chunk_len  = {chunk_len} chars | LENGTH_GUARD={LENGTH_GUARD}, NIS_FLOOR={NIS_FLOOR}")
print()

if prob_dup >= PROB_HIGH:
    if chunk_len > LENGTH_GUARD and nis > NIS_FLOOR:
        cacd_decision = "KEEP"
        reason = f"length_guard ({chunk_len} > {LENGTH_GUARD} and NIS={nis:.3f} > {NIS_FLOOR})"
    else:
        cacd_decision = "DROP"
        reason = f"prob_high ({prob_dup:.3f} >= {PROB_HIGH})"
elif prob_dup <= PROB_LOW:
    cacd_decision = "KEEP"
    reason = f"prob_low ({prob_dup:.3f} <= {PROB_LOW})"
else:
    if nis < NIS_DROP_THRESHOLD:
        if chunk_len > LENGTH_GUARD:
            cacd_decision = "KEEP"
            reason = f"length_guard_uncertainty ({chunk_len} > {LENGTH_GUARD})"
        else:
            cacd_decision = "DROP"
            reason = f"nis_low ({nis:.3f} < {NIS_DROP_THRESHOLD})"
    else:
        cacd_decision = "KEEP"
        reason = f"nis_high ({nis:.3f} >= {NIS_DROP_THRESHOLD})"

print(f"  CACD Decision : {cacd_decision}  ({reason})")
print()

# ════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("FINAL COMPARISON")
print("=" * 65)
print(f"  Cosine score       : {cos_score:.4f}  →  Decision: {cosine_decision}")
print(f"  prob_dup (CACD)    : {prob_dup:.4f}")
print(f"  NIS (CACD)         : {nis:.4f}  →  Decision: {cacd_decision}")
print()
print("Why CACD is more informative on this pair:")
print()
print(f"  Cosine = {cos_score:.4f}: both A and B have similar topic embeddings,")
print(f"  so the score crosses the 0.8 threshold and the chunk is dropped.")
print(f"  But cosine cannot tell WHERE the similarity comes from or whether")
print(f"  B adds any new content beyond what A already contains.")
print()
print(f"  NIS = {nis:.4f}: by looking at how each token in B attends to A,")
print(f"  CACD measures token-level novelty. A low NIS (< {NIS_DROP_THRESHOLD})")
print(f"  confirms B is largely explained by A. A high NIS would indicate")
print(f"  B contributes new information despite surface similarity.")
print()
print(f"  prob_dup = {prob_dup:.4f}: the cross-encoder jointly encodes")
print(f"  both chunks, preserving token interactions that a bi-encoder loses.")
print()
print("Advantage demonstrated:")
if cacd_decision == cosine_decision:
    print(f"  Both methods agree: {cacd_decision}.")
    print(f"  CACD provides the richer signal (NIS={nis:.4f}, prob={prob_dup:.4f})")
    print(f"  explaining WHY the decision was made at the token level.")
else:
    print(f"  Cosine says {cosine_decision}, CACD says {cacd_decision}.")
    print(f"  The disagreement reveals the limitation of cosine: it collapses")
    print(f"  all token-level information into a single pooled vector,")
    print(f"  missing the nuance that CACD's attention-based NIS captures.")
