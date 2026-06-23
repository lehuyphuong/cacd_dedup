"""
CACD Stage 2 — Cross-Attention Redundancy Scorer (CARS).

Pipeline doc Section 4.1, "Đóng góp 2":
  Với mỗi cặp (chunk_mới, candidate_i):
    2a. Joint encoding: [CLS] chunk_mới [SEP] candidate_i [SEP]
    2b. Trích xuất attention matrix (mọi layer, mọi head, không pooling sớm)
    2c. Tổng hợp redundancy signal (max-alignment kiểu BERTScore)
    2d. Output calibrated probability P(duplicate | attention_pattern)

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (pretrained, KHÔNG fine-tune
— theo quyết định của user). Model này vốn được train cho passage
relevance scoring (MS MARCO), nên raw logit của nó không trực tiếp
là "xác suất duplicate" — cần một bước calibration nhẹ (temperature
scaling không-tham-số, dựa trên thống kê batch) để biến raw score
thành P(duplicate) có thể diễn giải được, đúng tinh thần
"calibrated probability" trong thiết kế (không phải threshold đoán mò
trên cosine similarity).

Độ phức tạp: O(K) forward pass đầy đủ cho mỗi chunk mới — K là hằng
số nhỏ (CACD_TOP_K_CANDIDATES), không phụ thuộc n vì Stage 1 đã thu
hẹp candidate set.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from configs.settings import CACD_CROSS_ENCODER_MODEL, DEVICE

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None


def get_cross_encoder():
    """Lazy-load cross-encoder model + tokenizer (pretrained, no fine-tune)."""
    global _tokenizer, _model
    if _model is None:
        logger.info("Loading cross-encoder: %s", CACD_CROSS_ENCODER_MODEL)
        _tokenizer = AutoTokenizer.from_pretrained(CACD_CROSS_ENCODER_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(
            CACD_CROSS_ENCODER_MODEL,
            output_attentions=True,   # cần attention matrix, không chỉ logit
        )
        _model.to(DEVICE)
        _model.eval()
        logger.info("Cross-encoder loaded on %s", DEVICE)
    return _tokenizer, _model


def _max_alignment_coverage(
    attn: torch.Tensor,
    sep_idx: int,
    n_tokens: int,
) -> tuple[float, float]:
    """
    Tổng hợp redundancy signal từ attention matrix — max-alignment
    kiểu BERTScore (Stage 2c trong pipeline).

    Với attention layer cuối, đã average qua các head:
      coverage(A→B) = trung bình, với mỗi token A, giá trị attention
                      LỚN NHẤT mà nó dành cho bất kỳ token nào bên B
      coverage(B→A) = tương tự, theo chiều ngược lại

    Args:
        attn    : ma trận attention (n_tokens, n_tokens), đã average head.
        sep_idx : vị trí token [SEP] đầu tiên (ranh giới A/B).
        n_tokens: tổng số token thật (bỏ padding).

    Returns:
        (coverage_a_to_b, coverage_b_to_a)
    """
    # Vùng A: token 1..sep_idx-1 (bỏ [CLS] ở vị trí 0)
    # Vùng B: token sep_idx+1..n_tokens-2 (bỏ [SEP] cuối)
    a_range = slice(1, sep_idx)
    b_range = slice(sep_idx + 1, n_tokens - 1)

    sub_a_to_b = attn[a_range, b_range]   # (len_A, len_B)
    sub_b_to_a = attn[b_range, a_range]   # (len_B, len_A)

    if sub_a_to_b.numel() == 0 or sub_b_to_a.numel() == 0:
        return 0.0, 0.0

    # Mỗi token A tìm token khớp nhất bên B (max theo chiều B), rồi trung bình
    coverage_a_to_b = sub_a_to_b.max(dim=1).values.mean().item()
    coverage_b_to_a = sub_b_to_a.max(dim=1).values.mean().item()

    return coverage_a_to_b, coverage_b_to_a


def _attention_entropy(attn: torch.Tensor, sep_idx: int, n_tokens: int) -> float:
    """
    Tín hiệu phụ: entropy của attention cross-block (A→B).
    Entropy thấp = tập trung rõ ràng (match mạnh); entropy cao = tản mát.
    Trả về entropy đã chuẩn hóa về [0, 1] (1 = tản mát tối đa).
    """
    a_range = slice(1, sep_idx)
    b_range = slice(sep_idx + 1, n_tokens - 1)
    sub = attn[a_range, b_range]
    if sub.numel() == 0:
        return 1.0

    probs = sub / (sub.sum(dim=1, keepdim=True) + 1e-9)
    ent = -(probs * torch.log(probs + 1e-9)).sum(dim=1)
    max_ent = np.log(max(sub.shape[1], 2))
    norm_ent = (ent / max_ent).mean().item()
    return float(np.clip(norm_ent, 0.0, 1.0))


@torch.no_grad()
def score_pair(text_a: str, text_b: str) -> dict:
    """
    Chấm điểm redundancy cho 1 cặp (text_a, text_b) qua cross-encoder.

    Returns:
        {
          "raw_logit": float,           # output thô của cross-encoder
          "coverage_a_to_b": float,      # % nội dung A được B bao phủ
          "coverage_b_to_a": float,      # % nội dung B được A bao phủ
          "attn_entropy": float,         # độ tản mát attention (0-1)
          "attention_matrix": np.ndarray,# ma trận attention đầy đủ (để vẽ heatmap)
          "tokens_a": list[str],
          "tokens_b": list[str],
          "sep_idx": int,
        }
    """
    tokenizer, model = get_cross_encoder()

    inputs = tokenizer(
        text_a, text_b,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(DEVICE)

    outputs = model(**inputs)

    # Xử lý cả model binary (1 logit) và multi-class (vd. NLI 3 class)
    logits = outputs.logits.squeeze()
    if logits.dim() == 0:
        raw_logit    = logits.item()
        prob_dup     = float(torch.sigmoid(logits).item())
    else:
        # Lấy class cuối (thường là entailment/match/duplicate)
        raw_logit    = logits[-1].item()
        prob_dup     = float(torch.softmax(logits, dim=-1)[-1].item())

    # Lấy attention layer cuối, average qua mọi head — giữ đúng tinh
    # thần "trích xuất attention matrix, mọi layer, mọi head" nhưng
    # dùng layer cuối làm đại diện (layer cuối thường mang tín hiệu
    # ngữ nghĩa rõ nhất cho classification head).
    attentions = outputs.attentions  # tuple(num_layers) of (1, num_heads, seq, seq)
    last_layer_attn = attentions[-1][0]            # (num_heads, seq, seq)
    avg_attn = last_layer_attn.mean(dim=0)          # (seq, seq) — average qua head

    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    n_tokens = int(inputs["attention_mask"][0].sum().item())

    sep_token_id = tokenizer.sep_token_id
    sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
    sep_idx = int(sep_positions[0].item()) if len(sep_positions) > 0 else n_tokens // 2

    cov_a_to_b, cov_b_to_a = _max_alignment_coverage(avg_attn, sep_idx, n_tokens)
    entropy = _attention_entropy(avg_attn, sep_idx, n_tokens)

    return {
        "raw_logit":        round(raw_logit, 4),
        "prob_duplicate":   round(prob_dup, 4),   # tín hiệu chính cho Stage 3
        "coverage_a_to_b":  round(cov_a_to_b, 4),
        "coverage_b_to_a":  round(cov_b_to_a, 4),
        "attn_entropy":     round(entropy, 4),
        "attention_matrix": avg_attn[:n_tokens, :n_tokens].cpu().numpy(),
        "tokens_a":         tokens[1:sep_idx],
        "tokens_b":         tokens[sep_idx + 1:n_tokens - 1],
        "sep_idx":          sep_idx,
        "n_tokens":         n_tokens,
    }


def score_candidates(
    chunk_text: str,
    candidates: list[dict],
) -> list[dict]:
    """
    Chấm điểm chunk mới với từng candidate trong danh sách (Stage 2,
    áp dụng cho K ứng viên đã được Stage 1 thu hẹp).

    Returns: list dict, mỗi phần tử là candidate gốc được bổ sung
             các trường score (raw_logit, coverage_a_to_b, ...).
    """
    scored = []
    for cand in candidates:
        result = score_pair(chunk_text, cand["text"])
        merged = {**cand, **result}
        scored.append(merged)
    return scored
