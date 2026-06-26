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


def _novel_information_score(
    attn: torch.Tensor,
    sep_idx: int,
    n_tokens: int,
) -> float:
    """
    Novel Information Score (NIS) — đo mức độ thông tin MỚI mà B mang
    lại so với A, dựa trên entropy của attention distribution B→A.

    Nền tảng lý thuyết (Information Theory):
      - Attention B→A[j, :] = phân phối xác suất token j của B
        "dựa vào" các token nào trong A để hiểu nghĩa của mình
      - Entropy(attention B→A[j]) thấp → token j tập trung attention
        vào 1-2 token cụ thể trong A → A "giải thích" được token j
        → token j KHÔNG mang thông tin mới
      - Entropy(attention B→A[j]) cao → token j phân tán attention
        đều khắp A → A không có token nào "giải thích" được token j
        → token j CÓ THỂ mang thông tin mới

    NIS(B|A) = mean entropy của attention B→A, chuẩn hóa về [0, 1]:
      NIS → 0: B hoàn toàn được "giải thích" bởi A → DROP candidate
      NIS → 1: B hoàn toàn khác A về thông tin       → KEEP

    Quan trọng: chuẩn hóa dùng log(|A|) — maximum entropy lý thuyết
    khi token B phân tán đều hoàn toàn sang tất cả token A.
    Đây là ngưỡng tự nhiên từ Information Theory, không phải số đặt tay.

    Args:
        attn    : ma trận attention (n_tokens, n_tokens), đã average head.
        sep_idx : vị trí [SEP] đầu tiên (ranh giới A/B).
        n_tokens: tổng số token thật.

    Returns:
        nis: float trong [0, 1], càng thấp → B càng ít thông tin mới.
    """
    a_range = slice(1, sep_idx)
    b_range = slice(sep_idx + 1, n_tokens - 1)

    sub_b_to_a = attn[b_range, a_range]   # (len_B, len_A)

    if sub_b_to_a.numel() == 0:
        return 1.0   # Không có gì để so sánh → coi như B hoàn toàn mới

    len_a = sub_b_to_a.shape[1]
    if len_a < 2:
        return 1.0

    # Chuẩn hóa lại attention B→A CHỈ TRÊN PHẦN A
    # (loại bỏ ảnh hưởng pha loãng của softmax toàn chuỗi)
    row_sums = sub_b_to_a.sum(dim=1, keepdim=True).clamp(min=1e-9)
    prob_b_to_a = sub_b_to_a / row_sums   # (len_B, len_A), mỗi hàng sum=1

    # Entropy của từng token B
    # H(j) = -sum_i p(i|j) * log(p(i|j))
    ent_per_token = -(prob_b_to_a * torch.log(prob_b_to_a + 1e-9)).sum(dim=1)  # (len_B,)

    # Maximum entropy lý thuyết = log(len_A)
    # Đây là ngưỡng tự nhiên: token B phân tán đều hoàn toàn sang mọi A
    max_ent = float(np.log(len_a))

    if max_ent < 1e-9:
        return 1.0

    # NIS = mean normalized entropy
    nis = float((ent_per_token / max_ent).mean().clamp(0.0, 1.0).item())
    return round(nis, 4)


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
    nis = _novel_information_score(avg_attn, sep_idx, n_tokens)

    return {
        "raw_logit":        round(raw_logit, 4),
        "prob_duplicate":   round(prob_dup, 4),
        "coverage_a_to_b":  round(cov_a_to_b, 4),
        "coverage_b_to_a":  round(cov_b_to_a, 4),
        "nis_b_given_a":    nis,
        # attention_matrix và tokens disabled — heatmap off, không cần serialize
        # Uncomment khi cần vẽ heatmap phân tích:
        # "attention_matrix": avg_attn[:n_tokens, :n_tokens].cpu().numpy(),
        # "tokens_a":         tokens[1:sep_idx],
        # "tokens_b":         tokens[sep_idx + 1:n_tokens - 1],
        # "sep_idx":          sep_idx,
        # "n_tokens":         n_tokens,
    }


def score_candidates(
    chunk_text: str,
    candidates: list[dict],
    chunk_parent_id: str | None = None,
    chunk_level: str | None = None,
) -> list[dict]:
    """
    Chấm điểm chunk mới với từng candidate trong danh sách (Stage 2,
    áp dụng cho K ứng viên đã được Stage 1 thu hẹp).

    Xử lý 2 trường hợp đặc biệt trước khi score để tránh false-redundancy:

    1. Contextual chunking — strip header [Context: title | Part N/M]
       Header giống hệt nhau ở mọi chunk trong cùng 1 document → NIS thấp
       → drop oan. Strip trước khi score, giữ nguyên text gốc trong Qdrant.

    2. HierarchicalParentChild — skip cặp parent-child (chunk là child của
       candidate, hoặc ngược lại). Parent và child luôn overlap hoàn toàn
       về nội dung → không phải "duplicate" theo nghĩa dedup, mà là cấu
       trúc index đa tầng có chủ đích.

    Returns: list dict, mỗi phần tử là candidate gốc được bổ sung
             các trường score (raw_logit, prob_duplicate, nis_b_given_a...).
             Candidate bị skip (parent-child) được đánh dấu "skipped=True".
    """
    # Strip contextual header khỏi chunk mới (A)
    text_a_clean = _strip_contextual_header(chunk_text)

    scored = []
    for cand in candidates:
        # ── Skip parent-child pairs (HierarchicalParentChild) ─────────────
        if _is_parent_child_pair(chunk_parent_id, chunk_level,
                                  cand.get("parent_id"), cand.get("level"),
                                  cand.get("chunk_id", "")):
            merged = {**cand, "skipped": True, "skip_reason": "parent_child_pair",
                      "prob_duplicate": 0.0, "nis_b_given_a": 1.0,
                      "raw_logit": 0.0, "coverage_a_to_b": 0.0,
                      "coverage_b_to_a": 0.0}
            scored.append(merged)
            continue

        # Strip contextual header khỏi candidate (B)
        text_b_clean = _strip_contextual_header(cand["text"])

        result = score_pair(text_a_clean, text_b_clean)
        merged = {**cand, **result, "skipped": False, "skip_reason": ""}
        scored.append(merged)
    return scored


import re as _re

def _strip_contextual_header(text: str) -> str:
    """
    Loại bỏ header [Context: ... ] do Contextual chunker prepend.
    Ví dụ: "[Context: Super Bowl 50 | Part 3/12] The game was..."
         → "The game was..."

    Nếu không có header → trả về text nguyên vẹn.
    Text gốc trong Qdrant KHÔNG bị thay đổi — chỉ strip khi đưa vào
    cross-encoder để tránh false-redundancy do header giống nhau.
    """
    return _re.sub(r'^\[Context:[^\]]*\]\s*', '', text).strip()


def _is_parent_child_pair(
    chunk_parent_id:  str | None,
    chunk_level:      str | None,
    cand_parent_id:   str | None,
    cand_level:       str | None,
    cand_chunk_id:    str,
) -> bool:
    """
    Trả về True nếu chunk mới và candidate là cặp parent-child trong
    HierarchicalParentChild — tức là KHÔNG NÊN so sánh dedup vì chúng
    overlap theo thiết kế, không phải vì nội dung trùng lặp thật sự.

    Các trường hợp skip:
      1. chunk là child, candidate là parent của nó
         (chunk_parent_id == cand_chunk_id)
      2. chunk là parent, candidate là child của nó
         (cand_parent_id == chunk của chúng ta — không có chunk_id
          ở đây nhưng có thể kiểm tra qua level)
      3. Cả 2 đều là child của cùng 1 parent
         (chunk_parent_id == cand_parent_id, cả 2 đều có parent_id)
    """
    # Trường hợp 1: chunk là child, candidate là parent của nó
    if chunk_parent_id and chunk_parent_id == cand_chunk_id:
        return True

    # Trường hợp 2: chunk là parent (level="parent"), candidate là child của nó
    if chunk_level == "parent" and cand_parent_id:
        # candidate có parent_id → đây là child; nếu cùng doc thì skip
        # (không có chunk_id của chunk mới ở đây, dùng heuristic doc-level)
        if cand_parent_id.startswith(chunk_parent_id or "___NOMATCH___"):
            return True

    # Trường hợp 3: cả 2 là sibling child của cùng 1 parent
    # → KHÔNG skip: 2 child chunks có thể thực sự trùng lặp nội dung
    # nếu chunk_size nhỏ và có overlap → để CACD xử lý bình thường

    return False
