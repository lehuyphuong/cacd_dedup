# cacd-dedup

**Cross-Attention Calibrated Deduplication (CACD)** — một phương pháp chunk-filtering mới cho RAG, thay thế cosine-similarity threshold cố định bằng cross-encoder + calibrated probability.

Project độc lập, tách riêng khỏi `rag-bench-v4`. Giữ nguyên 5 chunking strategies để đánh giá, nhưng thay toàn bộ 5 filter methods cũ (NoFilter / ExactNorm / MinHashLSH / Similarity / NERExact) bằng **một pipeline duy nhất: CACD**.

---

## 1. Vấn đề CACD giải quyết

Cosine similarity trên pooled embedding chỉ cho ra **1 con số duy nhất** để quyết định 2 chunk có trùng lặp hay không, so với **1 threshold cố định** (thường là 0.8) không có cơ sở khoa học rõ ràng. Thực nghiệm trên `rag-bench-v4` (500 SQuAD documents) cho thấy hệ quả cụ thể:

| Filter cũ | Strategy bị ảnh hưởng | Recall delta |
|---|---|---|
| NERExact | Contextual | **-0.143** |
| NERExact | TopicBased | **-0.131** |

Nguyên nhân: chunk có header/entity giống nhau bề ngoài (do cấu trúc chunking) bị filter nhầm coi là "trùng lặp", dù nội dung thực tế khác nhau — gọi là **false redundancy**.

CACD giải quyết bằng cách:
1. Dùng **cross-encoder** (joint encoding, không phải 2 vector độc lập) để giữ chi tiết token-to-token tới tận bước so sánh cuối.
2. Trích xuất **attention matrix** thay vì 1 con số similarity duy nhất — biết chính xác phần nào của 2 chunk thực sự "khớp" với nhau.
3. Output một **calibrated probability** (không phải threshold đoán mò) — quyết định dựa trên Bayes-optimal cutoff suy ra từ tỷ lệ chi phí, không phải hằng số tùy tiện.

---

## 2. Pipeline CACD (4 stages)

```
Chunk mới
    │
    ▼
Stage 0 — Embedding
    all-MiniLM-L6-v2, vector 384 chiều
    │
    ▼
Stage 1 — Coarse retrieval (HNSW, batch query)
    Tìm top-K ứng viên gần nhất ĐÃ TỒN TẠI trong Qdrant
    (persistent index — không reset giữa các lần ingest)
    Độ phức tạp: O(m log n)
    │
    ▼  (chỉ K ứng viên, K=5 mặc định)
Stage 2 — Cross-attention redundancy scoring
    2a. Joint encoding: [CLS] chunk_mới [SEP] candidate [SEP]
    2b. Trích xuất attention matrix (layer cuối, average qua head)
    2c. Tổng hợp redundancy signal (max-alignment kiểu BERTScore)
    2d. Output calibrated probability P(duplicate)
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (pretrained, không fine-tune)
    │
    ▼
Stage 3 — Quyết định threshold-free (CHỈ nhánh DROP)
    cutoff = Bayes-optimal, suy ra từ cost_FP / (cost_FP + cost_FN)
    P(duplicate) > cutoff  →  Bỏ qua, không insert
    Ngược lại               →  Insert vào Qdrant
```

**Lưu ý phạm vi**: Bản triển khai này **chỉ có nhánh Drop**, không có Merge. Việc so sánh Drop vs Merge (đánh giá trade-off tốc độ vs độ chính xác) để dành cho experiment sau.

---

## 3. Chunking Strategies (giữ nguyên từ rag-bench-v4)

| Strategy | Configs | Nguyên lý |
|---|---|---|
| AdaptiveEntropy | size=300, size=500 | Chunk size co giãn theo Shannon entropy |
| AdaptiveSentenceLen | target=4, target=6 câu | Chunk size co giãn theo độ dài câu trung bình |
| HierarchicalParentChild | child=200/parent=600, child=400/parent=800 | 2 tầng parent+child, cả 2 đều index |
| Contextual | size=300, size=500 | Prepend header `[Context: title | Part i/n]` |
| TopicBased | n_topics=4, n_topics=6 | K-means clustering trên sentence embeddings |

→ **10 configs tổng** (5 strategies × 2 size variants), mỗi config chạy qua đúng 1 pipeline CACD.

---

## 4. Evaluation Metrics (giữ nguyên từ rag-bench-v4)

| Metric | Công thức |
|---|---|
| Precision | \|Te ∩ Tr\| / \|Tr\| |
| Recall | \|Te ∩ Tr\| / \|Te\| |
| IoU | \|Te ∩ Tr\| / \|Te ∪ Tr\| |
| Index Size | chunk_count_after_filter + storage_mb |

Tính trên cả 2 chế độ tokenization: `raw` (lowercase) và `preprocessed` (bỏ stopword + lemmatize).

---

## 5. Cài đặt và chạy

### 5.1 Setup

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Lần chạy đầu tiên sẽ tự động tải về:
- `sentence-transformers/all-MiniLM-L6-v2` (~80MB) — Stage 0
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90MB) — Stage 2

### 5.2 Verify cài đặt

```bash
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('embed OK:', m.encode(['test']).shape)"
python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; m = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('cross-encoder OK')"
python -c "from qdrant_client import QdrantClient; c = QdrantClient(':memory:'); print('qdrant OK')"
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### 5.3 Chạy benchmark

```bash
# Debug nhỏ (20 docs, 30 questions)
python scripts/benchmark.py --max-docs 20 --max-questions 30

# Một strategy cụ thể — đề xuất bắt đầu với Contextual (case "khó nhất"
# nơi NERExact baseline cũ bị phá hỏng nặng nhất, -0.143 Recall)
python scripts/benchmark.py --strategy Contextual --max-docs 50 --max-questions 50

# Một config cụ thể
python scripts/benchmark.py --config "Contextual_300_0"

# Full benchmark (toàn bộ 10 configs)
python scripts/benchmark.py

# Background run
nohup python scripts/benchmark.py > results/bench.log 2>&1 &
tail -f results/bench.log
```

### 5.4 Xem kết quả

```bash
# Xem CSV kết quả
column -t -s',' results/benchmark_results.csv | less -S

# Xem audit log (quyết định drop/keep cho từng chunk, kèm P(duplicate))
column -t -s',' results/audit_Contextual_300_0.csv | less -S

# Xem heatmap attention (phân tích trực quan đoạn nào trùng)
open results/heatmaps/Contextual_300_0/   # macOS
xdg-open results/heatmaps/Contextual_300_0/  # Linux
```

---

## 6. Output

### `results/benchmark_results.csv` — 1 row / config

| Column | Mô tả |
|---|---|
| `config_name` | vd. `Contextual_300_0` |
| `chunk_count_before_filter` / `chunk_count_after_filter` | Số chunk trước/sau CACD dedup |
| `filter_reduction_pct` | % chunk bị drop |
| `ingest_time_s` | Tổng thời gian chunk + CACD (4 stages) + upsert |
| `storage_mb` / `storage_du` | Dung lượng collection Qdrant |
| `precision_raw` / `recall_raw` / `iou_raw` | Token metrics, raw mode |
| `precision_pre` / `recall_pre` / `iou_pre` | Token metrics, preprocessed mode |
| `cacd_cutoff_used` | Bayes-optimal cutoff thực tế đã dùng |

### `results/audit_{config_name}.csv` — 1 row / chunk

Ghi lại quyết định CACD cho từng chunk: `decision` (drop/keep), `best_p_duplicate`, `best_candidate_id`, `coverage_a_to_b`, `coverage_b_to_a`, `attn_entropy`.

### `results/heatmaps/{config_name}/*.png`

Heatmap attention matrix giữa chunk mới và candidate gần nhất — trục X là candidate (B), trục Y là chunk mới (A), màu càng đậm = attention weight càng cao. Tối đa 30 heatmap/config (giới hạn `max_heatmaps` trong `stage3_decision.py`).

---

## 7. Cấu trúc project

```
cacd-dedup/
├── configs/
│   └── settings.py              # Tham số CACD + 10 chunking configs
├── src/
│   ├── ingestion/
│   │   ├── loader.py            # SQuAD 1.1 loader
│   │   ├── chunker.py           # 5 chunking strategies (giữ nguyên từ v4)
│   │   ├── embedder.py          # all-MiniLM-L6-v2 (Stage 0)
│   │   └── vector_store.py      # Qdrant embedded — persistent index
│   ├── dedup/                   # ★ Module mới — thay thế filtering/
│   │   ├── stage1_coarse_retrieval.py   # Batch HNSW query
│   │   ├── stage2_cross_attention.py    # Cross-encoder + attention extraction
│   │   ├── calibration.py               # Calibrated probability + Bayes cutoff
│   │   └── stage3_decision.py           # Drop/keep decision + heatmap
│   ├── retrieval/
│   │   └── retriever.py         # Dense cosine retrieval (đánh giá cuối)
│   ├── evaluation/
│   │   └── metrics.py           # Precision/Recall/IoU
│   └── utils/
│       └── logger.py
├── scripts/
│   └── benchmark.py             # CLI chính
├── data/
│   └── qdrant_storage/          # Qdrant collections
├── results/
│   ├── heatmaps/                # Attention heatmap PNG theo từng config
│   ├── benchmark_results.csv
│   ├── per_question_*.csv
│   └── audit_*.csv
├── requirements.txt
└── README.md
```

---

## 8. Giới hạn của bản triển khai hiện tại

- **Chỉ có nhánh Drop**, chưa có Merge. Trường hợp "partial overlap" (2 chunk trùng 50% nội dung, mỗi bên còn 50% thông tin riêng) sẽ bị xử lý nhị phân (drop 1, giữ 1) — có rủi ro mất thông tin riêng của chunk bị drop. Đây là giới hạn đã biết, để dành cho experiment Merge sau.
- **Cross-encoder pretrained, không fine-tune** — `ms-marco-MiniLM-L-6-v2` được train cho passage relevance ranking (MS MARCO), không phải binary duplicate classification. Calibration hiện tại dùng z-score + sigmoid trên phân phối logit quan sát được (`RunningLogitCalibrator`), không phải calibration đã được chứng minh chính xác tuyệt đối (cần dữ liệu có nhãn để calibrate chuẩn).
- **Stage 1 không dùng GPU song song hóa thật** trong môi trường hiện tại — kiến trúc batch-query vẫn giữ nguyên, có thể nâng cấp lên cuVS/FAISS-GPU sau mà không đổi logic.
- **Stage 2 xử lý tuần tự từng chunk** (không batch cross-encoder calls) — đây là điểm có thể tối ưu thêm nếu cần tăng tốc.
