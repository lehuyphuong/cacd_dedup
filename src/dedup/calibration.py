"""
CACD Stage 2d — Calibration: raw cross-encoder logit → calibrated
probability P(duplicate | attention_pattern).

Vì model cross-encoder dùng pretrained (ms-marco-MiniLM-L-6-v2) và
KHÔNG fine-tune cho bài toán dedup cụ thể (quyết định của user), raw
logit của nó không tự động là một xác suất hợp lệ — nó được train
cho mục đích relevance ranking (MS MARCO passage ranking), không
phải binary duplicate classification.

Để vẫn giữ đúng tinh thần "threshold-free / calibrated probability"
(Section 2.4 trong báo cáo novel idea) mà không cần fine-tune, ta
dùng temperature scaling KHÔNG-THAM-SỐ-HỌC: nhiệt độ T được ước
lượng từ chính phân phối raw_logit quan sát được trên dữ liệu đang
chạy (không cần nhãn duplicate/not-duplicate có sẵn), theo nguyên
tắc Platt-scaling đơn giản hóa — chuẩn hóa logit về phân phối có
ý nghĩa xác suất thông qua sigmoid, với T = độ lệch chuẩn của batch
logit hiện tại.

Đây KHÔNG phải calibration đã được chứng minh chính xác tuyệt đối
(cần dữ liệu có nhãn để calibrate chuẩn — xem Desai & Durrett, EMNLP
2020), nhưng giải quyết đúng vấn đề thực dụng: loại bỏ việc áp một
threshold cố định tùy tiện (như cosine 0.8 trước đây) bằng một phép
biến đổi có cơ sở thống kê tối thiểu, dựa trên phân phối dữ liệu
thực tế thay vì một hằng số đoán mò.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RunningLogitCalibrator:
    """
    Calibrator chạy động (online), tự cập nhật theo phân phối raw_logit
    quan sát được qua quá trình ingest, dùng z-score + sigmoid để biến
    raw logit thành xác suất P(duplicate) có ý nghĩa tương đối trong
    chính phân phối dữ liệu hiện tại.

    Đây thay thế cho threshold cố định: thay vì so logit với 1 hằng số
    tuyệt đối, ta so nó với PHÂN PHỐI logit đã quan sát được — một
    cặp được coi là "khả nghi cao" nếu logit của nó nằm ở phần đuôi
    cao của phân phối, bất kể domain/document cụ thể có shift thang
    đo logit thế nào.
    """

    def __init__(self, min_samples: int = 30):
        self.min_samples = min_samples
        self._logits: list[float] = []

    def update(self, raw_logit: float) -> None:
        self._logits.append(raw_logit)

    def calibrated_probability(self, raw_logit: float) -> float:
        """
        P(duplicate) = sigmoid( z-score(raw_logit) )

        Nếu chưa đủ mẫu để ước lượng phân phối ổn định, fallback về
        sigmoid thô trên raw_logit (vẫn tốt hơn so sánh trực tiếp
        với 1 threshold tùy tiện, vì sigmoid bị chặn trong [0,1] và
        có diễn giải xác suất).
        """
        if len(self._logits) < self.min_samples:
            return float(1.0 / (1.0 + np.exp(-raw_logit)))

        arr = np.array(self._logits)
        mean = arr.mean()
        std = arr.std() + 1e-6
        z = (raw_logit - mean) / std
        return float(1.0 / (1.0 + np.exp(-z)))

    def stats(self) -> dict:
        if not self._logits:
            return {"n": 0, "mean": 0.0, "std": 0.0}
        arr = np.array(self._logits)
        return {
            "n":    len(arr),
            "mean": round(float(arr.mean()), 4),
            "std":  round(float(arr.std()), 4),
        }


def bayes_optimal_cutoff(cost_false_positive: float, cost_false_negative: float) -> float:
    """
    Stage 3 — suy ra cutoff từ tỷ lệ chi phí, KHÔNG phải số đoán mò.

    cost_false_positive : chi phí khi NHẦM coi 1 chunk là duplicate
                           rồi DROP nó (mất thông tin thật).
    cost_false_negative : chi phí khi NHẦM coi 1 chunk KHÔNG phải
                           duplicate rồi GIỮ nó (lãng phí index size).

    Công thức Bayes-optimal cho bài toán phân loại nhị phân với chi
    phí bất đối xứng (Elkan, 2001):
        cutoff = cost_false_positive / (cost_false_positive + cost_false_negative)

    Khi 2 chi phí bằng nhau → cutoff = 0.5 (trường hợp đối xứng mặc định).
    Khi việc DROP nhầm (mất thông tin) bị coi là tệ hơn nhiều so với
    việc giữ thừa 1 chunk (lãng phí storage) → cost_false_positive cao
    → cutoff cao hơn 0.5 → hệ thống THẬN TRỌNG hơn khi quyết định drop.
    """
    denom = cost_false_positive + cost_false_negative
    if denom <= 0:
        return 0.5
    return cost_false_positive / denom
