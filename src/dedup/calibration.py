"""
CACD Stage 2d — Calibration: redundancy signal (từ attention coverage)
→ calibrated probability P(duplicate | attention_pattern).

QUAN TRỌNG — lý do KHÔNG dùng raw_logit của cross-encoder trực tiếp:
ms-marco-MiniLM-L-6-v2 được train cho RELEVANCE RANKING (MS MARCO
passage ranking) — raw_logit cao nghĩa là "passage B liên quan tới
query A", KHÔNG phải "A và B là bản sao của nhau". Hai chunk thuộc
cùng một chủ đề (rất phổ biến trong cùng 1 document SQuAD) có thể có
raw_logit rất cao dù nội dung hoàn toàn khác nhau. Dùng raw_logit trực
tiếp làm tín hiệu dedup đã được quan sát thực nghiệm gây ra over-drop
nghiêm trọng (77-90% chunk bị drop oan trên debug run đầu tiên).

Tín hiệu đúng bản chất hơn — được tính trong stage3_decision.py từ
chính attention matrix (Stage 2b/2c, không cần thêm model khác):
    redundancy_signal = min(coverage_a_to_b, coverage_b_to_a)
Một cặp chỉ thực sự "trùng lặp" khi CẢ HAI chiều đều bao phủ nhau cao.

Calibrator ở đây biến redundancy_signal thô (thường nằm trong khoảng
nhỏ, 0.0-0.3 trên dữ liệu thực tế) thành một xác suất có ý nghĩa
TƯƠNG ĐỐI trong chính phân phối dữ liệu đang chạy, qua z-score +
sigmoid — không cần nhãn duplicate/not-duplicate có sẵn (online,
không-tham-số-học).

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
    Calibrator chạy động (online), tự cập nhật theo phân phối của tín
    hiệu redundancy quan sát được qua quá trình ingest, dùng z-score +
    sigmoid để biến tín hiệu thô thành xác suất P(duplicate) có ý nghĩa
    tương đối trong chính phân phối dữ liệu hiện tại.

    Input là `redundancy_signal = min(coverage_a_to_b, coverage_b_to_a)`
    (xem stage3_decision.py) — KHÔNG phải raw_logit của cross-encoder.

    Đây thay thế cho threshold cố định: thay vì so tín hiệu với 1 hằng
    số tuyệt đối, ta so nó với PHÂN PHỐI tín hiệu đã quan sát được —
    một cặp được coi là "khả nghi cao" nếu redundancy_signal của nó nằm
    ở phần đuôi cao của phân phối, bất kể domain/document cụ thể có
    shift thang đo thế nào.
    """

    def __init__(self, min_samples: int = 30):
        self.min_samples = min_samples
        self._signals: list[float] = []

    def update(self, signal: float) -> None:
        self._signals.append(signal)

    def calibrated_probability(self, signal: float) -> float:
        """
        P(duplicate) = sigmoid( z-score(signal) )

        Khi chưa có đủ dữ liệu để ước lượng phân phối (< 2 mẫu), trả
        về 0.0 — trung lập, nghiêng về phía "giữ lại" (an toàn hơn so
        với drop nhầm 1 chunk khi chưa biết gì về phân phối dữ liệu
        đang xử lý).
        """
        if len(self._signals) < 2:
            return 0.0

        arr = np.array(self._signals)
        mean = arr.mean()
        std = arr.std() + 1e-6
        z = (signal - mean) / std
        return float(1.0 / (1.0 + np.exp(-z)))

    def stats(self) -> dict:
        if not self._signals:
            return {"n": 0, "mean": 0.0, "std": 0.0}
        arr = np.array(self._signals)
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
