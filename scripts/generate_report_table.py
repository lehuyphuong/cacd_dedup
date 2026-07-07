"""
generate_report_table.py

Đọc 2 file CSV:
  1. Full_dataset_old_paper_theshold_0_8.csv   -> chứa NoFilter, ExactNorm,
     MinHashLSH0.8, Similarity0.8, NERExact (mỗi method x 18 chunking config)
  2. <file CACD>.csv                            -> chứa 18 dòng kết quả CACD
     (1 dòng / chunking config)

In ra:
  - Toàn bộ giá trị thô của từng config cho từng method (để đối chiếu tay)
  - Trung bình (mean) trên 18 config cho: Precision, Recall, IoU, Drop%,
    Storage (MB), Time (s)

Cách chạy:
    python generate_report_table.py <path_to_old_paper_csv> <path_to_cacd_csv>
"""

import sys
import pandas as pd


# Các cột dùng để tính trung bình — LẤY THẲNG TỪ CỘT CÓ SẴN TRONG CSV,
# không tự suy ra / không tính lại bằng công thức khác.
METRIC_COLUMNS = {
    "Precision":     "precision_raw",
    "Recall":        "recall_raw",
    "IoU":           "iou_raw",
    "Drop %":        "filter_reduction_pct",
    "Storage (MB)":  "storage_mb",
    "Time (s)":      "ingest_time_s",
}


def load_baseline_methods(old_paper_csv: str) -> dict[str, pd.DataFrame]:
    """Tách file baseline (nhiều filter_method) thành dict {method_name: sub-dataframe}."""
    df = pd.read_csv(old_paper_csv)

    print(f"\n[DEBUG] Đã đọc '{old_paper_csv}': {len(df)} dòng.")
    print("[DEBUG] Các filter_method có trong file:", sorted(df["filter_method"].unique()))

    methods = {}
    for method in df["filter_method"].unique():
        sub = df[df["filter_method"] == method].copy()
        methods[method] = sub
    return methods


def load_cacd(cacd_csv: str) -> pd.DataFrame:
    df = pd.read_csv(cacd_csv)
    print(f"\n[DEBUG] Đã đọc '{cacd_csv}': {len(df)} dòng.")
    return df


def print_raw_values(label: str, df: pd.DataFrame) -> None:
    """In từng dòng + từng cột metric để đối chiếu bằng tay."""
    print(f"\n--- Giá trị thô từng config — {label} ({len(df)} dòng) ---")
    cols_to_show = ["config_name"] + list(METRIC_COLUMNS.values())
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    print(df[cols_to_show].to_string(index=False))


def compute_mean_row(label: str, df: pd.DataFrame) -> dict:
    """Tính trung bình (mean) trên toàn bộ các dòng (config) của 1 method."""
    row = {"Method": label, "n_configs": len(df)}
    for display_name, col in METRIC_COLUMNS.items():
        if col not in df.columns:
            row[display_name] = None
            continue
        values = df[col].astype(float)
        mean_val = values.mean()
        row[display_name] = mean_val
        # In công thức trung bình tường minh để đối chiếu bằng tay
        vals_str = " + ".join(f"{v:.4f}" for v in values)
        print(f"\n[{label}] Mean({display_name}) = ({vals_str}) / {len(values)} = {mean_val:.4f}")
    return row


def main():
    if len(sys.argv) != 3:
        print("Cách dùng: python generate_report_table.py <old_paper_csv> <cacd_csv>")
        sys.exit(1)

    old_paper_csv, cacd_csv = sys.argv[1], sys.argv[2]

    baseline_methods = load_baseline_methods(old_paper_csv)
    cacd_df = load_cacd(cacd_csv)

    # In toàn bộ giá trị thô trước, để dễ đối chiếu tay từng dòng
    for method_name, sub_df in baseline_methods.items():
        print_raw_values(method_name, sub_df)
    print_raw_values("CACD (file thứ 2)", cacd_df)

    # Tính trung bình cho từng method, kèm công thức tường minh
    report_rows = []
    # Thứ tự hiển thị cố định, khớp với các bảng trước đó
    display_order = ["NoFilter", "ExactNorm", "MinHashLSH0.8", "Similarity0.8", "NERExact"]
    for method_name in display_order:
        if method_name not in baseline_methods:
            print(f"[CẢNH BÁO] Không tìm thấy filter_method = '{method_name}' trong file baseline!")
            continue
        report_rows.append(compute_mean_row(method_name, baseline_methods[method_name]))

    report_rows.append(compute_mean_row("CACD (file thứ 2)", cacd_df))

    # In bảng tổng hợp cuối cùng
    report = pd.DataFrame(report_rows)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 160)

    print("\n\n================ BẢNG BÁO CÁO CUỐI CÙNG (mean / 18 config) ================")
    print(report.to_string(index=False))

    # Xuất luôn ra CSV để tiện đối chiếu / paste vào báo cáo
    out_path = "report_table_output.csv"
    report.to_csv(out_path, index=False)
    print(f"\nĐã lưu bảng vào: {out_path}")


if __name__ == "__main__":
    main()
