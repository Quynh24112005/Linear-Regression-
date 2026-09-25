import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np


class StandardScalerScratch:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Tránh chia cho 0 với các thuộc tính không đổi
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def download_and_extract_bike_sharing(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    hour_csv = os.path.join(data_dir, "hour.csv")

    if os.path.exists(hour_csv):
        return hour_csv

    zip_path = os.path.join(data_dir, "bike_sharing.zip")
    url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 100000:
        print(f"[+] Đang tải Bike Sharing dataset từ {url}...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    print("[+] Giải nén file zip...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    if os.path.exists(hour_csv):
        return hour_csv
    raise FileNotFoundError("Không tìm thấy hour.csv sau khi giải nén.")


def load_and_preprocess_bike_data(data_dir="data", use_log_target=True):
    csv_path = download_and_extract_bike_sharing(data_dir=data_dir)
    df = pd.read_csv(csv_path)
    print(
        f"[+] Đã tải dữ liệu Bike Sharing: {df.shape[0]} dòng, {df.shape[1]} cột.",
        flush=True,
    )

    # 1. Phát hiện và loại bỏ bẫy rò rỉ dữ liệu (Leakage Trap)
    leak_check = np.all(df["casual"] + df["registered"] == df["cnt"])
    print(
        f"[+] Kiểm tra Bẫy Rò Rỉ: casual + registered == cnt? -> {leak_check} (ĐÃ LOẠI BỎ để ngăn chặn data leakage)",
        flush=True,
    )

    drop_cols = ["instant", "dteday", "casual", "registered", "cnt"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Phân chia dữ liệu theo Năm để kiểm tra Temporal Generalization
    train_mask = df["yr"] == 0
    test_mask = df["yr"] == 1

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    # Loại bỏ cột 'yr' sau khi đã dùng để chia tập (để tránh bias hằng số trong từng tập)
    feature_cols = [c for c in feature_cols if c != "yr"]

    X_train_df = df_train[feature_cols]
    y_train = df_train["cnt"].values.astype(np.float64)

    X_test_df = df_test[feature_cols]
    y_test = df_test["cnt"].values.astype(np.float64)

    if use_log_target:
        print(
            "[+] Áp dụng biến đổi log1p cho target count: y = log(1 + cnt)", flush=True
        )
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)

    print(
        f"[+] Phân chia Temporal Split: {len(df_train)} mẫu Train (Năm 2011), {len(df_test)} mẫu Test (Năm 2012).",
        flush=True,
    )
    print(
        f"[+] Các đặc trưng cơ sở ban đầu ({len(feature_cols)}): {feature_cols}",
        flush=True,
    )

    return {
        "X_train_df": X_train_df,
        "y_train": y_train,
        "X_test_df": X_test_df,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "df_train_raw": df_train,
        "df_test_raw": df_test,
    }


if __name__ == "__main__":
    data = load_and_preprocess_bike_data()
    print("Train shape:", data["X_train_df"].shape)
    print("Test shape:", data["X_test_df"].shape)
