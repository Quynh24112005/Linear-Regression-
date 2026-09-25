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
        # Tránh chia cho 0 với các đặc trưng hằng số (zero variance)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def download_and_extract_news_popularity(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    csv_filename = "OnlineNewsPopularity.csv"

    # 1. Kiểm tra nếu file CSV đã tồn tại
    for root, dirs, files in os.walk(data_dir):
        if csv_filename in files:
            return os.path.join(root, csv_filename)

    zip_path = os.path.join(data_dir, "online_news_popularity.zip")
    url = "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip"

    # 2. Nếu file zip chưa có hoặc rỗng, tiến hành tải
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000000:
        print(f"[+] Đang tải dữ liệu từ {url}...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 3. Giải nén file zip
    print("[+] Giải nén dữ liệu...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    for root, dirs, files in os.walk(data_dir):
        if csv_filename in files:
            return os.path.join(root, csv_filename)

    raise FileNotFoundError(
        "Không tìm thấy file OnlineNewsPopularity.csv sau khi giải nén."
    )


def load_and_preprocess_news_data(use_log_target=True, test_ratio=0.2, data_dir="data"):
    csv_path = download_and_extract_news_popularity(data_dir=data_dir)
    df = pd.read_csv(csv_path)

    # Xóa khoảng trắng thừa ở tên cột
    df.columns = [col.strip() for col in df.columns]

    print(f"[+] Dữ liệu gốc: {df.shape[0]} mẫu, {df.shape[1]} thuộc tính.")

    # Loại bỏ các cột phi số/không tiên đoán
    non_predictive = ["url", "timedelta"]
    feature_cols = [
        col for col in df.columns if col not in non_predictive and col != "shares"
    ]

    X = df[feature_cols].values.astype(np.float64)
    y = df["shares"].values.astype(np.float64)

    # Xử lý biến mục tiêu (target) bị skewed
    if use_log_target:
        print(
            "[+] Áp dụng biến đổi log1p cho target (log(1 + shares)) nhằm giảm độ lệch."
        )
        y = np.log1p(y)

    # Phân chia train/test theo thứ tự xuất bản (Sequential split)
    n_samples = len(df)
    n_test = int(n_samples * test_ratio)
    n_train = n_samples - n_test

    X_train_raw = X[:n_train]
    y_train = y[:n_train]
    X_test_raw = X[n_train:]
    y_test = y[n_train:]

    print(
        f"[+] Phân chia dữ liệu (Sequential Split): {n_train} mẫu Train, {n_test} mẫu Test ({len(feature_cols)} đặc trưng)."
    )

    # Chuẩn hóa đặc trưng (Fit trên Train, Transform cho cả Train và Test)
    scaler = StandardScalerScratch()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return {
        "X_train_raw": X_train_raw,
        "X_train_scaled": X_train_scaled,
        "y_train": y_train,
        "X_test_raw": X_test_raw,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "feature_names": feature_cols,
        "scaler": scaler,
    }


if __name__ == "__main__":
    data = load_and_preprocess_news_data()
    print("X_train_scaled shape:", data["X_train_scaled"].shape)
    print("y_train shape:", data["y_train"].shape)
