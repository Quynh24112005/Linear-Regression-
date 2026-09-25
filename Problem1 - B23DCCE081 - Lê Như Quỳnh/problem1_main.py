import os
import sys

# Cấu hình hiển thị tiếng Việt trên Windows Console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import modules đã xây dựng
from dataset_loader import load_and_preprocess_news_data
from problem1_scratch import (
    LinearRegressionNormalEq,
    LinearRegressionBGD,
    LinearRegressionMBGD,
    LinearRegressionSGD,
)

# Cấu hình Matplotlib
plt.rcParams["font.sans-serif"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)


def calculate_metrics(y_true, y_pred, time_ms, w_learned=None):
    """
    Tính các chỉ số đo lường hiệu năng: MSE, RMSE, MAE, R2 Score và Thời gian chạy
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "time_ms": time_ms,
        "weights": w_learned,
        "y_pred": y_pred,
    }


def run_problem1_pipeline():
    print("=" * 95)
    print(
        "      PROBLEM 1: LINEAR REGRESSION FROM FIRST PRINCIPLES (ONLINE NEWS POPULARITY DATASET)"
    )
    print("=" * 95)

    # 1. Tải và tiền xử lý dữ liệu
    print("\n---> [BƯỚC 1] NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU DỰ ÁN...")
    data = load_and_preprocess_news_data(use_log_target=True, test_ratio=0.2)

    X_tr_scaled = data["X_train_scaled"]
    y_tr = data["y_train"]
    X_te_scaled = data["X_test_scaled"]
    y_te = data["y_test"]

    X_tr_raw = data["X_train_raw"]
    X_te_raw = data["X_test_raw"]

    print(
        "\n---> [BƯỚC 2] HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH SCRATCH (FIRST PRINCIPLES)..."
    )

    # Đăng ký các mô hình cần thử nghiệm với siêu tham số ổn định
    models_scratch = {
        "1. Normal Equation (Closed-Form)": LinearRegressionNormalEq(),
        "2. Batch GD (BGD, lr=0.01, 500 ep)": LinearRegressionBGD(
            learning_rate=0.01, epochs=500
        ),
        "3. Mini-Batch GD (MBGD, B=256, lr=0.005, 300 ep)": LinearRegressionMBGD(
            learning_rate=0.005, epochs=300, batch_size=256
        ),
        "4. Stochastic GD (SGD, lr=0.001, 30 ep)": LinearRegressionSGD(
            learning_rate=0.001, epochs=30, decay=1e-4
        ),
    }

    results = {}

    for name, model in models_scratch.items():
        model.fit(X_tr_scaled, y_tr)
        y_pred = model.predict(X_te_scaled)
        res = calculate_metrics(y_te, y_pred, model.execution_time_ms, model.w)
        results[name] = res

    # 3. Đánh giá Mô hình Scikit-learn (Least Squares OLS) làm Benchmark
    print("\n---> [BƯỚC 3] HUẤN LUYỆN MÔ HÌNH BENCHMARK SCIKIT-LEARN (OLS)...")
    sk_model = SklearnLinearRegression()
    t0 = time.time()
    sk_model.fit(X_tr_scaled, y_tr)
    sk_time = (time.time() - t0) * 1000.0
    y_pred_sk = sk_model.predict(X_te_scaled)

    w_sk = np.hstack(([sk_model.intercept_], sk_model.coef_)).reshape(-1, 1)
    res_sk = calculate_metrics(y_te, y_pred_sk, sk_time, w_sk)
    results["5. Scikit-Learn LinearRegression (OLS)"] = res_sk

    # 4. In bảng so sánh kết quả
    print("\n" + "=" * 105)
    print(
        f"{'THUẬT TOÁN / MÔ HÌNH':<42} | {'MSE':<9} | {'RMSE':<9} | {'MAE':<9} | {'R² SCORE':<9} | {'THỜI GIAN (ms)':<14}"
    )
    print("=" * 105)
    for name, res in results.items():
        print(
            f"{name:<42} | {res['mse']:<9.4f} | {res['rmse']:<9.4f} | {res['mae']:<9.4f} | {res['r2']:<9.4f} | {res['time_ms']:<14.2f}"
        )
    print("=" * 105)

    # 5. Kiểm chứng tính công bằng (Fair Benchmark Verification)
    w_scratch_norm = results["1. Normal Equation (Closed-Form)"]["weights"]
    w_diff_norm = np.linalg.norm(w_scratch_norm - w_sk)
    print(
        f"\n[+] ĐỘ LỆCH VECTOR TRỌNG SỐ (Scratch Normal Eq vs Sklearn OLS): ||w_scratch - w_sklearn||₂ = {w_diff_norm:.8e}"
    )
    print(
        f"[+] ĐỘ LỆCH MSE GIỮA SCRATCH NORMAL EQ VÀ SKLEARN: {abs(results['1. Normal Equation (Closed-Form)']['mse'] - res_sk['mse']):.8e}"
    )

    # 5b. Phân tích Đa cộng tuyến (Multicollinearity Analysis)
    print("\n--- [BƯỚC 3b] PHÂN TÍCH ĐA CỘNG TUYẾN (MULTICOLLINEARITY) ---")
    cond_number = np.linalg.cond(np.dot(X_tr_scaled.T, X_tr_scaled))
    print(f"[+] Condition Number (κ) của X^T X: {cond_number:.2e}")
    print(
        f"    GIẢI THÍCH: κ rất lớn => ma trận X^T X gần suy biến do multicollinearity."
    )
    print(f"    => Tồn tại nhiều nghiệm w tối ưu khác nhau cho cùng một MSE tối thiểu.")
    print(
        f"    => Pseudoinverse (SVD) và sklearn (LAPACK) chọn nghiệm khác nhau trong null space,"
    )
    print(
        f"       dẫn đến ||w_scratch - w_sklearn||₂ = {w_diff_norm:.2e} nhưng MSE chênh lệch chỉ {abs(results['1. Normal Equation (Closed-Form)']['mse'] - res_sk['mse']):.2e}."
    )
    print(
        f"    KẾT LUẬN: So sánh MSE/R² là CÔNG BẰNG. So sánh trọng số không có ý nghĩa khi có multicollinearity."
    )

    # 5c. Thảo luận R² thấp
    r2_best = max(res["r2"] for res in results.values())
    print(f"\n[+] THẢO LUẬN VỀ R² THẤP (R² tốt nhất = {r2_best:.4f}):")
    print(
        f"    Online News Popularity dự đoán lượt chia sẻ bài viết — bản chất phi tuyến,"
    )
    print(
        f"    phụ thuộc vào nội dung ngữ nghĩa, thời điểm viral, hiệu ứng mạng xã hội."
    )
    print(
        f"    Mô hình tuyến tính chỉ dựa vào 58 meta-features => R² ~ 0.12 là kỳ vọng."
    )
    print(
        f"    Đây là GIỚI HẠN CỐ HỮU của mô hình tuyến tính, không phải lỗi implementation."
    )

    # 6. KHẢO SÁT 1: TÁC ĐỘNG CỦA FEATURE SCALING ĐẾN HỘI TỤ
    print(
        "\n---> [BƯỚC 4] THỰC NGHIỆM KHẢO SÁT TÁC ĐỘNG CỦA FEATURE SCALING VÀ LEARNING RATE..."
    )
    bgd_raw = LinearRegressionBGD(learning_rate=1e-12, epochs=500)
    bgd_raw.fit(X_tr_raw, y_tr)

    bgd_scaled = LinearRegressionBGD(learning_rate=0.05, epochs=500)
    bgd_scaled.fit(X_tr_scaled, y_tr)

    # 7. KHẢO SÁT 2: ĐỘ NHẠY VỚI TỐC ĐỘ HỌC (LEARNING RATE SENSITIVITY)
    lrs = [0.001, 0.01, 0.05, 0.1, 0.5]
    lr_histories = {}
    for lr in lrs:
        bgd_lr = LinearRegressionBGD(learning_rate=lr, epochs=200)
        bgd_lr.fit(X_tr_scaled, y_tr)
        lr_histories[lr] = bgd_lr.loss_history

    # 8. VẼ VÀ LƯU ĐỒ THỊ TRỰC QUAN HÓA
    visualize_problem1_results(
        models_scratch, results, bgd_raw, bgd_scaled, lr_histories, y_te
    )


def visualize_problem1_results(
    models_scratch, results, bgd_raw, bgd_scaled, lr_histories, y_te
):
    """
    Sinh 3 file đồ thị chất lượng cao cho báo cáo Problem 1
    """
    os.makedirs("plots", exist_ok=True)

    # --------------------------------------------------------------------------
    # ĐỒ THỊ 1: TỐC ĐỘ HỘI TỤ LOSS BGD vs MBGD vs SGD
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(
        models_scratch["2. Batch GD (BGD, lr=0.01, 500 ep)"].loss_history[:300],
        label="Batch GD (BGD, lr=0.01)",
        color="#1f77b4",
        linewidth=2,
    )
    plt.plot(
        models_scratch["3. Mini-Batch GD (MBGD, B=256, lr=0.005, 300 ep)"].loss_history[
            :300
        ],
        label="Mini-Batch GD (MBGD, B=256)",
        color="#2ca02c",
        linewidth=2,
    )
    plt.plot(
        models_scratch["4. Stochastic GD (SGD, lr=0.001, 30 ep)"].loss_history[:30],
        label="Stochastic GD (SGD)",
        color="#ff7f0e",
        alpha=0.8,
        linewidth=1.5,
    )

    plt.title(
        "Problem 1: So sánh Tốc độ Hội tụ MSE Loss theo Epochs",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("MSE Loss (Log Scale)", fontsize=11)
    plt.yscale("log")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot_conv = "plots/p1_convergence_comparison.png"
    plt.savefig(plot_conv, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ hội tụ tại: {plot_conv}")

    # --------------------------------------------------------------------------
    # ĐỒ THỊ 2: ẢNH HƯỞNG CỦA FEATURE SCALING & LEARNING RATE
    # --------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    # Subplot 2.1: Scaled vs Raw
    plt.subplot(1, 2, 1)
    plt.plot(
        bgd_scaled.loss_history[:200],
        label="Có Scaling (StandardScaler, lr=0.05)",
        color="green",
        linewidth=2,
    )
    plt.plot(
        bgd_raw.loss_history[:200],
        label="Không Scaling (Raw X, lr=1e-12)",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.title(
        "Tác động của Feature Scaling đến Hội tụ BGD", fontsize=12, fontweight="bold"
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()

    # Subplot 2.2: Learning Rate Sensitivity
    plt.subplot(1, 2, 2)
    for lr, hist in lr_histories.items():
        if lr == 0.5:
            # Vẽ 10 epoch đầu của lr=0.5 để thấy bùng nổ loss
            plt.plot(
                hist[:20],
                label=f"lr = {lr} (Bùng nổ loss)",
                linestyle=":",
                color="crimson",
                linewidth=2,
            )
        else:
            plt.plot(hist[:100], label=f"lr = {lr}", linewidth=1.8)
    plt.title(
        "Ảnh hưởng của Tốc độ học (Learning Rate η)", fontsize=12, fontweight="bold"
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plot_scaling = "plots/p1_scaling_and_lr_impact.png"
    plt.savefig(plot_scaling, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ khảo sát Scaling & LR tại: {plot_scaling}")

    # --------------------------------------------------------------------------
    # ĐỒ THỊ 3: CHẨN ĐOÁN THẶNG DƯ (RESIDUAL DIAGNOSTICS) & BENCHMARK METRICS
    # --------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    # Subplot 3.1: Residual Plot
    plt.subplot(1, 2, 1)
    y_pred_norm = results["1. Normal Equation (Closed-Form)"]["y_pred"]
    residuals = y_te - y_pred_norm
    plt.scatter(
        y_pred_norm, residuals, alpha=0.3, color="royalblue", edgecolors="none", s=15
    )
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1.5)
    plt.title(
        "Biểu đồ Thặng dư (Residuals vs Fitted Values)", fontsize=12, fontweight="bold"
    )
    plt.xlabel("Giá trị dự báo (Fitted Values y_pred)")
    plt.ylabel("Thặng dư (Residuals = y_true - y_pred)")

    # Subplot 3.2: Histogram Phân bố Thặng dư
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=50, color="teal", alpha=0.7, edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    plt.title(
        "Phân bố Thặng dư (Residual Distribution)", fontsize=12, fontweight="bold"
    )
    plt.xlabel("Thặng dư (Residuals)")
    plt.ylabel("Tần suất (Frequency)")

    plt.tight_layout()
    plot_residuals = "plots/p1_residuals_diagnostics.png"
    plt.savefig(plot_residuals, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ chẩn đoán thặng dư tại: {plot_residuals}")

    # --------------------------------------------------------------------------
    # ĐỒ THỊ BỔ SUNG: Q-Q PLOT KIỂM TRA NORMALITY CỦA THẶNG DƯ
    # --------------------------------------------------------------------------
    try:
        from scipy import stats

        plt.figure(figsize=(7, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(
            "Q-Q Plot: Kiểm tra Phân phối Chuẩn của Thặng dư",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        plot_qq = "plots/p1_qq_plot.png"
        plt.savefig(plot_qq, dpi=300)
        plt.close()
        print(f"[+] Đã lưu Q-Q Plot tại: {plot_qq}")
    except ImportError:
        print("[!] Cần cài scipy để vẽ Q-Q Plot: pip install scipy")

    # --------------------------------------------------------------------------
    # ĐỒ THỊ 4: SO SÁNH CHỈ SỐ BENCHMARK GIỮA CÁC MÔ HÌNH
    # --------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    model_names = [name.split(".")[1].split("(")[0].strip() for name in results.keys()]
    mse_values = [res["mse"] for res in results.values()]
    r2_values = [res["r2"] for res in results.values()]
    time_values = [res["time_ms"] for res in results.values()]

    # Subplot 4.1: MSE Compare
    plt.subplot(1, 2, 1)
    bars1 = plt.barh(
        model_names,
        mse_values,
        color=["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"],
    )
    plt.title(
        "So sánh MSE Loss trên tập Test (Thấp hơn là tốt hơn)",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("MSE Loss")
    for bar in bars1:
        w = bar.get_width()
        plt.text(
            w + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    # Subplot 4.2: Execution Time Compare
    plt.subplot(1, 2, 2)
    bars2 = plt.barh(
        model_names,
        time_values,
        color=["#2ecc71", "#3498db", "#9b59b6", "#e67e22", "#e74c3c"],
    )
    plt.title("So sánh Thời gian Huấn luyện (ms)", fontsize=11, fontweight="bold")
    plt.xlabel("Thời gian (ms)")
    plt.xscale("log")
    for bar in bars2:
        w = bar.get_width()
        plt.text(
            w * 1.1,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}ms",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plot_benchmark = "plots/p1_benchmark_metrics.png"
    plt.savefig(plot_benchmark, dpi=300)
    plt.close()
    print(f"[+] Đã lưu biểu đồ so sánh Benchmark tại: {plot_benchmark}")


if __name__ == "__main__":
    run_problem1_pipeline()
