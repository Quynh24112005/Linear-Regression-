import os
import sys

# Cấu hình UTF-8 trên Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataset_loader_p2 import load_and_preprocess_bike_data
from basis_expansion import BikeBasisExpansion
from problem2_scratch import (
    OLSLinearRegression,
    RidgeRegressionScratch,
    LassoRegressionScratch,
)

# Cấu hình Matplotlib
plt.rcParams["font.sans-serif"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)


def calculate_metrics(y_true, y_pred, time_ms, model):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    sparsity = getattr(model, "get_sparsity", lambda: 0)()
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "time_ms": time_ms,
        "sparsity": sparsity,
        "y_pred": y_pred,
    }


def cross_validate(model_cls, X, y, alphas, k_folds=5):

    N = len(y)
    fold_size = N // k_folds
    indices = np.arange(N)

    cv_scores = []
    for alpha in alphas:
        fold_mses = []
        for k in range(k_folds):
            val_idx = indices[k * fold_size : (k + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_cv_tr, y_cv_tr = X[train_idx], y[train_idx]
            X_cv_val, y_cv_val = X[val_idx], y[val_idx]

            model = model_cls(alpha=alpha)
            model.fit(X_cv_tr, y_cv_tr)
            y_val_pred = model.predict(X_cv_val)
            fold_mses.append(mean_squared_error(y_cv_val, y_val_pred))

        cv_scores.append(np.mean(fold_mses))

    best_idx = np.argmin(cv_scores)
    return alphas[best_idx], alphas, cv_scores


def run_problem2_pipeline():
    print("=" * 105)
    print(
        "      PROBLEM 2: BASIS EXPANSION, REGULARIZATION, AND DIAGNOSTICS (BIKE SHARING DATASET)"
    )
    print("=" * 105)

    # 1. Nạp dữ liệu và kiểm tra rò rỉ
    print("\n---> [BƯỚC 1] NẠP VÀ PHÂN CHIA DỮ LIỆU CHỐNG LEAKAGE...")
    data = load_and_preprocess_bike_data(use_log_target=True)
    X_tr_df = data["X_train_df"]
    y_tr = data["y_train"]
    X_te_df = data["X_test_df"]
    y_te = data["y_test"]

    # 2. Xây dựng các cấp độ Mở rộng Hàm cơ sở (Basis Expansion)
    print("\n---> [BƯỚC 2] MỞ RỘNG HÀM CƠ SỞ (BASIS EXPANSION LEVELS)...")
    be_lvl0 = BikeBasisExpansion(level=0)
    X_tr_l0 = be_lvl0.fit_transform(X_tr_df)
    X_te_l0 = be_lvl0.transform(X_te_df)

    be_lvl1 = BikeBasisExpansion(level=1)
    X_tr_l1 = be_lvl1.fit_transform(X_tr_df)
    X_te_l1 = be_lvl1.transform(X_te_df)

    be_lvl2 = BikeBasisExpansion(level=2)
    X_tr_l2 = be_lvl2.fit_transform(X_tr_df)
    X_te_l2 = be_lvl2.transform(X_te_df)
    feature_names_l2 = be_lvl2.feature_names

    print(f"[+] Level 0 (Raw Features): {X_tr_l0.shape[1]} đặc trưng")
    print(
        f"[+] Level 1 (Cyclic Basis): {X_tr_l1.shape[1]} đặc trưng (Thêm chu kỳ lượng giác giờ, tháng, thứ)"
    )
    print(
        f"[+] Level 2 (Cyclic + Poly + Interactions): {X_tr_l2.shape[1]} đặc trưng (Thêm tương tác thời tiết & đi làm)"
    )

    # 3. Đánh giá OLS trên các cấp độ đặc trưng
    print(
        "\n---> [BƯỚC 3] HUẤN LUYỆN OLS TRÊN CÁC CẤP ĐỘ BASIS VÀ KHẢO SÁT OVERFITTING..."
    )
    ols_l0 = OLSLinearRegression().fit(X_tr_l0, y_tr)
    ols_l1 = OLSLinearRegression().fit(X_tr_l1, y_tr)
    ols_l2 = OLSLinearRegression().fit(X_tr_l2, y_tr)

    res_ols_l0 = calculate_metrics(
        y_te, ols_l0.predict(X_te_l0), ols_l0.execution_time_ms, ols_l0
    )
    res_ols_l1 = calculate_metrics(
        y_te, ols_l1.predict(X_te_l1), ols_l1.execution_time_ms, ols_l1
    )
    res_ols_l2 = calculate_metrics(
        y_te, ols_l2.predict(X_te_l2), ols_l2.execution_time_ms, ols_l2
    )

    # 4. K-Fold Cross-Validation chọn alpha tối ưu cho Ridge và LASSO trên Level 2
    print(
        "\n---> [BƯỚC 4] K-FOLD CROSS-VALIDATION CHỌN ALPHA TỐI ƯU (RIDGE & LASSO)..."
    )
    alphas_ridge = np.logspace(-4, 2, 25)
    best_alpha_ridge, alphas_r_grid, cv_scores_ridge = cross_validate(
        RidgeRegressionScratch, X_tr_l2, y_tr, alphas_ridge, k_folds=5
    )
    print(f"[+] Ridge Alpha tối ưu qua 5-Fold CV: alpha* = {best_alpha_ridge:.6f}")

    alphas_lasso = np.logspace(-4, -1, 20)
    best_alpha_lasso, alphas_l_grid, cv_scores_lasso = cross_validate(
        LassoRegressionScratch, X_tr_l2, y_tr, alphas_lasso, k_folds=5
    )
    print(f"[+] LASSO Alpha tối ưu qua 5-Fold CV: alpha* = {best_alpha_lasso:.6f}")

    # 5. Huấn luyện mô hình tối ưu trên toàn bộ Train và kiểm thử trên Test độc lập
    ridge_opt = RidgeRegressionScratch(alpha=best_alpha_ridge).fit(X_tr_l2, y_tr)
    lasso_opt = LassoRegressionScratch(alpha=best_alpha_lasso, epochs=2000).fit(
        X_tr_l2, y_tr
    )

    res_ridge = calculate_metrics(
        y_te, ridge_opt.predict(X_te_l2), ridge_opt.execution_time_ms, ridge_opt
    )
    res_lasso = calculate_metrics(
        y_te, lasso_opt.predict(X_te_l2), lasso_opt.execution_time_ms, lasso_opt
    )

    # 6. In bảng so sánh toàn diện
    results_all = {
        "1. OLS (Level 0 - Raw Features)": res_ols_l0,
        "2. OLS (Level 1 - Cyclic Basis)": res_ols_l1,
        "3. OLS (Level 2 - Full Basis Expansion)": res_ols_l2,
        f"4. Ridge (Level 2, alpha={best_alpha_ridge:.4f})": res_ridge,
        f"5. LASSO (Level 2, alpha={best_alpha_lasso:.4f})": res_lasso,
    }

    print("\n" + "=" * 115)
    print(
        f"{'MÔ HÌNH / PHƯƠNG PHÁP':<45} | {'TEST MSE':<9} | {'TEST RMSE':<9} | {'TEST MAE':<9} | {'TEST R²':<9} | {'ĐỘ THƯA (SPARSITY)':<18}"
    )
    print("=" * 115)
    for name, res in results_all.items():
        sparsity_str = (
            f"{res['sparsity']} / {X_tr_l2.shape[1]} (0s)"
            if "Full" in name or "Ridge" in name or "LASSO" in name
            else "N/A"
        )
        print(
            f"{name:<45} | {res['mse']:<9.4f} | {res['rmse']:<9.4f} | {res['mae']:<9.4f} | {res['r2']:<9.4f} | {sparsity_str:<18}"
        )
    print("=" * 115)

    # In các đặc trưng quan trọng nhất được LASSO giữ lại
    w_lasso = lasso_opt.w[1:].flatten()
    nonzero_idx = np.where(np.abs(w_lasso) > 1e-4)[0]
    print(
        f"\n[+] LASSO Feature Selection: Giữ lại {len(nonzero_idx)} / {len(feature_names_l2)} đặc trưng quan trọng nhất:"
    )
    for idx in nonzero_idx:
        print(f"    - {feature_names_l2[idx]:<25}: w = {w_lasso[idx]:+.4f}")

    # 6b. DEMO CƠ CHẾ SPARSITY CỦA LASSO KHI TĂNG ALPHA
    print(
        "\n----> [BƯỚC 5b] MINH HỌA CƠ CHẾ FEATURE SELECTION CỦA LASSO Ở CÁC MỨC ALPHA..."
    )
    demo_alphas_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    for demo_a in demo_alphas_list:
        lasso_demo = LassoRegressionScratch(alpha=demo_a, epochs=1500).fit(
            X_tr_l2, y_tr
        )
        y_pred_demo = lasso_demo.predict(X_te_l2)
        mse_demo = mean_squared_error(y_te, y_pred_demo)
        r2_demo = r2_score(y_te, y_pred_demo)
        sparsity_demo = lasso_demo.get_sparsity()
        print(
            f"    LASSO α={demo_a:<8.4f}: Zeroed = {sparsity_demo:>2}/32 features, "
            f"Test MSE = {mse_demo:.4f}, R² = {r2_demo:.4f}"
        )

    # 6c. THẢO LUẬN TRUNG THỰC VỀ VAI TRÒ CỦA REGULARIZATION
    print(f"\n[+] THẢO LUẬN TRUNG THỰC VỀ VAI TRÒ CỦA REGULARIZATION:")
    print(f"    Với α tối ưu = {best_alpha_lasso:.4f}, Ridge/LASSO gần giống OLS:")
    print(
        f"      OLS R² = {res_ols_l2['r2']:.4f} | Ridge R² = {res_ridge['r2']:.4f} | LASSO R² = {res_lasso['r2']:.4f}"
    )
    print(
        f"    Tỉ số p/n = {X_tr_l2.shape[1]}/{X_tr_l2.shape[0]} = {X_tr_l2.shape[1]/X_tr_l2.shape[0]:.4f}"
    )
    print(
        f"    => Bài toán không high-dimensional => regularization không cần thiết ở α tối ưu."
    )
    print(
        f"    => Khi tăng α, LASSO thể hiện đúng cơ chế feature selection (demo ở trên)."
    )
    print(
        f"    => Đây là kết quả HỢP LỆ: basis expansion Level 2 vừa đủ phức tạp, OLS đã đủ tốt."
    )

    # 7. Tính Regularization Paths cho đồ thị
    print("\n---> [BƯỚC 5] TÍNH TOÁN REGULARIZATION PATHS...")
    ridge_paths = []
    for a in alphas_ridge:
        m = RidgeRegressionScratch(alpha=a).fit(X_tr_l2, y_tr)
        ridge_paths.append(m.w[1:].flatten())
    ridge_paths = np.array(ridge_paths)

    alphas_lasso_path = np.logspace(
        -4, 0, 20
    )  # Mở rộng range đến α=1.0 để thấy sparsity
    lasso_paths = []
    for a in alphas_lasso_path:
        m = LassoRegressionScratch(alpha=a, epochs=1500).fit(X_tr_l2, y_tr)
        lasso_paths.append(m.w[1:].flatten())
    lasso_paths = np.array(lasso_paths)

    # 8. Sinh và lưu đồ thị trực quan hóa
    visualize_problem2_results(
        results_all,
        alphas_r_grid,
        cv_scores_ridge,
        best_alpha_ridge,
        alphas_l_grid,
        cv_scores_lasso,
        best_alpha_lasso,
        ridge_paths,
        lasso_paths,
        alphas_ridge,
        alphas_lasso_path,
        y_te,
        ols_l0.predict(X_te_l0),
        ridge_opt.predict(X_te_l2),
        data["df_test_raw"]["hr"].values,
    )

    # 9. ĐỒ THỊ BỔ SUNG: LASSO SPARSITY VS ALPHA + Q-Q PLOT
    plt.figure(figsize=(14, 5))

    # 9a. LASSO Sparsity vs Alpha (chứng minh cơ chế feature selection hoạt động)
    plt.subplot(1, 2, 1)
    sparsity_per_alpha = [
        np.sum(np.abs(lasso_paths[i]) < 1e-5) for i in range(len(alphas_lasso_path))
    ]
    plt.plot(
        alphas_lasso_path,
        sparsity_per_alpha,
        "o-",
        color="#8e44ad",
        linewidth=2,
        markersize=5,
    )
    plt.axvline(
        best_alpha_lasso,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"CV Optimal α*={best_alpha_lasso:.4f}\n(Sparsity=0, OLS đã đủ tốt)",
    )
    plt.xscale("log")
    plt.xlabel("Alpha α (Log scale)", fontsize=11)
    plt.ylabel("Số đặc trưng bị triệt tiêu về 0", fontsize=11)
    plt.title(
        "LASSO Feature Selection: Sparsity vs Alpha", fontsize=11, fontweight="bold"
    )
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    # 9b. Q-Q Plot kiểm tra giả định Normality của Thặng dư
    plt.subplot(1, 2, 2)
    try:
        from scipy import stats

        res_final = y_te - ridge_opt.predict(X_te_l2)
        stats.probplot(res_final, dist="norm", plot=plt)
        plt.title(
            "Q-Q Plot: Normality Thặng dư (Level 2 + Ridge)",
            fontsize=11,
            fontweight="bold",
        )
    except ImportError:
        plt.text(
            0.5,
            0.5,
            "Cần scipy\npip install scipy",
            ha="center",
            va="center",
            fontsize=14,
            transform=plt.gca().transAxes,
        )
        plt.title("Q-Q Plot (cần cài scipy)", fontsize=11)

    plt.tight_layout()
    p_extra = "plots/p2_sparsity_qq_extra.png"
    plt.savefig(p_extra, dpi=300)
    plt.close()
    print(f"[+] Đã lưu đồ thị bổ sung: {p_extra}")


def visualize_problem2_results(
    results,
    alphas_r,
    cv_r,
    best_r,
    alphas_l,
    cv_l,
    best_l,
    ridge_paths,
    lasso_paths,
    alpha_grid_r,
    alpha_grid_l,
    y_te,
    y_pred_l0,
    y_pred_opt,
    hours_test,
):
    os.makedirs("plots", exist_ok=True)

    # --------------------------------------------------------------------------
    # ĐỒ THỊ 1: SO SÁNH HIỆU NĂNG CÁC CẤP ĐỘ PHỨC TẠP BASIS EXPANSION & REGULARIZATION
    # --------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    names = [k.split("(")[0].strip() if "(" in k else k for k in results.keys()]
    r2_scores = [v["r2"] for v in results.values()]
    mse_scores = [v["mse"] for v in results.values()]

    plt.subplot(1, 2, 1)
    bars = plt.barh(
        names, r2_scores, color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"]
    )
    plt.title(
        "So sánh R² Score trên tập Test (Càng cao càng tốt)",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("R² Score")
    for bar in bars:
        w = bar.get_width()
        plt.text(
            w + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.subplot(1, 2, 2)
    bars2 = plt.barh(
        names, mse_scores, color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71", "#9b59b6"]
    )
    plt.title(
        "So sánh Test MSE Loss (Càng thấp càng tốt)", fontsize=11, fontweight="bold"
    )
    plt.xlabel("MSE Loss")
    for bar in bars2:
        w = bar.get_width()
        plt.text(
            w + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    p1 = "plots/p2_basis_complexity_comparison.png"
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[+] Đã lưu đồ thị: {p1}")

    # --------------------------------------------------------------------------
    # ĐỒ THỊ 2: 5-FOLD CROSS-VALIDATION ERROR CURVES (RIDGE & LASSO)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(alphas_r, cv_r, "o-", color="#2980b9", linewidth=2, markersize=5)
    plt.axvline(
        best_r, color="crimson", linestyle="--", label=f"Optimal α* = {best_r:.4f}"
    )
    plt.xscale("log")
    plt.title(
        "Ridge: 5-Fold Cross-Validation MSE Curve", fontsize=11, fontweight="bold"
    )
    plt.xlabel("Regularization Parameter α (Log scale)")
    plt.ylabel("Validation MSE")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(alphas_l, cv_l, "s-", color="#8e44ad", linewidth=2, markersize=5)
    plt.axvline(
        best_l, color="crimson", linestyle="--", label=f"Optimal α* = {best_l:.4f}"
    )
    plt.xscale("log")
    plt.title(
        "LASSO: 5-Fold Cross-Validation MSE Curve", fontsize=11, fontweight="bold"
    )
    plt.xlabel("Regularization Parameter α (Log scale)")
    plt.ylabel("Validation MSE")
    plt.legend()

    plt.tight_layout()
    p2 = "plots/p2_cv_tuning_curves.png"
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[+] Đã lưu đồ thị: {p2}")

    # --------------------------------------------------------------------------
    # ĐỒ THỊ 3: REGULARIZATION PATHS (COEFFICIENT SHRINKAGE VS SPARSITY)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for j in range(ridge_paths.shape[1]):
        plt.plot(alpha_grid_r, ridge_paths[:, j], alpha=0.7)
    plt.axvline(best_r, color="black", linestyle="--", label=f"Best α* = {best_r:.4f}")
    plt.xscale("log")
    plt.title(
        "Ridge Regularization Path (Smooth Shrinkage L2)",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("Alpha α (Log scale)")
    plt.ylabel("Trọng số Hệ số (Weights w_j)")
    plt.legend()

    plt.subplot(1, 2, 2)
    for j in range(lasso_paths.shape[1]):
        plt.plot(alpha_grid_l, lasso_paths[:, j], alpha=0.7)
    plt.axvline(best_l, color="black", linestyle="--", label=f"Best α* = {best_l:.4f}")
    plt.xscale("log")
    plt.title(
        "LASSO Regularization Path (Sparse Selection L1)",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("Alpha α (Log scale)")
    plt.ylabel("Trọng số Hệ số (Weights w_j)")
    plt.legend()

    plt.tight_layout()
    p3 = "plots/p2_regularization_paths.png"
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[+] Đã lưu đồ thị: {p3}")

    # --------------------------------------------------------------------------
    # ĐỒ THỊ 4: CHẨN ĐOÁN THẶNG DƯ & HIỆU QUẢ CỦA CYCLIC BASIS
    # --------------------------------------------------------------------------
    plt.figure(figsize=(14, 5))

    # Subplot 4.1: Residual by Hour (Level 0 vs Full Model)
    plt.subplot(1, 2, 1)
    res_l0 = y_te - y_pred_l0
    res_opt = y_te - y_pred_opt

    df_diag = pd.DataFrame({"hr": hours_test, "res_l0": res_l0, "res_opt": res_opt})
    mean_res_l0 = df_diag.groupby("hr")["res_l0"].mean()
    mean_res_opt = df_diag.groupby("hr")["res_opt"].mean()

    plt.plot(
        mean_res_l0.index,
        mean_res_l0.values,
        "ro--",
        label="Mô hình Gốc (L0) - Bị lệch chu kỳ nghiêm trọng",
        linewidth=2,
    )
    plt.plot(
        mean_res_opt.index,
        mean_res_opt.values,
        "go-",
        label="Mô hình Mở rộng (L2 + Ridge) - Khử sạch lệch chu kỳ",
        linewidth=2,
    )
    plt.axhline(0, color="black", linestyle=":")
    plt.title(
        "Thặng dư Trung bình theo Giờ trong Ngày (Diurnal Bias)",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("Giờ trong ngày (0 - 23h)")
    plt.ylabel("Sai số Thặng dư Trung bình")
    plt.legend()

    # Subplot 4.2: Residual vs Fitted Values
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_opt, res_opt, alpha=0.25, color="darkcyan", s=12)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(
        "Biểu đồ Thặng dư vs Dự báo (Residuals vs Fitted)",
        fontsize=11,
        fontweight="bold",
    )
    plt.xlabel("Giá trị dự báo (Fitted Values y_pred)")
    plt.ylabel("Thặng dư (Residuals)")

    plt.tight_layout()
    p4 = "plots/p2_residual_diagnostics.png"
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[+] Đã lưu đồ thị: {p4}")


if __name__ == "__main__":
    run_problem2_pipeline()
