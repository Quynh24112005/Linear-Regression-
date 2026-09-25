import os
import sys

# Configure UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import nbformat as nbf
from nbclient import NotebookClient

def create_problem2_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Cell 1: Markdown Title & Overview
    c1_md = r"""# Báo Cáo Thực Hành: Problem 2 — Model Comparison and Thresholds

**Sinh viên:** Lê Như Quỳnh - B23DCCE081  
**Học phần:** Học Máy (Machine Learning)  
**Tập dữ liệu lựa chọn:** [Wine Dataset (UCI / Scikit-learn)](https://archive.ics.uci.edu/dataset/109/wine)  

---

## Tổng quan bài toán
Theo yêu cầu của đề bài:
1. **Phần A (Compare multiclass models):** Sử dụng chung tập dữ liệu Wine (3 lớp) đã phân chia từ Problem 1. Thiết lập một so sánh công bằng giữa mô hình `OneVsRestClassifier` kết hợp Logistic Regression và mô hình `Multinomial Logistic Regression` của thư viện `scikit-learn`. Báo cáo Accuracy, Macro F1, Log Loss và phân tích sự khác biệt về xác suất dự đoán trên một mẫu cụ thể.
2. **Phần B (Choose a threshold):**
   * Định nghĩa lớp dương tính (Positive class) là lớp **Cultivar 0**, gộp 2 lớp còn lại thành lớp âm tính (Negative).
   * Thiết lập một **hàm chi phí sai số (error-cost function)** để định lượng mức độ nghiêm trọng của False Positive (FP) so với False Negative (FN).
   * Khảo sát các ngưỡng ra quyết định (thresholds) trên tập Validation, báo cáo Precision, Recall, F1, Cost và chọn ra ngưỡng tối ưu làm cực tiểu hóa Cost.
   * Áp dụng ngưỡng cố định này lên tập Test: Báo cáo ma trận nhầm lẫn nhị phân, các chỉ số nhị phân, và ma trận nhầm lẫn đa lớp.
   * So sánh với một mô hình Baseline luôn dự đoán theo majority class của tập Train và thảo luận kết quả."""
    cells.append(nbf.v4.new_markdown_cell(c1_md))

    # Cell 2: Code - Data Loading & Splitting
    c2_code = """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# Cấu hình thẩm mỹ đồ thị
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 1. Tải và phân chia tập dữ liệu Wine (Giống Problem 1)
wine = load_wine(as_frame=True)
X_raw = wine.data.values
y_raw = wine.target.values
feature_names = wine.feature_names
target_names = wine.target_names

# Stratified Split: 70% Train, 15% Val, 15% Test
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(X_raw, y_raw, test_size=0.30, random_state=RANDOM_SEED, stratify=y_raw)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp_raw, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp)

# Chuẩn hóa Z-score (Fit trên Train set)
mean_train = np.mean(X_train_raw, axis=0)
std_train = np.std(X_train_raw, axis=0)
std_train[std_train == 0.0] = 1.0

X_train = (X_train_raw - mean_train) / std_train
X_val = (X_val_raw - mean_train) / std_train
X_test = (X_test_raw - mean_train) / std_train

print("[Hoàn tất] Dữ liệu đã được tải, phân chia phân tầng và chuẩn hóa chống rò rỉ (Zero Data Leakage).")"""
    cells.append(nbf.v4.new_code_cell(c2_code))

    # Cell 3: Markdown - Part A Theory
    c3_md = r"""## Phần A: So sánh các Mô hình Đa lớp (Multiclass Models)

**1. Tính công bằng trong so sánh (Fair Comparison Design):**
* Cả 2 mô hình sử dụng chung thuật toán tối ưu `solver='lbfgs'`, chung mức độ điều chuẩn ngược `C=1e4` (gần như không điều chuẩn để giống thuật toán gốc), và chung tập huấn luyện `X_train`.

**2. Khác biệt cơ bản giữa 2 mô hình:**
* **OneVsRestClassifier (OvR):** Chia bài toán $K=3$ lớp thành 3 bài toán nhị phân. Các xác suất nhị phân đầu ra độc lập với nhau, có thể dẫn đến việc tổng xác suất không bằng 1, đòi hỏi phải được chuẩn hóa (normalize) lại bằng một số thủ thuật ngầm (heuristic).
* **Multinomial Logistic Regression (Softmax):** Tối ưu hóa trực tiếp hàm mục tiêu Multinomial Cross-Entropy cho tất cả $K$ lớp cùng một lúc. Đầu ra sử dụng hàm Softmax $\frac{e^{z_k}}{\sum_j e^{z_j}}$ đảm bảo tổng xác suất của tất cả các lớp luôn bằng 1 một cách toán học chặt chẽ. Nó xem xét tương quan giữa các lớp, thường mang lại các giá trị xác suất (probabilities) đáng tin cậy và được hiệu chuẩn (calibrated) tốt hơn."""
    cells.append(nbf.v4.new_markdown_cell(c3_md))

    # Cell 4: Code - Part A Implementation
    c4_code = """# Mô hình 1: One-vs-Rest Logistic Regression
model_ovr = OneVsRestClassifier(LogisticRegression(solver='lbfgs', C=1e4, max_iter=2000, random_state=RANDOM_SEED))
model_ovr.fit(X_train, y_train)

y_val_pred_ovr = model_ovr.predict(X_val)
y_val_proba_ovr = model_ovr.predict_proba(X_val)

acc_ovr = accuracy_score(y_val, y_val_pred_ovr)
f1_ovr = f1_score(y_val, y_val_pred_ovr, average='macro')
loss_ovr = log_loss(y_val, y_val_proba_ovr)

# Mô hình 2: Multinomial Logistic Regression (Softmax)
model_multi = LogisticRegression(solver='lbfgs', C=1e4, max_iter=2000, random_state=RANDOM_SEED)
model_multi.fit(X_train, y_train)

y_val_pred_multi = model_multi.predict(X_val)
y_val_proba_multi = model_multi.predict_proba(X_val)

acc_multi = accuracy_score(y_val, y_val_pred_multi)
f1_multi = f1_score(y_val, y_val_pred_multi, average='macro')
loss_multi = log_loss(y_val, y_val_proba_multi)

# Bảng báo cáo chỉ số
print("=" * 70)
print("BẢNG SO SÁNH TRÊN TẬP VALIDATION")
print("=" * 70)
df_comp = pd.DataFrame({
    'Metric': ['Validation Accuracy', 'Validation Macro F1', 'Validation Log Loss'],
    'One-vs-Rest (OvR)': [f"{acc_ovr*100:.2f}%", f"{f1_ovr:.4f}", f"{loss_ovr:.4f}"],
    'Multinomial (Softmax)': [f"{acc_multi*100:.2f}%", f"{f1_multi:.4f}", f"{loss_multi:.4f}"]
})
print(df_comp.to_string(index=False))

# Kiểm tra xác suất trên MỘT mẫu cụ thể (mẫu số 5 trong tập Val)
sample_idx = 5
print("\\n" + "=" * 70)
print(f"XEM XÉT CHI TIẾT MẪU SỐ {sample_idx} TRONG TẬP VALIDATION")
print("=" * 70)
print(f"Nhãn thực tế (True Label): {target_names[y_val[sample_idx]]} (Class {y_val[sample_idx]})")
print("\\n[Mô hình OvR]")
print(f"Xác suất dự đoán: {y_val_proba_ovr[sample_idx]}")
print(f"Lớp dự đoán: Class {y_val_pred_ovr[sample_idx]}")
print("\\n[Mô hình Multinomial]")
print(f"Xác suất dự đoán: {y_val_proba_multi[sample_idx]}")
print(f"Lớp dự đoán: Class {y_val_pred_multi[sample_idx]}")"""
    cells.append(nbf.v4.new_code_cell(c4_code))

    # Cell 5: Markdown - Part B Theory
    c5_md = r"""## Phần B: Lựa chọn Ngưỡng quyết định (Choose a Threshold)

### 1. Định nghĩa Lớp Dương Tính (Positive Outcome)
* Ta chọn **Lớp 0 (Cultivar 0)** làm lớp dương tính (Positive), gộp Lớp 1 và Lớp 2 thành Lớp âm tính (Negative). 
* **Lưu ý:** Mô hình vẫn được huấn luyện là một mô hình đa lớp hoàn chỉnh. Tuy nhiên, khi khảo sát ngưỡng, ta chỉ tách riêng mảng xác suất dự đoán của Lớp 0 (cột 0) để so sánh với một ngưỡng (threshold) thay đổi từ $0.1 \to 0.9$.

### 2. Định nghĩa và Biện luận Hàm Chi Phí (Error-Cost Function)
* *Giả định kinh doanh:* Rượu vang Cultivar 0 là dòng rượu siêu cao cấp có giá trị thương mại lớn nhất.
* **False Positive (FP - Dương tính giả):** Mô hình dự đoán rượu thường (Class 1/2) là rượu cao cấp (Class 0).
  * *Hậu quả:* Khách hàng bỏ số tiền lớn mua phải hàng thường $\rightarrow$ Tức giận, tẩy chay, hãng mất uy tín nghiêm trọng. Thiệt hại rất cao. Đặt chi phí $C_{FP} = 5$.
* **False Negative (FN - Âm tính giả):** Mô hình dự đoán rượu cao cấp (Class 0) là rượu thường (Class 1/2).
  * *Hậu quả:* Hãng bán hớ giá trị thực, bị lỗ một khoản doanh thu, nhưng khách hàng không phàn nàn và uy tín không suy giảm. Thiệt hại thấp hơn. Đặt chi phí $C_{FN} = 1$.
* **Hàm Chi phí (Cost Criterion):** $\text{Total Cost} = 5 \times FP + 1 \times FN$
* Mục tiêu: Tìm một ngưỡng quyết định (threshold) sao cho giảm thiểu tối đa FP (tăng Precision) dẫu có phải hi sinh một chút FN (giảm Recall) để cực tiểu hóa Hàm Chi Phí này."""
    cells.append(nbf.v4.new_markdown_cell(c5_md))

    # Cell 6: Code - Part B Threshold Search
    c6_code = """# Trích xuất nhãn nhị phân và xác suất Lớp 0 cho tập Validation
y_val_binary = (y_val == 0).astype(int)
proba_val_class0 = model_multi.predict_proba(X_val)[:, 0]

thresholds = np.linspace(0.1, 0.9, 9)
results = []

best_cost = float('inf')
best_threshold = 0.5

for thresh in thresholds:
    y_pred_thresh = (proba_val_class0 >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val_binary, y_pred_thresh, labels=[0, 1]).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Cost = 5 * FP + 1 * FN
    cost = 5 * fp + 1 * fn
    
    results.append({
        'Threshold': thresh,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Cost': cost
    })
    
    if cost < best_cost:
        best_cost = cost
        best_threshold = thresh

df_thresh = pd.DataFrame(results)
print("=" * 70)
print("KHẢO SÁT CÁC NGƯỠNG TRÊN TẬP VALIDATION")
print("=" * 70)
print(df_thresh.to_string(index=False))

print(f"\\n=> Ngưỡng tối ưu được chọn: {best_threshold:.1f} (để cực tiểu hóa Chi phí = {best_cost})")

# Trực quan hóa
plt.figure(figsize=(10, 5))
plt.plot(df_thresh['Threshold'], df_thresh['Precision'], marker='o', label='Precision')
plt.plot(df_thresh['Threshold'], df_thresh['Recall'], marker='s', label='Recall')
plt.plot(df_thresh['Threshold'], df_thresh['F1 Score'], marker='^', label='F1 Score')
plt.plot(df_thresh['Threshold'], df_thresh['Cost']/10, marker='d', linestyle='--', label='Cost (scaled / 10)')
plt.axvline(x=best_threshold, color='red', linestyle=':', label=f'Best Threshold ({best_threshold:.1f})')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Trade-off giữa Threshold, Precision, Recall và Cost')
plt.legend()
plt.grid(True, alpha=0.4)
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/problem2_threshold_tradeoff.png', dpi=300)
plt.show()"""
    cells.append(nbf.v4.new_code_cell(c6_code))

    # Cell 7: Code - Part B Final Test Evaluation
    c7_code = """# ======================================================================
# ĐÁNH GIÁ TRÊN TẬP TEST VỚI NGƯỠNG TỐI ƯU ĐÃ CHỌN
# ======================================================================
y_test_binary = (y_test == 0).astype(int)
proba_test_class0 = model_multi.predict_proba(X_test)[:, 0]

# 1. Đánh giá Nhị phân với ngưỡng đã chọn (Binary metrics)
y_test_pred_binary = (proba_test_class0 >= best_threshold).astype(int)
tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test_binary, y_test_pred_binary, labels=[0, 1]).ravel()

prec_test = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
rec_test = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
f1_test = 2 * prec_test * rec_test / (prec_test + rec_test) if (prec_test + rec_test) > 0 else 0
acc_test = (tp_t + tn_t) / len(y_test_binary)
cost_test = 5 * fp_t + 1 * fn_t

print("=" * 70)
print(f"KẾT QUẢ ĐÁNH GIÁ NHỊ PHÂN TRÊN TẬP TEST (THRESHOLD = {best_threshold:.1f})")
print("=" * 70)
print(f"Accuracy : {acc_test*100:.2f}%")
print(f"Precision: {prec_test:.4f}")
print(f"Recall   : {rec_test:.4f}")
print(f"F1 Score : {f1_test:.4f}")
print(f"Total Cost: {cost_test}")
print("\\nConfusion Matrix (Binary):")
cm_bin_df = pd.DataFrame([[tn_t, fp_t], [fn_t, tp_t]], 
                         index=['Actual: Neg (Not 0)', 'Actual: Pos (Class 0)'], 
                         columns=['Predicted: Neg', 'Predicted: Pos'])
print(cm_bin_df)

# 2. Đánh giá Baseline (Majority Class của tập Train)
# Trên tập Train, class chiếm đa số là Lớp 1 (50 mẫu). Lớp 0 có 41 mẫu.
# Nếu một Baseline luôn dự đoán theo majority class của Train (Tức là luôn dự đoán Lớp 1), 
# thì trong ngữ cảnh nhị phân (Class 0 vs Rest), Baseline này sẽ LUÔN dự đoán âm tính (0).
dummy_clf = DummyClassifier(strategy='prior') # Tự động lấy lớp đa số
y_train_binary = (y_train == 0).astype(int)
dummy_clf.fit(X_train, y_train_binary)
y_test_dummy_pred = dummy_clf.predict(X_test)
acc_baseline = accuracy_score(y_test_binary, y_test_dummy_pred)
print(f"\\nTest Accuracy của Baseline Classifier (Luôn dự đoán Majority Class): {acc_baseline*100:.2f}%")

# 3. Đánh giá Toàn cảnh Đa Lớp (Full Multiclass Evaluation trên Test)
y_test_pred_multi_final = model_multi.predict(X_test)
multi_acc_test = accuracy_score(y_test, y_test_pred_multi_final)
multi_f1_test = f1_score(y_test, y_test_pred_multi_final, average='macro')

print("\\n" + "=" * 70)
print("KẾT QUẢ ĐÁNH GIÁ ĐA LỚP TRÊN TẬP TEST (FULL MULTICLASS)")
print("=" * 70)
print(f"Multiclass Accuracy : {multi_acc_test*100:.2f}%")
print(f"Multiclass Macro F1 : {multi_f1_test:.4f}")
print("\\nFull Multiclass Confusion Matrix:")
cm_multi = confusion_matrix(y_test, y_test_pred_multi_final)
cm_multi_df = pd.DataFrame(cm_multi, 
                           index=[f'Thực tế {c}' for c in target_names], 
                           columns=[f'Dự đoán {c}' for c in target_names])
print(cm_multi_df)"""
    cells.append(nbf.v4.new_code_cell(c7_code))

    # Cell 8: Markdown - Final Discussion
    c8_md = r"""## 3. Thảo luận và Trả lời câu hỏi (Discussion)

**1. Việc hạ thấp ngưỡng (lowering the threshold) thay đổi sai số như thế nào?**
* Khi hạ thấp ngưỡng (ví dụ từ $0.5$ xuống $0.1$), mô hình dễ dàng gán nhãn dương tính (Lớp 0) hơn cho dù độ tin cậy thấp.
* Điều này dẫn đến sự gia tăng mạnh mẽ **False Positives (Sai lầm loại I)** và sự suy giảm **False Negatives (Sai lầm loại II)**. Tóm lại, Recall sẽ tăng nhưng Precision sẽ giảm thê thảm.

**2. Việc thay đổi ngưỡng có làm thay đổi phân phối xác suất dự đoán (probabilities) không?**
* **Không.** Giá trị xác suất (probabilities) đầu ra được sinh ra hoàn toàn từ phương trình của mô hình $\sigma(x^T w + b)$. Thay đổi ngưỡng (threshold) chỉ tác động tới thao tác hậu xử lý (post-processing): quyết định cắt phân loại cuối cùng (nhãn $0$ hay $1$) chứ không làm thay đổi hay can thiệp vào bộ trọng số của mô hình cũng như giá trị xác suất thuần.

**3. Tại sao Accuracy của Majority-Class Baseline có thể gây hiểu lầm?**
* Nếu dữ liệu bị mất cân bằng trầm trọng (ví dụ: $90\%$ mẫu thuộc lớp âm tính, $10\%$ thuộc lớp dương tính).
* Một mô hình Baseline (nhắm mắt đánh lụi) luôn dự đoán mọi mẫu là lớp âm tính sẽ tự động đạt **Accuracy $90\%$**. Nhìn vào con số này dễ tưởng mô hình hoạt động tốt, nhưng thực chất **Recall = 0%**, nó bỏ lỡ hoàn toàn mục tiêu cốt lõi là phát hiện lớp dương tính. Do đó, các chỉ số như Precision, Recall, F1 và đặc biệt là phân tích hàm Chi phí (Cost matrix) là bắt buộc trong thực tiễn."""
    cells.append(nbf.v4.new_markdown_cell(c8_md))

    nb.cells = cells

    # Save unexecuted notebook first
    notebook_path = os.path.join("d:/PTIT/NAM4/KI1/Học máy", "Problem2_Wine_ModelComparison.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[+] Notebook created at: {notebook_path}")

    # Execute notebook using NotebookClient
    print("[*] Executing notebook to populate all outputs and figures...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    # Save executed notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[+] Notebook successfully executed and saved with outputs at: {notebook_path}")

if __name__ == "__main__":
    create_problem2_notebook()
