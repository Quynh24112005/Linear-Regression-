# Báo Cáo Thực Hành: Hồi Quy Tuyến Tính (Linear Regression)

**Học phần:** Học Máy (Machine Learning)  
**Sinh viên thực hiện:** Lê Như Quỳnh  
**Mã sinh viên:** B23DCCE081  
**Học viện:** Học viện Công nghệ Bưu chính Viễn thông (PTIT)  

---

## 📌 Tổng Quan Dự Án

Kho lưu trữ này chứa toàn bộ mã nguồn, báo cáo kỹ thuật và 2 notebook Jupyter hoàn chỉnh cho bài tập lớn môn **Học Máy — Chuyên đề Hồi quy tuyến tính (Linear Regression)**:
* **Problem 1 (40 điểm):** Hồi quy tuyến tính từ các nguyên lý cơ bản (*Linear Regression from First Principles*) trên tập dữ liệu **Online News Popularity**.
* **Problem 2 (60 điểm):** Mở rộng hàm cơ sở, Điều chuẩn và Chẩn đoán sai số (*Basis Expansion, Regularization, and Diagnostics*) trên tập dữ liệu **Bike Sharing Dataset**.

Tất cả các thuật toán cốt lõi đều được tự lập trình hoàn toàn từ đầu bằng thư viện **NumPy**, liên hệ chặt chẽ với cơ sở toán học (đạo hàm riêng, hàm mất mát bình phương, toán tử ngưỡng mềm) và được đối chứng độc lập với thư viện chuẩn mực `scikit-learn`.

---

## 📂 Cấu Trúc Kho Lưu Trữ

```text
Linear-Regression-/
├── README.md                                           # Báo cáo tổng hợp dự án
├── .gitignore                                          # Cấu hình bỏ qua dữ liệu lớn & file tạm
│
├── Problem1 - B23DCCE081 - Lê Như Quỳnh/
│   ├── Problem1_LinearRegression.ipynb                 # Jupyter Notebook hoàn chỉnh có outputs & biểu đồ
│   ├── Problem1_Report.md                              # Báo cáo kết quả và phân tích toán học
│   ├── problem1_scratch.py                             # Cài đặt Base, NormalEq, BGD, MBGD, SGD thuần NumPy
│   ├── dataset_loader.py                               # Tải, chuẩn hóa Z-score và phân chia dữ liệu chống rò rỉ
│   ├── problem1_main.py                                # Pipeline thực nghiệm và benchmark
│   └── plots/                                          # Đồ thị xuất bản độ phân giải cao
│       ├── p1_convergence_comparison.png               # Đồ thị so sánh tốc độ hội tụ BGD, MBGD, SGD
│       ├── p1_scaling_and_lr_impact.png                # Ảnh hưởng của chuẩn hóa đặc trưng & tốc độ học
│       ├── p1_benchmark_metrics.png                    # So sánh MSE, RMSE, MAE, R2 giữa các mô hình
│       ├── p1_residuals_diagnostics.png                # Đồ thị Residuals vs Fitted & Histogram thặng dư
│       └── p1_qq_plot.png                              # Normal Q-Q Plot kiểm định phân phối thặng dư
│
└── Problem2 - B23DCCE081 - Lê Như Quỳnh/
    ├── Problem2_BasisExpansion_Regularization.ipynb    # Jupyter Notebook hoàn chỉnh có outputs & biểu đồ
    ├── Problem2_Report.md                              # Báo cáo kết quả và phân tích chi tiết
    ├── basis_expansion.py                              # Mở rộng hàm cơ sở qua 3 cấp độ (Thô, Tuần hoàn, Đa thức)
    ├── problem2_scratch.py                             # Cài đặt OLS, Ridge, LASSO (Coordinate Descent) thuần NumPy
    ├── dataset_loader_p2.py                            # Tải, xóa bẫy rò rỉ và phân chia 2011/2012
    ├── problem2_main.py                                # Pipeline huấn luyện, 5-Fold CV và chẩn đoán
    └── plots/                                          # Đồ thị xuất bản độ phân giải cao
        ├── p2_cv_tuning_curves.png                     # Đường cong tìm alpha tối ưu qua 5-Fold CV
        ├── p2_regularization_paths.png                 # Đường dẫn co rút hệ số và tính thưa thớt của LASSO
        ├── p2_basis_complexity_comparison.png          # Bước nhảy hiệu năng R2 từ Level 0 lên Level 2
        ├── p2_residual_diagnostics.png                 # Khử sạch sai số thiên lệch chu kỳ 24h & Q-Q Plot
        └── p2_sparsity_qq_extra.png                    # Phân tích thưa thớt nâng cao
```

---

## 🔬 Problem 1: Hồi Quy Tuyến Tính Từ Các Nguyên Lý Cơ Bản

### 1. Định dạng bài toán & Chiến lược chuẩn bị dữ liệu
* **Tập dữ liệu:** [Online News Popularity (UCI)](https://archive.ics.uci.edu/dataset/332/online+news+popularity) gồm 39,644 bài báo và 58 đặc trưng định lượng.
* **Biến mục tiêu:** Dự đoán số lượt chia sẻ trên mạng xã hội (`shares`). Do phân phối của `shares` lệch phải cực độ (heavy right-skewed), ta áp dụng phép biến đổi:
  $$y = \ln(1 + \text{shares}) = \text{log1p}(\text{shares})$$
* **Ngăn chặn rò rỉ dữ liệu qua thời gian (Temporal Leakage):** Phân chia 80% Huấn luyện — 20% Kiểm thử theo đúng thứ tự thời gian xuất bản bài viết (Sequential Split). Dữ liệu quá khứ dự đoán tương lai, tuyệt đối không xáo trộn ngẫu nhiên.
* **Chuẩn hóa Z-score:** Bộ tham số $(\mu_{\text{train}}, \sigma_{\text{train}})$ được trích xuất duy nhất trên tập Huấn luyện.

### 2. Xây dựng 4 Thuật toán Tối ưu hóa từ First Principles
* **Mô hình có hệ số chặn (Intercept):** $\hat{y} = \bar{X} w$ với $\bar{X} = [\mathbf{1}, X]$.
* **Phần thặng dư (Residuals):** $e = \hat{y} - y = \bar{X} w - y$.
* **Hàm mất mát MSE:** $J(w) = \frac{1}{2N} e^T e$.
* **Gradient đạo hàm:** $\nabla_w J(w) = \frac{1}{N} \bar{X}^T e$.

| Thuật toán | Cơ chế cập nhật | Đặc tính hội tụ |
| :--- | :--- | :--- |
| **1. Normal Equation** | $w^* = (\bar{X}^T \bar{X})^\dagger \bar{X}^T y$ (Pseudoinverse) | Nghiệm giải tích chính xác ngay lập tức |
| **2. Batch GD** | $w \leftarrow w - \alpha \frac{1}{N} \bar{X}^T e$ | Đường cong loss đơn điệu, ổn định tuyệt đối |
| **3. Mini-Batch GD** | Cập nhật theo từng mini-batch kích thước $B = 256$ | Tối ưu hóa bộ nhớ đệm, tốc độ hội tụ nhanh nhất |
| **4. Stochastic GD** | Cập nhật trên từng mẫu $i$ kèm learning rate decay | Tiến nhanh về cực tiểu sau ít epoch |

### 3. Kết quả Thực nghiệm & Đối chứng Công bằng

| Mô hình / Thuật toán | MSE Kiểm thử | RMSE | MAE | $R^2$ Score | Thời gian (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal Equation (NumPy)** | **0.70994** | 0.84258 | 0.65584 | **0.12352** | 98.4 ms |
| **Batch GD ($\alpha=0.01, 300\text{ ep}$)** | 0.71820 | 0.84747 | 0.66012 | 0.11333 | 820.5 ms |
| **Mini-Batch GD ($B=256, 150\text{ ep}$)** | 0.71542 | 0.84583 | 0.65870 | 0.11676 | 450.2 ms |
| **Stochastic GD ($15\text{ ep}$)** | 0.72410 | 0.85094 | 0.66350 | 0.10605 | 2800.0 ms |
| **Scikit-Learn OLS (Benchmark)** | **0.70994** | 0.84258 | 0.65584 | **0.12352** | 42.1 ms |

> ⚖️ **Bằng chứng Tính Công bằng & Phân tích Đa cộng tuyến:**  
> Hệ số điều kiện của ma trận hiệp phương sai là $\kappa(X^T X) \approx 3.79 \times 10^{16}$, cho thấy dữ liệu bị đa cộng tuyến nghiêm trọng. Tồn tại vô số vector nghiệm tối ưu trong không gian null. Thuật toán SVD Pseudoinverse tự cài đặt và bộ giải LAPACK của Scikit-Learn chọn các vector cơ sở khác nhau, nhưng sai số dự báo MSE trên tập kiểm thử **trùng khớp chính xác ở cấp độ $10^{-13}$**, chứng minh tính tin cậy tuyệt đối của thuật toán tự viết.

---

## 🚴 Problem 2: Mở Rộng Hàm Cơ Sở, Điều Chuẩn và Chẩn Đoán

### 1. Định dạng bài toán & Triệt tiêu Cạm bẫy Rò rỉ Dữ liệu
* **Tập dữ liệu:** [Bike Sharing Dataset (UCI)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) (`hour.csv`) gồm 17,379 giờ thuê xe đạp.
* **Mục tiêu:** Dự đoán lượng thuê xe theo giờ (`cnt`) sau khi làm mịn bằng $\ln(1 + \text{cnt})$.
* **Triệt tiêu Cạm bẫy Rò rỉ Dữ liệu Cốt tử (Leakage Trap):** Loại bỏ hoàn toàn 2 cột `casual` và `registered` vì $\text{casual} + \text{registered} = \text{cnt}$ 100%. Nếu giữ lại, mô hình đạt $R^2 = 1.0$ tầm thường và hoàn toàn vô dụng khi triển khai dự báo tương lai.
* **Phân chia theo thời gian:** Dữ liệu năm **2011 (8,645 mẫu)** làm tập Huấn luyện; năm **2012 (8,734 mẫu)** làm tập Kiểm thử.

### 2. Mở rộng Hàm cơ sở (Basis Expansion) 3 Cấp độ
* **Cấp độ 0 (Dữ liệu thô - 11 biến):** Giữ nguyên các biến thời tiết và lịch biểu thô.
* **Cấp độ 1 (Hàm cơ sở tuần hoàn Fourier - 19 biến):** Áp dụng $\sin, \cos$ cho `hr` (chu kỳ 24h và sóng hài 12h cho 2 đỉnh cao điểm đi làm 8h & 17h), `mnth` (12 tháng), `weekday` (7 ngày). Xóa bỏ bước đứt gãy phi lý giữa 23h và 0h.
* **Cấp độ 2 (Tuần hoàn + Đa thức + Tương tác chéo - 32 biến):** Bổ sung đa thức bậc 2, bậc 3 thời tiết ($\text{temp}^2, \text{hum}^2, \text{wind}^2, \text{temp}^3$), tương tác thời tiết ($\text{temp} \times \text{hum}$), và tương tác nghiệp vụ $\text{hr}_{\sin, \cos} \times \text{workingday}$.

### 3. Tinh chỉnh Siêu tham số & So sánh Hiệu năng

* **5-Fold Cross-Validation:** Được tiến hành nghiêm ngặt chỉ trên tập Train (2011) để tìm ra hệ số phạt tối ưu $\alpha^* = 0.0001$ cho cả Ridge và LASSO (Hạ tọa độ - Coordinate Descent).
* **Kết quả Đánh giá Độc lập trên Tập Kiểm thử (Năm 2012):**

| Cấp độ / Mô hình | MSE | RMSE | MAE | $R^2$ Score | Sparsity (Biến = 0) | Thời gian (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. OLS Cấp độ 0 (Thô - 11 biến)** | 1.09635 | 1.04707 | 0.81745 | **0.37009** | 0 / 11 | 4.5 ms |
| **2. OLS Cấp độ 1 (Tuần hoàn - 19 biến)** | 0.58912 | 0.76754 | 0.58930 | **0.66152** | 0 / 19 | 8.2 ms |
| **3. OLS Cấp độ 2 (Đa thức + Tương tác - 32 biến)**| 0.43825 | 0.66199 | 0.49980 | **0.74820** | 0 / 32 | 14.1 ms |
| **4. Ridge Cấp độ 2 ($\alpha^* = 0.0001$)** | 0.43780 | 0.66166 | 0.49920 | **0.74850** | 0 / 32 | 12.0 ms |
| **5. LASSO Cấp độ 2 ($\alpha^* = 0.0001$)** | **0.43760** | **0.66151** | **0.49890** | **0.74860** | 0 / 32 | 185.0 ms |

> 🚀 **Bước nhảy Hiệu năng Vượt bậc:**  
> Nhờ việc áp dụng hàm cơ sở tuần hoàn Fourier và tương tác giờ làm việc, chỉ số $R^2$ tăng vọt từ **37.01%** lên **74.86%**, giảm hơn 60% sai số MSE.  
> 
> 🎯 **Triệt tiêu Lỗi Thiên lệch Chu kỳ (Cyclical Bias Elimination):**  
> Ở Cấp độ 0, mô hình tuyến tính thẳng đánh giá thiếu trầm trọng vào 8h sáng và 17h chiều, đồng thời đánh giá thừa vào ban đêm. Khi lên Cấp độ 2, thặng dư trung bình theo giờ được san phẳng hoàn toàn về mức 0 trên suốt 24 giờ trong ngày.

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Thực Nghiệm

### 1. Sao chép kho lưu trữ
```bash
git clone https://github.com/Quynh24112005/Linear-Regression-.git
cd Linear-Regression-
```

### 2. Cài đặt các thư viện cần thiết
```bash
pip install numpy pandas matplotlib scikit-learn scipy jupyter
```

### 3. Mở và khám phá các Notebook
Bạn có thể mở trực tiếp trong VS Code hoặc khởi chạy Jupyter Lab / Notebook:
```bash
jupyter notebook
```
* Mở `Problem1 - B23DCCE081 - Lê Như Quỳnh/Problem1_LinearRegression.ipynb` để xem bài toán 1.
* Mở `Problem2 - B23DCCE081 - Lê Như Quỳnh/Problem2_BasisExpansion_Regularization.ipynb` để xem bài toán 2.

Hoặc chạy các pipeline bằng dòng lệnh:
```bash
# Chạy thực nghiệm Problem 1
python "Problem1 - B23DCCE081 - Lê Như Quỳnh/problem1_main.py"

# Chạy thực nghiệm Problem 2
python "Problem2 - B23DCCE081 - Lê Như Quỳnh/problem2_main.py"
```

---

## 👤 Thông Tin Tác Giả
* **Họ và tên:** Lê Như Quỳnh
* **Mã sinh viên:** B23DCCE081
* **Lớp:** D23CQCN01-B
* **Khoa:** Công nghệ Thông tin — Học viện Công nghệ Bưu chính Viễn thông (PTIT)
