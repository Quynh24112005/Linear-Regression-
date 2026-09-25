# Báo Cáo Thực Hành: Hồi Quy Tuyến Tính (Linear Regression)

**Học phần:** Học Máy  
**Sinh viên thực hiện:** Lê Như Quỳnh  
**Mã sinh viên:** B23DCCE081  
**Học viện:** Học viện Công nghệ Bưu chính Viễn thông (PTIT)  

---

## Tổng quan dự án

Repository này chứa mã nguồn, báo cáo kỹ thuật và 2 notebook Jupyter cho bài tập môn Học Máy về chủ đề Hồi quy tuyến tính:
* **Problem 1:** Hồi quy tuyến tính từ các nguyên lý cơ bản trên tập dữ liệu **Online News Popularity**.
* **Problem 2:** Mở rộng hàm cơ sở, Điều chuẩn và Chẩn đoán sai số trên tập dữ liệu **Bike Sharing Dataset**.

Toàn bộ các thuật toán cốt lõi được xây dựng bằng thư viện **NumPy**, tính toán chi tiết phần dư, hàm mất mát và đạo hàm, sau đó được đối chứng với thư viện `scikit-learn`.

---

## Cấu trúc thư mục

```text
Linear-Regression-/
├── README.md                                           # Báo cáo tổng hợp dự án
├── .gitignore                                          # Cấu hình bỏ qua dữ liệu lớn & file tạm
│
├── Problem1 - B23DCCE081 - Lê Như Quỳnh/
│   ├── Problem1_LinearRegression.ipynb                 # Jupyter Notebook hoàn chỉnh có outputs & biểu đồ
│   ├── Problem1_Report.md                              # Báo cáo kết quả và phân tích chi tiết
│   ├── problem1_scratch.py                             # Cài đặt Base, NormalEq, BGD, MBGD, SGD bằng NumPy
│   ├── dataset_loader.py                               # Tải, tiền xử lý và chuẩn hóa dữ liệu
│   ├── problem1_main.py                                # Pipeline huấn luyện và đánh giá
│   └── plots/                                          # Các biểu đồ kết quả
│       ├── p1_convergence_comparison.png               # So sánh tốc độ hội tụ BGD, MBGD, SGD
│       ├── p1_scaling_and_lr_impact.png                # Ảnh hưởng của chuẩn hóa đặc trưng & learning rate
│       ├── p1_benchmark_metrics.png                    # So sánh các chỉ số MSE, RMSE, MAE, R2
│       ├── p1_residuals_diagnostics.png                # Đồ thị Residuals vs Fitted & Histogram phần dư
│       └── p1_qq_plot.png                              # Normal Q-Q Plot kiểm định phân phối phần dư
│
└── Problem2 - B23DCCE081 - Lê Như Quỳnh/
    ├── Problem2_BasisExpansion_Regularization.ipynb    # Jupyter Notebook hoàn chỉnh có outputs & biểu đồ
    ├── Problem2_Report.md                              # Báo cáo kết quả và phân tích chi tiết
    ├── basis_expansion.py                              # Mở rộng hàm cơ sở qua 3 cấp độ
    ├── problem2_scratch.py                             # Cài đặt OLS, Ridge, Lasso (Coordinate Descent)
    ├── dataset_loader_p2.py                            # Tải và tiền xử lý dữ liệu Bike Sharing
    ├── problem2_main.py                                # Pipeline huấn luyện, 5-Fold CV và chẩn đoán
    └── plots/                                          # Các biểu đồ kết quả
        ├── p2_cv_tuning_curves.png                     # Đường cong tìm alpha tối ưu qua 5-Fold CV
        ├── p2_regularization_paths.png                 # Đường đi tham số và tính thưa thớt của Lasso
        ├── p2_basis_complexity_comparison.png          # So sánh hiệu năng giữa các cấp độ mở rộng
        ├── p2_residual_diagnostics.png                 # Phân tích phần dư theo giờ và Q-Q Plot
        └── p2_sparsity_qq_extra.png                    # Phân tích tính thưa thớt
```

---

## Problem 1: Hồi quy tuyến tính từ các nguyên lý cơ bản

### 1. Định dạng bài toán và Chuẩn bị dữ liệu
* **Tập dữ liệu:** Online News Popularity (39,644 bài báo, 58 đặc trưng).
* **Mục tiêu:** Dự đoán số lượt chia sẻ của bài viết (`shares`).
* **Biến đổi logarit:** Biến `shares` có phân phối lệch phải mạnh nên áp dụng biến đổi $y = \ln(1 + \text{shares})$ để ổn định phương sai.
* **Chia dữ liệu:** Chia 80% Train và 20% Test theo thứ tự thời gian xuất bản để mô phỏng bài toán thực tế (dùng dữ liệu quá khứ dự đoán tương lai).
* **Chuẩn hóa đặc trưng:** Chuẩn hóa Z-score với mean và std tính trên tập Train.

### 2. Các thuật toán tối ưu hóa tự cài đặt
* Mô hình có hệ số chặn: $\hat{y} = \bar{X} w$ với $\bar{X} = [\mathbf{1}, X]$.
* Phần dư: $e = \bar{X} w - y$.
* Hàm mất mát MSE: $J(w) = \frac{1}{2N} e^T e$.
* Gradient: $\nabla_w J(w) = \frac{1}{N} \bar{X}^T e$.

Cài đặt 4 phương pháp tối ưu:
1. **Normal Equation:** $w^* = (\bar{X}^T \bar{X})^\dagger \bar{X}^T y$ (dùng giả nghịch đảo Moore-Penrose).
2. **Batch Gradient Descent (BGD):** Cập nhật theo gradient trên toàn bộ tập dữ liệu.
3. **Mini-Batch Gradient Descent (MBGD):** Cập nhật theo từng batch kích thước $B = 256$.
4. **Stochastic Gradient Descent (SGD):** Cập nhật theo từng mẫu ngẫu nhiên có giảm dần tốc độ học.

### 3. Kết quả thực nghiệm trên tập Test

| Mô hình / Thuật toán | MSE | RMSE | MAE | R2 Score | Thời gian (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal Equation (NumPy)** | **0.70994** | 0.84258 | 0.65584 | **0.12352** | 98.4 ms |
| **Batch GD (lr=0.01, 300 ep)** | 0.71820 | 0.84747 | 0.66012 | 0.11333 | 820.5 ms |
| **Mini-Batch GD (B=256, 150 ep)** | 0.71542 | 0.84583 | 0.65870 | 0.11676 | 450.2 ms |
| **Stochastic GD (15 ep)** | 0.72410 | 0.85094 | 0.66350 | 0.10605 | 2800.0 ms |
| **Scikit-Learn OLS (Benchmark)** | **0.70994** | 0.84258 | 0.65584 | **0.12352** | 42.1 ms |

* **Đa cộng tuyến:** Ma trận $X^T X$ có condition number $\kappa \approx 3.79 \times 10^{16}$, cho thấy các đặc trưng có độ tương quan cao. Thuật toán tự viết và scikit-learn chọn nghiệm khác nhau trong không gian null nhưng sai số MSE trên tập test trùng khớp ở mức $10^{-13}$.
* **Chỉ số R2 ~ 0.12:** Bài toán dự đoán độ lan truyền bài báo phụ thuộc nhiều vào yếu tố xã hội và nội dung ngữ nghĩa nên mô hình tuyến tính với các đặc trưng thống kê đạt mức $R^2 \approx 0.12$ là phù hợp với thực tế.

---

## Problem 2: Mở rộng hàm cơ sở, Điều chuẩn và Chẩn đoán

### 1. Định dạng bài toán và Tiền xử lý
* **Tập dữ liệu:** Bike Sharing Dataset (`hour.csv`, 17,379 giờ).
* **Mục tiêu:** Dự đoán lượng thuê xe theo giờ (`cnt`), áp dụng biến đổi $\ln(1 + \text{cnt})$.
* **Xử lý rò rỉ dữ liệu:** Loại bỏ 2 cột `casual` và `registered` vì tổng của chúng bằng đúng `cnt` ($\text{casual} + \text{registered} = \text{cnt}$).
* **Chia dữ liệu:** Năm 2011 (8,645 mẫu) làm tập Train, năm 2012 (8,734 mẫu) làm tập Test.

### 2. Mở rộng hàm cơ sở qua 3 cấp độ
* **Cấp độ 0 (11 biến):** Dữ liệu thô ban đầu.
* **Cấp độ 1 (19 biến):** Thêm biến đổi Fourier $\sin, \cos$ cho giờ (chu kỳ 24h và chu kỳ 12h cho 2 đỉnh đi làm), tháng (chu kỳ 12) và thứ (chu kỳ 7).
* **Cấp độ 2 (32 biến):** Bổ sung đa thức thời tiết ($\text{temp}^2, \text{hum}^2, \text{wind}^2, \text{temp}^3$), tương tác thời tiết ($\text{temp} \times \text{hum}$) và tương tác giờ $\times$ ngày làm việc.

### 3. Kết quả đánh giá trên tập Test (Năm 2012)

* Siêu tham số $\alpha^* = 0.0001$ được lựa chọn thông qua 5-fold cross-validation trên tập Train.

| Cấp độ / Mô hình | MSE | RMSE | MAE | R2 Score | Sparsity (Biến = 0) | Thời gian (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **OLS Level 0 (Thô - 11 biến)** | 1.09635 | 1.04707 | 0.81745 | **0.37009** | 0 / 11 | 4.5 ms |
| **OLS Level 1 (Tuần hoàn - 19 biến)** | 0.58912 | 0.76754 | 0.58930 | **0.66152** | 0 / 19 | 8.2 ms |
| **OLS Level 2 (Đa thức + Tương tác - 32 biến)**| 0.43825 | 0.66199 | 0.49980 | **0.74820** | 0 / 32 | 14.1 ms |
| **Ridge Level 2 ($\alpha^* = 0.0001$)** | 0.43780 | 0.66166 | 0.49920 | **0.74850** | 0 / 32 | 12.0 ms |
| **Lasso Level 2 ($\alpha^* = 0.0001$)** | **0.43760** | **0.66151** | **0.49890** | **0.74860** | 0 / 32 | 185.0 ms |

* **Hiệu quả của hàm cơ sở tuần hoàn:** Chỉ số $R^2$ tăng từ 37.01% lên 74.86%, giảm hơn một nửa sai số MSE.
* **Khử sai số chu kỳ:** Ở Cấp độ 0, mô hình dự đoán thiếu vào các giờ cao điểm (8h, 17h) và dự đoán thừa vào ban đêm. Khi lên Cấp độ 2, phần dư trung bình theo giờ được san phẳng về mức 0 trên suốt 24 giờ.

---

## Hướng dẫn cài đặt và chạy thực nghiệm

### 1. Clone repository
```bash
git clone https://github.com/Quynh24112005/Linear-Regression-.git
cd Linear-Regression-
```

### 2. Cài đặt môi trường
```bash
pip install numpy pandas matplotlib scikit-learn scipy jupyter
```

### 3. Mở Notebook
Khởi chạy Jupyter Notebook hoặc mở trực tiếp trên VS Code:
```bash
jupyter notebook
```
* Xem bài toán 1: `Problem1 - B23DCCE081 - Lê Như Quỳnh/Problem1_LinearRegression.ipynb`
* Xem bài toán 2: `Problem2 - B23DCCE081 - Lê Như Quỳnh/Problem2_BasisExpansion_Regularization.ipynb`

### 4. Chạy file script
```bash
python "Problem1 - B23DCCE081 - Lê Như Quỳnh/problem1_main.py"
python "Problem2 - B23DCCE081 - Lê Như Quỳnh/problem2_main.py"
```

---

## Thông tin sinh viên
* **Họ và tên:** Lê Như Quỳnh
* **Mã sinh viên:** B23DCCE081
* **Lớp:** D23CQCN01-B
* **Khoa:** Công nghệ Thông tin — Học viện Công nghệ Bưu chính Viễn thông
