# Báo Cáo Kết Quả - Bài toán 2: Mở rộng hàm cơ sở, Điều chuẩn và Chẩn đoán

**Sinh viên:** Lê Như Quỳnh - B23DCCE081
**Học phần:** Học Máy (Machine Learning)

---

## 1. Định dạng bài toán và Dữ liệu
* **Tập dữ liệu:** Dữ liệu chia sẻ xe đạp (Dự đoán số lượng xe đạp được thuê theo giờ).
* **Mục tiêu:** Dự đoán lượng xe được thuê dựa trên các yếu tố thời tiết và thời gian. Đã áp dụng phép biến đổi logarit để đối phó với sự phân bố không đồng đều của dữ liệu lượng thuê.
* **Ngăn chặn rò rỉ dữ liệu:** Tập dữ liệu gốc cung cấp số lượng khách vãng lai và khách đăng ký mà tổng của chúng chính xác bằng tổng số xe được thuê. Đã tiến hành xóa hai cột này để ngăn chặn bẫy rò rỉ dữ liệu.
* **Phân chia dữ liệu:** Phân chia theo thời gian, sử dụng dữ liệu năm 2011 để làm tập Huấn luyện, và năm 2012 làm tập Kiểm thử để đánh giá khả năng tổng quát hóa của mô hình theo thời gian.

## 2. Mở rộng Đặc trưng (Hàm cơ sở)
Do các biến thời gian (giờ trong ngày, tháng trong năm) có tính chất tuần hoàn khép kín, một mô hình tuyến tính thẳng đơn giản (Cấp độ 0) sẽ thất bại trong việc nắm bắt bản chất này (ví dụ: 23h đêm và 0h sáng rất gần nhau về mặt thời gian nhưng giá trị số học lại cách xa nhau).
Mô hình đã được mở rộng đặc trưng qua 3 cấp độ:
* **Cấp độ 0 (Dữ liệu thô):** 11 đặc trưng gốc.
* **Cấp độ 1 (Tuần hoàn):** Dùng các hàm lượng giác biến đổi không gian (sin và cos) cho `giờ`, `tháng`, `thứ`. Số lượng đặc trưng tăng lên 19.
* **Cấp độ 2 (Tuần hoàn + Đa thức + Tương tác chéo):** Bổ sung thêm đặc trưng đa thức (bình phương, lập phương của nhiệt độ, sức gió) và các đặc trưng tương tác chéo (tương tác giữa giờ với ngày làm việc, nhiệt độ với thời tiết). Số đặc trưng là 32.

## 3. Điều chuẩn và Đánh giá chéo
Sử dụng 3 mô hình: **Bình phương tối thiểu thông thường (OLS)**, **Hồi quy Ridge**, và **Hồi quy LASSO** (tự cài đặt từ đầu bằng thuật toán Hạ tọa độ - Coordinate Descent).
* **Tránh rò rỉ dữ liệu kiểm thử trong quá trình tinh chỉnh:** Sử dụng phương pháp Đánh giá chéo 5 thư mục (5-Fold Cross-Validation) nghiêm ngặt *chỉ áp dụng trên tập Huấn luyện* để vẽ ra đường cong sai số và tìm siêu tham số tinh chỉnh ($\alpha$) tối ưu nhất.
* Kết quả đánh giá chéo chỉ ra $\alpha^* = 0.0001$ là mức tốt nhất cho cả mô hình Ridge và LASSO.
*(Tham khảo đường cong đánh giá chéo tại: `plots/p2_cv_tuning_curves.png`)*

## 4. Chẩn đoán & Vai trò của quá trình Điều chuẩn
**Kết quả trên tập Kiểm thử:**
* OLS Cấp độ 0: R² = 0.3701 (Hiệu năng kém, biểu đồ thặng dư trung bình theo giờ cho thấy sự thiên lệch chu kỳ rất nghiêm trọng).
* OLS Cấp độ 2: R² = 0.7482 (Khử sạch hoàn toàn lỗi thiên lệch chu kỳ, thể hiện việc mở rộng đặc trưng bằng hàm lượng giác đã hoạt động cực kỳ hoàn hảo).
* Ridge Cấp độ 2 ($\alpha=0.0001$): R² = 0.7485.
* LASSO Cấp độ 2 ($\alpha=0.0001$): R² = 0.7486 (Tốt nhất).

**Bàn luận về tính thưa thớt (Sparsity) và Điều chuẩn:**
Tại mức $\alpha$ tối ưu ($0.0001$), cả Ridge, LASSO và OLS đều cho kết quả gần y hệt nhau, và LASSO quyết định giữ lại toàn bộ 32/32 biến (không tạo ra tính thưa thớt). 
**Lý do:** Tỉ số giữa số đặc trưng trên số lượng mẫu ($32/8645 \approx 0.0037$) là quá nhỏ. Khối lượng dữ liệu là rất dồi dào để mô hình tuyến tính thông thường không bị học vẹt (Overfitting) ngay cả ở Cấp độ 2. Do đó, việc áp dụng mức phạt Điều chuẩn lớn là không thực sự cần thiết để đạt cấu hình dự báo tốt nhất.
Tuy nhiên, khi thực nghiệm ép hệ số phạt $\alpha$ của LASSO lên mức cao (ví dụ $\alpha=0.5$), cơ chế lựa chọn đặc trưng lập tức kích hoạt, loại bỏ chính xác 29/32 biến về 0. Điều này minh chứng thuật toán LASSO đã được cài đặt đúng và hoạt động vô cùng chuẩn xác.
*(Tham khảo sự co rút hệ số tại: `plots/p2_regularization_paths.png`)*

## 5. Kết luận
* Việc kết hợp **Mở rộng đặc trưng tuần hoàn** là bước ngoặt quyết định sự thành bại của hồi quy tuyến tính trong bộ dữ liệu mang tính chu kỳ (cải thiện chỉ số R² từ 37% lên xấp xỉ 75%).
* Đa số các giả định về thặng dư đều thỏa mãn. Tuy ở các giá trị dự báo quá cao, phương sai có dấu hiệu hẹp lại đôi chút, điều này phản ánh những giới hạn cố hữu của cấu trúc mô hình cộng tính.
*(Tham khảo đồ thị sai số: `plots/p2_residual_diagnostics.png`)*
