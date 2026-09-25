# Báo Cáo Kết Quả - Bài toán 1: Hồi quy tuyến tính từ các nguyên lý cơ bản

**Sinh viên:** Lê Như Quỳnh - B23DCCE081
**Học phần:** Học Máy (Machine Learning)

---

## 1. Định dạng bài toán
* **Tập dữ liệu:** Mức độ phổ biến của tin tức trực tuyến (Dự đoán mức độ lan truyền của bài viết).
* **Mục tiêu dự đoán:** Dự đoán số lượt chia sẻ của một bài báo dựa trên 58 thuộc tính như độ dài từ khóa, số lượng liên kết, thông tin tác giả, v.v.
* **Đặc điểm mục tiêu:** Số lượt chia sẻ có phân phối lệch chuẩn rất nặng với phần đuôi dài. Do đó, mục tiêu dự đoán đã được áp dụng phép biến đổi logarit tự nhiên `y = log(1 + số lượt chia sẻ)` để làm mịn phân phối, giúp mô hình dễ học hơn.
* **Phân chia dữ liệu:** Do thời gian phát hành bài báo ảnh hưởng đến lượt chia sẻ, dữ liệu được giữ nguyên thứ tự ban đầu và chia theo tỷ lệ Huấn luyện/Kiểm thử là 80/20 mà không xáo trộn ngẫu nhiên. Điều này mô phỏng thực tế: dùng dữ liệu quá khứ để dự báo tương lai.

## 2. Xây dựng mô hình tuyến tính từ nguyên lý cơ bản
Thay vì dùng thư viện có sẵn, bài tập đã tự xây dựng 4 phương pháp hồi quy tuyến tính từ đầu (viết mã thuần) trong tệp `problem1_scratch.py`:
1. **Phương trình chuẩn (Nghiệm đại số chính xác):** Giải trực tiếp bằng phương trình chuẩn kết hợp với Giả nghịch đảo (Moore-Penrose Pseudoinverse) để xử lý hiện tượng đa cộng tuyến.
2. **Hạ độ dốc theo lô (Batch Gradient Descent):** Cập nhật trọng số bằng đạo hàm tính trên toàn bộ tập dữ liệu.
3. **Hạ độ dốc theo lô nhỏ (Mini-Batch Gradient Descent):** Chia dữ liệu thành các lô nhỏ (kích thước 256) để cập nhật, cân bằng giữa tốc độ tính toán và sự ổn định.
4. **Hạ độ dốc ngẫu nhiên (Stochastic Gradient Descent):** Cập nhật trên từng mẫu ngẫu nhiên, kết hợp phương pháp giảm dần tốc độ học.

Các thành phần cốt lõi như phần thặng dư (sai số), hàm mất mát bình phương trung bình (MSE), đạo hàm và các bước cập nhật tham số đều được tính toán bằng ma trận Numpy bám sát hoàn toàn vào lý thuyết toán học.

## 3. Khảo sát thực nghiệm

### 3.1 Ảnh hưởng của việc chuẩn hóa dữ liệu (Co giãn đặc trưng)
Mô hình tuyến tính rất nhạy cảm với thang đo của đặc trưng. Thực nghiệm so sánh giữa quá trình Hạ độ dốc theo lô trên dữ liệu gốc và dữ liệu đã chuẩn hóa cho thấy:
* Nếu không chuẩn hóa: Bắt buộc phải dùng tốc độ học cực kỳ nhỏ ($10^{-12}$) nếu không đạo hàm sẽ bị bùng nổ. Quá trình hội tụ diễn ra vô cùng chậm chạp.
* Khi có chuẩn hóa: Có thể tự tin dùng tốc độ học lớn ($0.05$) giúp hàm mất mát giảm dốc đứng ngay từ những bước lặp đầu tiên.
*(Tham khảo: `plots/p1_scaling_and_lr_impact.png`)*

### 3.2 Lựa chọn phương pháp tối ưu hóa
* **Theo lô:** Mượt mà, ổn định nhất nhưng khối lượng tính toán nặng nề cho mỗi bước lặp.
* **Ngẫu nhiên:** Nhiễu nhiều, dao động mạnh nhưng tiến về vùng cực tiểu rất nhanh.
* **Lô nhỏ:** Lựa chọn tốt nhất trong thực tế, hội tụ nhanh và ít nhiễu hơn so với phương pháp ngẫu nhiên.
* **Độ nhạy của Tốc độ học:** Khi tốc độ học quá lớn (0.5), hàm mất mát lập tức bùng nổ lên vô cùng. Khi ở mức 0.05 và 0.1, mô hình hội tụ rất ổn định.

## 4. Kiểm chứng và đánh giá công bằng
Mô hình tự viết bằng Phương trình chuẩn được đối chiếu với mô hình chuẩn của thư viện `scikit-learn`. 
**Phân tích sự khác biệt:**
* **Đa cộng tuyến:** Ma trận hiệp phương sai có Hệ số điều kiện (Condition Number) rất lớn ($\approx 3.79 \times 10^{16}$), cho thấy dữ liệu bị đa cộng tuyến nghiêm trọng.
* Do đó, phương trình chuẩn có vô số nghiệm trọng số cho ra cùng một mức sai số thấp nhất. Hàm tự viết và hàm của thư viện chọn 2 vector nghiệm khác nhau (trong không gian null) dẫn đến khoảng cách giữa 2 vector trọng số khá lớn.
* **Bằng chứng công bằng:** Mặc dù trọng số khác nhau, nhưng sai số dự báo (MSE) trên tập kiểm thử của cả hai mô hình lại trùng khớp chính xác lên đến $10^{-13}$. Điều này chứng minh mô hình tự viết đã hoàn toàn chính xác và việc so sánh qua sai số dự báo là hoàn toàn công bằng.

## 5. Chẩn đoán thặng dư và Hạn chế
* **Kết quả đo lường:** MSE $\approx 0.7099$, Hệ số xác định (R²) $\approx 0.12$. 
* **Hạn chế:** Chỉ số R² thấp (12%) không phải do mã nguồn sai, mà do bản chất của bài toán "Dự đoán mức độ lan truyền của bài viết" là cực kỳ phức tạp và phi tuyến. Nó phụ thuộc vào ngữ nghĩa, sự kiện xã hội - những thứ mà 58 đặc trưng định lượng đơn giản không thể phản ánh hết được.
* **Chẩn đoán thặng dư:** Đồ thị Q-Q và phân bố thặng dư cho thấy sai số xấp xỉ phân phối chuẩn, chứng tỏ mô hình tuyến tính đã khai thác tối đa giới hạn của nó trên bộ dữ liệu này.
*(Tham khảo: `plots/p1_residuals_diagnostics.png` và `plots/p1_qq_plot.png`)*
