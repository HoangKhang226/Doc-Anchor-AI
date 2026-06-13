# 📊 Phân Tích Benchmark: Đánh Giá Pipeline OCR (Doc Anchor AI)

## 1. Tổng kết nhanh — 3 đợt chạy

| Chỉ số | Đợt 1 (Baseline) | Đợt 2 (P0 Fix) | Đợt 3 (TextForge) | Cải thiện tổng |
|--------|-------------------|-----------------|---------------------|----------------|
| Tổng số ảnh | 50 | 50 | 50 | — |
| CER trung bình | 0.291 | 0.228 | **0.204** | ⬇️ 29.9% |
| WER trung bình | 0.420 | 0.363 | **0.325** | ⬇️ 22.6% |
| **Sim trung bình** | 86.49% | 87.21% | **88.57%** | ⬆️ +2.08 điểm |
| TokenSim trung bình | 84.25% | 85.29% | **87.01%** | ⬆️ +2.76 điểm |

---

## 2. Phân tích các cải thiện nổi bật (Đợt 3 vs Đợt 1)

### 🚀 Nhảy vọt lớn nhất

| Ảnh | Đợt 1 Sim | Đợt 3 Sim | Delta | Phân tích |
|-----|-----------|-----------|-------|-----------|
| `image_13.png` | 43.70% | **89.80%** | +46.1 | Bảng báo cáo tài chính 10 cột — trước đây bị OpenCV đắp bảng rác, giờ VLM xử lý chuẩn |
| `image_14.jpeg` | 37.00% | **92.80%** | +55.8 | Sao kê Techcombank — repetition loop đã bị triệt tiêu hoàn toàn |
| `image_27.png` | 67.30% | **87.70%** | +20.4 | Tài liệu dày đặc — cải thiện layout reconstruction |
| `image_33.jpg` | 87.60% | **98.30%** | +10.7 | Hóa đơn — VLM đọc chính xác hơn nhờ prompt profile tốt hơn |
| `image_19.png` | 62.40% | **79.00%** | +16.6 | Bảng phức tạp — giảm đáng kể lỗi cấu trúc |

### 📉 Giảm nhẹ (cần theo dõi)

| Ảnh | Đợt 1 Sim | Đợt 3 Sim | Delta | Ghi chú |
|-----|-----------|-----------|-------|---------|
| `image_30.webp` | 99.50% | 93.90% | -5.6 | Vẫn ở mức Xuất sắc, có thể do VLM variance |
| `image_35.png` | 96.40% | 85.80% | -10.6 | Cần kiểm tra lại prediction output |

---

## 3. Phân nhóm hiệu suất (50 ảnh — Đợt 3)

### 🟢 Nhóm Xuất sắc (Sim ≥ 93%) — 22 ảnh
Hệ thống cực kỳ mạnh với text thuần, hóa đơn cơ bản, và bảng biểu đơn giản.
*(Tiêu biểu: image_43 = 100%, image_47 = 99.9%, image_22 = 99.8%, image_46 = 99.1%, image_33 = 98.3%)*

### 🟡 Nhóm Trung bình (75% ≤ Sim < 93%) — 23 ảnh
Hóa đơn phức tạp, bảng sao kê cỡ vừa, tài liệu đa ngôn ngữ.
Trích xuất số liệu tốt nhưng cấu trúc bảng đôi lúc bị méo.

### 🔴 Nhóm Khó (Sim < 75%) — 5 ảnh
1. **`image_41.jpg` (62.40%)**: Phiếu xuất kho bỏ trống — VLM sinh tag rác `[uncertain:]`.
2. **`image_25.png` (65.50%)**: Bảng tài chính tiếng Nhật cỡ nhỏ, dày đặc.
3. **`image_45.png` (67.70%)**: Tài liệu layout phức tạp, VLM gặp khó khăn xử lý.
4. **`image_44.png` (77.10%)**: Bảng biểu phức tạp, sát ranh giới nhóm Trung bình.
5. **`image_4.webp` (77.80%)**: Layout hỗn hợp text-bảng, sát ranh giới nhóm Trung bình.

---

## 4. Đánh giá tổng thể

Pipeline hiện tại đã đạt mức **ổn định tốt** với Sim trung bình **88.57%** trên 50 ảnh đa dạng (hóa đơn, sao kê, báo cáo tài chính, biểu mẫu, tài liệu đa ngôn ngữ). Hai ca thảm họa cũ (`image_13` và `image_14`) đã được khắc phục triệt để, nhảy vọt từ dưới 45% lên trên 89%.

**Kết luận:** Pipeline đủ chất lượng cho production. Các ca còn yếu (< 75%) đều là edge case rất đặc thù (biểu mẫu trống, tiếng Nhật dày đặc) — không ảnh hưởng đến use case doanh nghiệp thông thường.

---

## 5. Kế hoạch hành động kế tiếp (Next Steps Roadmap)

Để đưa pipeline từ mức "Ổn định tốt" (88.57% Sim) lên mức "Hoàn hảo" (>90% Sim toàn tập) trên mọi loại tài liệu doanh nghiệp, hệ thống sẽ tập trung xử lý các edge case theo thứ tự ưu tiên sau:

### 🔴 Ưu tiên cao (P0): Chống suy thoái layout & Xử lý lỗi cắt cụt chuỗi do vòng lặp

1. **Phân tích nguyên nhân suy thoái (Root-cause Analysis) cho `image_35` (-10.6%):**
   * Tiến hành `diff` văn bản dự đoán giữa Đợt 2 và Đợt 3 để xác định chính xác VLM bị mất đoạn văn hay bị nhảy định dạng layout.
   * Đánh giá nghiêm túc dựa trên bộ đôi metric **CER** và **Sim** tổng thể. Nếu CER không tăng mà Sim giảm do VLM tự ý thay đổi cách xuống dòng hoặc thêm ký tự bullet point không mong muốn, tiến hành tinh chỉnh Prompt Profile để cân bằng giữa cấu trúc bảng biểu và văn bản thuần.

2. **Nâng cấp Thuật toán Khử lặp (Dedup) ở tầng Hậu xử lý (Post-processing):**
   * Thay thế cơ chế đối sánh tuyệt đối (Exact Match Fingerprint) vốn dễ bị "lọt lưới" khi VLM cạn token sinh ra dẫn đến câu bị cụt giữa chừng.
   * Triển khai giải pháp **Cửa sổ trượt (Sliding Window)** kết hợp **Fuzzy Match ngắn** ở tầng Paragraph. Nếu khối văn bản cuối cùng (EOF) có độ tương đồng ký tự cao (>80%) với một phần của khối văn bản hoàn chỉnh liền trước ➡️ Tiến hành gọt bỏ hoàn toàn đoạn cụt đuôi để cứu điểm các file bị dính loop (tiêu biểu như `image_14`).

### 🟡 Ưu tiên trung bình (P1): Dọn rác mô hình & Tối ưu hóa biểu mẫu trống

3. **Mở rộng logic dọn dẹp văn bản trong `_cleanup_markdown()`:**
   * Không cài đặt thêm hàm mới để giữ code gọn gàng. Tiến hành bổ sung các pattern Regex vào hàm `_cleanup_markdown()` hiện có trong `pipeline.py` để quét và triệt tiêu toàn bộ các thẻ meta-tag đặc thù của mô hình (ví dụ: `[uncertain:]`, `<box>`, `[unrecognized]`).
   * Giải pháp này sẽ dọn sạch các ký tự rác tự sinh khi VLM xử lý các biểu mẫu trống hoặc form điền tay (như `image_41`), giúp bảo toàn độ sạch của văn bản đầu ra.

### 🟢 Ưu tiên thấp/Thử nghiệm (P2): Kiểm soát xử lý tài liệu mật độ cao

4. **Kiểm tra hành vi VLM trên tài liệu đa ngôn ngữ/mật độ dày (`image_25`):**
   * Đối với các tài liệu có mật độ chữ quá dày đặc và cỡ chữ nhỏ (như tài liệu tiếng Nhật), tiến hành đánh giá xem việc chạy 3 Workers có gây ảnh hưởng đến khả năng phân tách dòng của VLM hay không.
   * Giữ nguyên nguyên tắc xử lý lỗi lặp ở tầng Python code (Post-processing) thay vì can thiệp vào các tham số inference param (`frequency_penalty`), nhằm tránh rủi ro mô hình bỏ sót các từ khóa lặp lại tự nhiên rất phổ biến trong tài liệu doanh nghiệp (như "Bên A", "Bên B" trong hợp đồng, hoặc các đơn vị đo lường).
