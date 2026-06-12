# 📊 Phân Tích Benchmark (Post-Fix): Đánh Giá Pipeline VLM OCR

## 1. Tổng kết nhanh (Sau khi apply P0 Fixes & 3 Workers)

| Chỉ số | Trước Fix (5 Workers) | Sau Fix (3 Workers) | Biến động |
|--------|-----------------------|---------------------|-----------|
| Tổng số ảnh | 50 | 50 | - |
| CER trung bình | 0.291 | **0.228** | ⬇️ Cải thiện 21% (Rất tốt) |
| WER trung bình | 0.420 | **0.363** | ⬇️ Giảm đáng kể |
| **Sim trung bình** | 86.49% | **87.21%** | ⬆️ Tăng 0.72% |
| TokenSim trung bình | 84.25% | **85.29%** | ⬆️ Tăng 1.04% |

---

## 2. Đánh giá hiệu quả các Fix P0 (Tác động MLOps)

### ✅ Giải quyết OOM Crash (Giới hạn 3 Workers)
Việc ép hệ thống chạy song song tối đa 3 workers đã khắc phục triệt để tình trạng VLM (Qwen2.5-VL 7B) bị sập giữa chừng do quá tải VRAM trên Colab T4.
*   **Minh chứng:** Các ảnh cuối tập trước đây bị ép fallback sang PaddleOCR thô, nay đã được VLM xử lý hoàn hảo. `image_43.jpg` (Bảng tiếng Nhật) nhảy vọt từ **74.40% lên 100.00%**. Các ảnh `45`, `46`, `47` đều đạt trên **99%**.

### ✅ Tối ưu Table Validator (Chặn đắp bảng rác OpenCV)
Nhận định "OpenCV đắp thêm bảng rác làm tệ hơn tình hình" đã được chứng minh là chính xác.
*   **Minh chứng:** `image_13.png` điểm Sim tăng từ **43.70% lên 54.40%**.
*   **Phân tích:** Đây là bảng tài chính 10 cột, vượt quá khả năng dựng Markdown table của VLM 7B nên nó rã ra thành dạng danh sách cặp (`Thuyết minh - Giá trị`). Nhờ logic mới, hệ thống nhận diện đây là cấu trúc có data và **hủy fallback**, giúp bảo toàn tính toàn vẹn của Vector Embedding thay vì nhồi nhét một cái bảng mất header do OpenCV dựng.

### 🟡 Dedup Repetition (Xóa vòng lặp nội dung)
Thuật toán gọt rác hậu kỳ (Post-processing) đã phát huy tác dụng mạnh mẽ.
*   **Minh chứng:** `image_14.jpeg` (Sao kê Techcombank 8 cột) điểm Sim tăng vọt từ **37.00% lên 71.70%**.
*   **Phân tích sâu:** Theo dõi file dự đoán, code dedup đã hoạt động và xóa thành công 3 trên 5 bản sao bị lặp lại, giảm dung lượng file rác từ 12KB xuống 4KB.
*   **Vấn đề tồn đọng:** Điểm số chưa đạt mức >90% là do ở bản sao cuối cùng, VLM đạt giới hạn token sinh ra (EOF) khiến câu bị cắt đứt đoạn. Do cụt giữa chừng, hàm băm (fingerprint) của nó không khớp 100% với bản hoàn chỉnh phía trên nên thuật toán dedup bị "lọt lưới".

---

## 3. Phân nhóm hiệu suất (50 ảnh)

### 🟢 Nhóm Xuất sắc (Sim ≥ 93%) — 23 ảnh (Tăng thêm 3 ảnh)
Hệ thống cực kỳ mạnh mẽ với các ảnh text thuần, hóa đơn cơ bản, và bảng biểu đơn giản (Dưới 5 cột). Mô hình trích xuất con số và layout gần như hoàn hảo.
*(Tiêu biểu: image_22, image_43, image_45, image_47 đạt 99% - 100%)*

### 🟡 Nhóm Trung bình (75% ≤ Sim < 93%) — 23 ảnh
Bao gồm các hóa đơn phức tạp, tài liệu tiếng Nhật, hoặc các bảng sao kê cỡ vừa.
Mô hình giữ được Content Score (trích xuất số liệu) rất cao, nhưng cấu trúc bảng đôi lúc bị móp méo khiến điểm Sim (Fuzzy Match) bị kéo xuống.

### 🔴 Nhóm Khó (Sim < 75%) — 4 ảnh (Đã giảm 3 ảnh thảm họa so với đợt trước)
Chỉ còn lại các ca "xương xẩu" nhất của bộ dữ liệu:
1.  **`image_13.png` (54.40%)**: Bảng báo cáo tài chính 10 cột chằng chịt.
2.  **`image_41.jpg` (63.30%)**: Phiếu xuất kho bị bỏ trống hoàn toàn. VLM tự động sinh ra các tag rác như `[uncertain:]` để điền vào chỗ trống.
3.  **`image_25.png` (68.20%)**: Bảng tài chính tiếng Nhật cỡ nhỏ và rất dày đặc.
4.  **`image_14.jpeg` (71.70%)**: Sao kê Techcombank vướng vòng lặp cắt cụt (Repetition loop cutoff).

---

## 4. Hành động tiếp theo (Next Steps cho P1/P2)

1.  **Gọt triệt để Repetition (P1):** Cập nhật thuật toán Dedup sử dụng logic đối sánh một phần (`startswith`) thay vì so sánh tuyệt đối (exact match) để gọt bỏ hoàn toàn các đoạn văn bị lặp nhưng bị cắt đứt do hết Token. (Sẽ xử lý dứt điểm `image_14`).
2.  **Bỏ qua khoảng trống (P2):** Post-process gỡ bỏ các đoạn `[uncertain:]` để khắc phục lỗi điền rác trên các biểu mẫu trống như `image_41`.
3.  **Thử nghiệm Inference Params:** Cân nhắc tinh chỉnh `frequency_penalty` hoặc `presence_penalty` (0.1 - 0.3) tại VLM engine để ngăn chặn vòng lặp từ trong trứng nước.
