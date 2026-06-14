# Báo Cáo Đánh Giá OCR (OCR Evaluation Report)

Đây là báo cáo tổng hợp kết quả đánh giá (benchmark) hệ thống OCR + VLM của dự án FinSight AI trên tập dữ liệu `custom_finsight` gồm 80 tài liệu thực tế (hóa đơn, báo cáo tài chính, biểu mẫu...).

## 1. Kết quả Tổng quan

Điểm số được tính toán dựa trên kịch bản trích xuất thông tin phục vụ Hệ thống RAG Tài chính. Điểm số trung bình toàn tập (80 files) như sau:

- **Tổng số file đánh giá:** 80
- **CER (Character Error Rate):** 0.226 (Tỷ lệ lỗi ký tự)
- **WER (Word Error Rate):** 0.347 (Tỷ lệ lỗi từ)
- **Sim (Similarity):** 86.06% (Độ tương đồng chuỗi)
- **TokenSim (Token Similarity):** 84.05% (Độ tương đồng từ vựng/cấu trúc)

### 📈 Chỉ số Trích xuất Nội dung (Content Extraction Score)

Đây là thước đo quan trọng nhất đánh giá năng lực của VLM trong việc bắt chính xác các con số và thuật ngữ tài chính:

- **Number F1 trung bình:** **90.22%**
- **Label Recall trung bình:** 73.63%
- **Content Score trung bình:** **83.58%**

## 1.5. Kết quả Benchmark Mở rộng (Categorized Dataset - 80 files)

Đánh giá mở rộng trên tập dữ liệu đa dạng hơn (đã lọc bỏ các file nhiễu nặng), bao gồm 4 danh mục cốt lõi (20 file mỗi danh mục):

- **[PURE_TEXT]** (Văn bản thuần): **CER:** 0.016 | **Sim:** 98.96%
- **[FORMS]** (Biểu mẫu): **CER:** 0.042 | **Sim:** 97.69% | **Form KIE F1:** **79.94%**
- **[TABLES]** (Bảng biểu): **CER:** 0.079 | **Sim:** 95.24% | **Table Cell F1:** **76.66%**
- **[MIXED_LAYOUTS]** (Bố cục hỗn hợp): **CER:** 0.158 | **Sim:** 89.98%

**Tổng kết Toàn hệ thống (80 files):**

- **CER trung bình:** 0.074
- **WER trung bình:** 0.140
- **Sim trung bình:** 95.47%

## 2. Nhận xét & Đánh giá

- **Khả năng trích xuất số liệu xuất sắc:** Điểm số **Number F1 > 90%** chứng minh rằng hệ thống bắt cực kỳ chính xác các con số tài chính (bao gồm số thập phân, số định dạng kế toán dạng `(100)`...). Điều này đảm bảo Database Vector của hệ thống RAG lưu trữ dữ liệu với độ tin cậy rất cao.
- **Độ ổn định với biểu mẫu trống:** Thuật toán chấm điểm đã được tối ưu hóa để phớt lờ các đường chấm đứt (`..........`) và gạch dưới (`______`) trong các biểu mẫu điền tay (như Phiếu xuất kho). Nhờ vậy, điểm số sát với nội dung cốt lõi của tài liệu hơn.
- **Khả năng bóc tách cấu trúc (KIE & Tables):** Hệ thống đạt KIE F1 ~80% cho Form và Cell F1 ~77% cho Bảng biểu. Mô hình VLM có xu hướng tự động tái cấu trúc các biểu mẫu (Forms) thành dạng bảng Markdown 2 cột rất khoa học, và thuật toán chấm điểm đã được nâng cấp để hiểu và đánh giá đúng định dạng này.
- **Điểm mạnh của VLM:** Khả năng tự động cấu trúc hóa văn bản thô thành Markdown rất tốt. Hệ thống tự động thêm các tiêu đề (Headers) giúp văn bản rõ ràng hơn, rất lý tưởng để chia nhỏ (chunking) khi nạp vào Qdrant.

## 3. Xem chi tiết

Để xem chi tiết điểm số của từng file, vui lòng kiểm tra file: `scripts/eval_report.txt` (được tự động sinh ra khi chạy script chấm điểm). Để tìm hiểu về thuật toán chấm điểm, xem tại `docs/evaluate.md`.
