# Báo Cáo Đánh Giá OCR (OCR Evaluation Report)

Đây là báo cáo tổng hợp kết quả đánh giá (benchmark) hệ thống OCR + VLM của dự án FinSight AI trên tập dữ liệu `custom_finsight` gồm 50 tài liệu thực tế (hóa đơn, báo cáo tài chính, biểu mẫu...).

## 1. Kết quả Tổng quan

Điểm số được tính toán dựa trên kịch bản trích xuất thông tin phục vụ Hệ thống RAG Tài chính. Điểm số trung bình toàn tập (50 files) như sau:

*   **Tổng số file đánh giá:** 50
*   **CER (Character Error Rate):** 0.226 (Tỷ lệ lỗi ký tự)
*   **WER (Word Error Rate):** 0.347 (Tỷ lệ lỗi từ)
*   **Sim (Similarity):** 86.06% (Độ tương đồng chuỗi)
*   **TokenSim (Token Similarity):** 84.05% (Độ tương đồng từ vựng/cấu trúc)

### 📈 Chỉ số Trích xuất Nội dung (Content Extraction Score)
Đây là thước đo quan trọng nhất đánh giá năng lực của VLM trong việc bắt chính xác các con số và thuật ngữ tài chính:
*   **Number F1 trung bình:** **90.22%**
*   **Label Recall trung bình:** 73.63%
*   **Content Score trung bình:** **83.58%**

## 2. Nhận xét & Đánh giá

*   **Khả năng trích xuất số liệu xuất sắc:** Điểm số **Number F1 > 90%** chứng minh rằng hệ thống bắt cực kỳ chính xác các con số tài chính (bao gồm số thập phân, số định dạng kế toán dạng `(100)`...). Điều này đảm bảo Database Vector của hệ thống RAG lưu trữ dữ liệu với độ tin cậy rất cao.
*   **Độ ổn định với biểu mẫu trống:** Thuật toán chấm điểm đã được tối ưu hóa để phớt lờ các đường chấm đứt (`..........`) và gạch dưới (`______`) trong các biểu mẫu điền tay (như Phiếu xuất kho). Nhờ vậy, điểm số sát với nội dung cốt lõi của tài liệu hơn.
*   **Điểm mạnh của VLM:** Khả năng tự động cấu trúc hóa văn bản thô thành Markdown rất tốt. Hệ thống tự động thêm các tiêu đề (Headers) giúp văn bản rõ ràng hơn, rất lý tưởng để chia nhỏ (chunking) khi nạp vào Qdrant.

## 3. Xem chi tiết

Để xem chi tiết điểm số của từng file, vui lòng kiểm tra file: `scripts/eval_report.txt` (được tự động sinh ra khi chạy script chấm điểm). Để tìm hiểu về thuật toán chấm điểm, xem tại `docs/evaluate.md`.
