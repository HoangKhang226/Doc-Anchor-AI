# Hướng dẫn Chấm điểm OCR (OCR Evaluation Guide)

Tài liệu này giải thích cách hoạt động của script chấm điểm `score_only.py`. Đây là một script chấm điểm độc lập, nhanh gọn, không yêu cầu load model PaddleOCR hay VLM, chỉ cần file dự đoán (`.pred.md`) và file đáp án gốc (`.gt.txt`).

## 1. Cơ chế Tiền xử lý (Text Normalization)

Mục đích tối thượng của pipeline OCR này là phục vụ **Hệ thống RAG**. Đối với RAG, cấu trúc thẩm mỹ (dấu gạch, dấu chấm, viền bảng) không quan trọng bằng việc trích xuất chính xác text và con số cốt lõi. Do đó, việc tiền xử lý xóa bỏ các yếu tố định dạng này không làm ảnh hưởng đến tính trung thực của bộ đánh giá, mà ngược lại, giúp phản ánh chính xác năng lực "hiểu" văn bản của mô hình.

Script `score_only.py` thực hiện chuẩn hóa mạnh mẽ như sau:
*   **Loại bỏ Headers:** Xóa toàn bộ các dòng tiêu đề do VLM tự sinh ra (bắt đầu bằng `#`). *(Lý do: Việc VLM sinh ra tiêu đề là một "tính năng" rất tốt giúp RAG dễ chia nhỏ văn bản (chunking), do đó file output vẫn giữ nguyên tiêu đề này. Tuy nhiên, khi đối chiếu với bản gốc không có tiêu đề, ta phải tạm ẩn chúng đi để VLM không bị phạt lỗi CER/WER một cách oan uổng)*.
*   **Loại bỏ định dạng biểu mẫu:** Xóa sạch các chuỗi chấm `..........` hoặc gạch dưới `________`. *(Lý do: Đây chỉ là các nét vẽ trực quan của form biểu mẫu điền tay, hoàn toàn không mang giá trị ngữ nghĩa đối với Vector Database)*.
*   **Loại bỏ tag của VLM:** Xóa các thẻ nghi ngờ như `[uncertain:]` do VLM tự chèn vào. *(Lý do: Đây là siêu dữ liệu log của mô hình, không nằm trong nội dung tài chính)*.
*   **Chuẩn hóa Markdown:** Xóa các nét sổ dọc `|`, dấu `*`, khoảng trắng thừa, v.v. *(Lý do: Loại bỏ "nhiễu định dạng" để thuật toán Levenshtein tập trung soi từng con chữ và số liệu thực tế)*.

## 2. Các chỉ số Đánh giá (Metrics)

### A. Nhóm Text-Level (Đánh giá hình thức)

*   **CER (Character Error Rate):** Tỷ lệ lỗi tính theo từng ký tự (Levenshtein distance). Càng thấp càng tốt.
    *   *Công thức:* 
        $$CER = \frac{\text{Levenshtein}(GT_{chars}, Pred_{chars})}{|GT_{chars}|}$$
    *   *Giải thích:*
        *   `Levenshtein`: Khoảng cách đổi/chèn/xóa nhỏ nhất để biến chuỗi này thành chuỗi kia.
        *   `GT_chars`: Chuỗi ký tự của bản gốc (Ground Truth).
        *   `Pred_chars`: Chuỗi ký tự do AI trích xuất (Prediction).
        *   `|GT_chars|`: Tổng số lượng ký tự của bản gốc.

*   **WER (Word Error Rate):** Tỷ lệ lỗi tính theo từng từ. Càng thấp càng tốt.
    *   *Công thức:* 
        $$WER = \frac{\text{Levenshtein}(GT_{words}, Pred_{words})}{|GT_{words}|}$$
    *   *Giải thích:*
        *   `GT_words`: Danh sách các từ (chia theo khoảng trắng) của bản gốc.
        *   `Pred_words`: Danh sách các từ do AI trích xuất.
        *   `|GT_words|`: Tổng số lượng từ của bản gốc.

*   **Sim (Similarity):** Mức độ tương đồng chuỗi ký tự (Fuzzy Matching). Tính theo %. Càng cao càng tốt.
    *   *Công thức:* 
        $$Sim = \left( 1 - \frac{\text{Levenshtein}(GT, Pred)}{\max(|GT|, |Pred|)} \right) \times 100\%$$
    *   *Giải thích:* (Áp dụng `rapidfuzz.fuzz.ratio`)
        *   `|GT|, |Pred|`: Độ dài tổng số ký tự của file gốc và file máy xuất. Nếu giống 100% nghĩa là trùng khớp hoàn toàn.

*   **TokenSim (Token Similarity):** Mức độ tương đồng về mặt từ vựng (không màng thứ tự từ). Càng cao càng tốt.
    *   *Công thức:* 
        $$TokenSim = \text{Sim}\Big(\text{Sort}(GT_{words}), \text{Sort}(Pred_{words})\Big)$$
    *   *Giải thích:* (Áp dụng `rapidfuzz.fuzz.token_sort_ratio`)
        *   `Sort`: Hàm băm nhỏ chuỗi thành các từ, bỏ qua dấu câu, chuyển chữ thường và sắp xếp lại các từ theo thứ tự chữ cái Alphabet trước khi đưa vào so sánh `Sim`. Giúp đánh giá việc bắt đúng từ khóa mà không bị ảnh hưởng quá lớn nếu VLM sắp xếp lệch cột.

### B. Nhóm Content Extraction (Đánh giá chuyên sâu Tài chính)
Do đặc thù làm RAG tài chính, việc bắt đúng con số quan trọng hơn việc gõ đúng dấu câu. Script bổ sung `content_extraction_score`:

1.  **Trích xuất Số liệu (Number F1):**
    *   Hỗ trợ đọc mọi định dạng: `1,000,000`, `1.000.000`, `-250`, và đặc biệt là chuẩn kế toán ngoặc đơn `(461)`.
    *   Sử dụng `Counter` để đếm tần suất xuất hiện của mỗi con số, so sánh Precision, Recall và tính ra điểm **Number F1**.
2.  **Trích xuất Nhãn thuật ngữ (Label Recall):**
    *   Lọc và so khớp các từ khóa tài chính cốt lõi (assets, liabilities, equity, thuế, dư nợ...).
    *   Tạo Bigram và Trigram để bắt các cụm thuật ngữ phức hợp.
3.  **Content Score Tổng hợp:**
    *   Được tính bằng công thức: `Content Score = 0.6 * Number_F1 + 0.4 * Label_Recall`
    *   Trọng số ưu tiên 60% cho độ chính xác của các con số.

### C. Nhóm Trích xuất Cấu trúc (Form KIE & Table Cell F1)
Để đánh giá chính xác các định dạng có cấu trúc phức tạp (Bảng và Biểu mẫu), script hỗ trợ:

1.  **Form KIE F1 (Key Information Extraction F1):**
    *   Đánh giá khả năng trích xuất các cặp `Key: Value`.
    *   Thuật toán sẽ quét cả Bullet points (`- Key: Value`), Bold text (`**Key**: Value`), và các Bảng Markdown 2 cột (`| Key | Value |`). 
    *   **Cập nhật quan trọng:** Đã khắc phục lỗi "Table Blindness" (không nhận diện được khi VLM tự cấu trúc Form thành bảng) và lỗi "Inline Hyphen" (bắt nhầm các từ có dấu gạch ngang như `COVID-19` thành Key).

2.  **Table Cell F1:**
    *   Đánh giá khả năng bảo toàn cấu trúc bảng.
    *   Bóc tách từng ô (cell) trong bảng Markdown và dùng thuật toán Fuzzy Matching (ngưỡng 80%) để so khớp số lượng ô trích xuất đúng vị trí so với bản gốc.

## 3. Cách chạy Script

Mở terminal trong môi trường ảo (venv) và chạy:
```bash
python evaluation/ocr/scripts/score_only.py
```
Script sẽ tự động quét thư mục `data/custom_finsight`, in bảng báo cáo ra màn hình với các icon màu sắc (🟢, 🟡, 🔴) đánh giá chất lượng, và lưu một bản báo cáo đầy đủ vào `evaluation/ocr/eval_report.txt`.
