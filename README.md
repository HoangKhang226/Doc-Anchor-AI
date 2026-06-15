# Doc Anchor AI: Enterprise OCR Pipeline

## Tầm nhìn Dự án (Project Vision)

Trích xuất tài liệu là một trong những thách thức cốt lõi trong môi trường doanh nghiệp. Dự án này vượt ra khỏi việc trích xuất văn bản đơn thuần; nó xây dựng một pipeline OCR có khả năng mở rộng, dễ dàng tái tạo và sẵn sàng cho môi trường thực tế.

Bằng cách chuyển đổi ảnh chụp tài liệu thô thành định dạng Markdown có cấu trúc rõ ràng thông qua một quy trình chuẩn hóa, dự án cung cấp nền tảng để biến tài liệu vật lý và kỹ thuật số thành dữ liệu chất lượng cao phục vụ cho các ứng dụng hạ nguồn (như RAG). Dự án tập trung hoàn toàn vào xử lý hình ảnh (không phân tích trực tiếp file Docling/Word/PDF) bằng cách sử dụng sức mạnh của OpenCV, PaddleOCR và các Mô hình Ngôn ngữ Thị giác (VLMs).

---

## Công nghệ & Kiến trúc (Tech Stack & Architecture)

| Phân loại                 | Công cụ                |
| :------------------------ | :--------------------- |
| Ngôn ngữ                  | Python 3.12            |
| Thị giác Máy tính         | OpenCV                 |
| Nhận dạng Chữ viết (OCR)  | PaddleOCR (PP-OCR)     |
| Mô hình Ngôn ngữ Thị giác | Qwen(VLM 2.5-7B)       |
| Đo lường & Đánh giá       | RapidFuzz, Levenshtein |

---

## Vòng đời Xử lý (The Lifecycle)

Mỗi giai đoạn là một module độc lập được đặt tại thư mục `src/ingestion`:

1. Tiền xử lý Dữ liệu: Căn chỉnh hình ảnh, cắt xén và giảm nhiễu bằng OpenCV.
2. Phân tích Bố cục: Nhận diện thông minh các đoạn văn, bảng biểu phức tạp và vùng chứa biểu đồ.
3. Trích xuất Văn bản: Trích xuất văn bản thô với hiệu suất cao sử dụng PaddleOCR.
4. Tái cấu trúc bằng VLM: Sử dụng các Mô hình Ngôn ngữ Thị giác (VLM) để tái cấu trúc các biểu mẫu và bảng biểu phức tạp thành định dạng Markdown chuẩn xác.
5. Tổng hợp Cuối cùng: Ghép tất cả các thành phần đã trích xuất thành một tài liệu Markdown duy nhất, hoàn chỉnh kèm theo tọa độ không gian.

---

## Bắt đầu Nhanh (Quick Start)

### Cài đặt Môi trường Cục bộ (Local Development)

```bash
# 1. Clone mã nguồn
git clone https://github.com/HoangKhang226/Doc-Anchor-AI.git && cd Doc-Anchor-AI

# 2. Thiết lập Môi trường
python -m venv venv
source venv/Scripts/activate  # Hoặc venv\Scripts\activate trên Windows
pip install -r requirements.txt

# 3. Chạy Pipeline
python main.py
```

---

## Cấu trúc Thư mục

```text
.
├── src/
│   └── ingestion/                 # Pipeline OCR Cốt lõi
│       ├── core/                  # Các thành phần tiền xử lý và logic phân loại lõi
│       ├── extractors/            # Module trích xuất (PaddleOCR & VLM)
│       ├── tables/                # Module xử lý và nhận diện bảng biểu
│       └── pipeline.py            # File điều phối trung tâm
├── evaluation/
│   └── ocr/                       # Bộ công cụ Benchmark & Đánh giá
│       ├── data/                  # Tập dữ liệu Ground Truth
│       ├── scripts/               # Các kịch bản chấm điểm
│       └── textforge_ocr_predictions/ # Kết quả dự đoán của AI
├── scratch/                       # Khu vực thử nghiệm & Gỡ lỗi tạm thời
└── README.md                      # Tài liệu Dự án
```

---

## Benchmark

Hiệu năng của toàn bộ quy trình được đánh giá khắt khe thông qua một tập dữ liệu tùy chỉnh gồm 76 tài liệu doanh nghiệp thực tế, được phân loại thành Biểu mẫu (Forms), Bố cục hỗn hợp (Mixed Layouts), Văn bản thuần (Pure Text), và Bảng biểu (Tables).

**Hiệu suất Toàn Hệ thống:**

- CER (Tỷ lệ lỗi ký tự): 0.075
- WER (Tỷ lệ lỗi từ): 0.143
- Độ tương đồng (Đã chuẩn hóa): 95.35%
- Độ tương đồng Từ vựng (Token Similarity): 93.97%

Chi tiết cách đánh giá và thuật toán chấm điểm, vui lòng xem tại:
[evaluation/ocr/docs/evaluate.md](evaluation/ocr/docs/evaluate.md)

Chi tiết số liệu, bảng điểm F1 (Form KIE F1, Table Cell F1) và báo cáo chuyên sâu, vui lòng xem tại:

- [evaluation/ocr/README.md](evaluation/ocr/README.md)
- [evaluation/ocr/eval_report.txt](evaluation/ocr/eval_report.txt)

---

## Đóng góp & Bảo mật

Dự án được duy trì bởi [HoangKhang226](https://github.com/HoangKhang226).
Được xây dựng với niềm đam mê dành cho các tiêu chuẩn Document AI và MLOps hiện đại.
