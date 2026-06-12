# 📊 Phân Tích Benchmark: Vì Sao Sim Trung Bình Chỉ 86.49%

## Tổng kết nhanh

| Chỉ số | Giá trị |
|--------|---------|
| Tổng ảnh | 50 |
| CER trung bình | 0.291 |
| WER trung bình | 0.420 |
| **Sim trung bình** | **86.49%** |
| TokenSim trung bình | 84.25% |

---

## 🔬 Phân nhóm chi tiết

### 🟢 Nhóm Xuất sắc (Sim ≥ 93%) — 20 ảnh

| Ảnh | Sim | Đặc điểm |
|-----|-----|----------|
| image_47.png | 99.90% | Text thuần, rõ nét |
| image_22.png | 99.80% | Text thuần |
| image_30.webp | 99.50% | Text thuần |
| image_6.png | 99.40% | Text thuần |
| image_46.png | 99.10% | Text thuần |
| image_45.png | 98.70% | Text thuần |
| image_32.png | 98.50% | Bảng đơn giản |
| image_37.jpg | 97.80% | Text + bảng nhẹ |
| image_12.jpg | 96.50% | Mixed layout |
| image_10.jpg | 96.50% | Mixed layout |
| image_35.png | 96.40% | Mixed layout |
| image_31.png | 95.80% | Mixed layout |
| image_1.webp | 95.70% | Webp format |
| image_20.png | 95.00% | Text thuần |
| image_38.webp | 94.70% | Webp format |
| image_28.jpg | 94.80% | Mixed layout |
| image_26.jpg | 93.80% | Mixed layout |
| image_8.png | 93.60% | Mixed layout |
| image_50.webp | 93.60% | Webp format |
| image_39.jpg | 93.50% | Mixed layout |
| image_21.jpg | 93.30% | Mixed layout |

→ **Nhận xét:** Hệ thống rất mạnh với text thuần và bảng đơn giản. **Không có gì cần sửa ở nhóm này.**

### 🟡 Nhóm Trung bình (75% ≤ Sim < 93%) — 19 ảnh

| Ảnh | Sim | Nguyên nhân chính |
|-----|-----|-------------------|
| image_23.png | 92.30% | Bảng phức tạp (8 cột), Table Validator fallback |
| image_49.webp | 93.00% | OK |
| image_36.jpg | 91.20% | Bảng phức tạp |
| image_11.png | 90.20% | Bảng lớn |
| image_3.webp | 90.10% | Bảng |
| image_18.png | 89.80% | Bảng |
| image_9.jpg | 88.10% | Bảng + text Việt |
| image_17.png | 87.60% | Bảng |
| image_33.jpg | 87.60% | Bảng |
| image_5.webp | 86.70% | VLM sinh dư (pred 5568 vs gt 3317 bytes) |
| image_40.jpg | 86.40% | Bảng |
| image_42.jpg | 85.20% | Bảng |
| image_48.webp | 85.20% | Bảng |
| image_24.jpg | 83.50% | Bảng tiếng Anh ngắn |
| image_16.png | 83.40% | Bảng |
| image_2.webp | 83.40% | Bảng |
| image_34.webp | 82.70% | Bảng |
| image_15.png | 82.00% | Bảng nhỏ |
| image_7.png | 81.80% | Bảng phức tạp |
| image_29.png | 79.30% | Bảng phức tạp |
| image_44.png | 77.60% | Bảng nhỏ |
| image_4.webp | 76.30% | VLM sinh dư (pred 4348 vs gt 2841 bytes) |
| image_43.jpg | 74.40% | Bảng phức tạp + tiếng Nhật |

### 🔴 Nhóm Thảm họa (Sim < 75%) — 7 ảnh, kéo trung bình xuống mạnh

| Ảnh | Sim | CER | Nguyên nhân gốc |
|-----|-----|-----|-----------------|
| **image_14.jpeg** | **37.00%** | 3.120 | 🔥 VLM lặp toàn bộ nội dung 5 LẦN (GT: 2.4KB → Pred: 12KB) |
| **image_13.png** | **43.70%** | 1.790 | 🔥 Bảng 10 cột phức tạp → VLM rã thành bullet list + Table Validator đắp thêm bảng rác |
| **image_41.jpg** | **62.00%** | 0.510 | Phiếu xuất kho trống → VLM điền `[uncertain:]` vào chỗ trống |
| **image_19.png** | **62.40%** | 0.970 | Bảng phức tạp → Table Validator fallback toàn bộ OpenCV |
| **image_25.png** | **68.20%** | 0.850 | Bảng tiếng Nhật → VLM rã thành bullet + reconstructed_table rác |
| **image_27.png** | **67.30%** | 0.437 | Bảng tiếng Nhật → VLM sập, chỉ PaddleOCR gánh (chạy 5 workers) |
| **image_43.jpg** | **74.40%** | 0.353 | Bảng phức tạp |

---

## 🧬 3 Nguyên nhân gốc rễ (Root Causes)

### 1. 🔥 VLM Repetition Loop (Lặp nội dung) — Ảnh hưởng: CỰC KỲ LỚN

**Ảnh tiêu biểu:** `image_14.jpeg` (Sim: 37%)

VLM (Qwen2.5-VL 7B) bị hiện tượng **repetition loop** — lặp đi lặp lại toàn bộ nội dung tài liệu nhiều lần. Ground truth chỉ có **2.4KB** nhưng prediction lên tới **12KB** (gấp 5 lần). Nội dung y hệt nhau được lặp lại 5 lần liên tiếp.

**Vì sao xảy ra?**
- Đây là bảng sao kê ngân hàng Techcombank có 8 cột phức tạp
- VLM model 7B bị vượt context window → rơi vào vòng lặp
- Temperature = 0.0 giúp giảm hallucination nhưng KHÔNG chống được repetition loop

**Giải pháp khả thi (không sửa prompt):**
- Thêm **post-processing dedup** trong `_cleanup_markdown_for_rag()`: phát hiện khối text ≥ 200 ký tự bị lặp lại → xóa bản sao

### 2. 🏗️ Bảng lớn bị rã thành Bullet List — Ảnh hưởng: LỚN

**Ảnh tiêu biểu:** `image_13.png` (Sim: 43.7%), `image_25.png` (Sim: 68.2%)

Khi bảng có **≥ 5 cột**, VLM thường không dựng được Markdown table đúng cấu trúc mà rã ra thành danh sách bullet (`- Property: value`). Sau đó Table Validator phát hiện "VLM không sinh bảng" → đắp thêm `reconstructed_table` từ OpenCV, tạo ra output thừa và hỗn loạn.

**Vì sao xảy ra?**
- Bảng 10 cột (image_13) hay bảng tiếng Nhật (image_25, 27) quá phức tạp cho context window 7B
- OpenCV reconstructed table thiếu header → bảng rác, không khớp GT

**Giải pháp khả thi:**
- Cải thiện logic **Table Validator**: Khi VLM đã rã bảng thành bullet có đủ data → KHÔNG đắp thêm reconstructed_table rác nữa, vì nó chỉ làm tệ hơn

### 3. 📡 VLM Crash (5 Workers OOM) — Ảnh hưởng: TRUNG BÌNH

**Ảnh tiêu biểu:** `image_27.png` (Sim: 67.3%)

Như đã phân tích ở phiên trước, chạy **5 workers** khiến VLM crash từ ảnh 33-36. Các ảnh sau đó bị fallback 100% PaddleOCR — bảng tiếng Nhật chỉ có số thô, mất toàn bộ header và cấu trúc.

**Giải pháp:** Đã rõ — chạy lại với `--workers 1`.

---

## 📈 Nếu fix 3 root causes, điểm sẽ tăng bao nhiêu?

| Ảnh | Sim hiện tại | Sim ước tính sau fix | Nguyên nhân fix |
|-----|-------------|---------------------|-----------------|
| image_14 | 37.0% | ~90% | Dedup repetition → chỉ giữ 1 bản |
| image_13 | 43.7% | ~75% | Bỏ reconstructed_table rác |
| image_25 | 68.2% | ~80% | Bỏ reconstructed_table rác |
| image_27 | 67.3% | ~85% | Chạy lại workers 1 (VLM sống) |
| image_19 | 62.4% | ~75% | Bỏ fallback toàn bộ OpenCV khi VLM đã có bullet |
| image_41 | 62.0% | ~75% | Post-process bỏ `[uncertain:]` trống |
| image_43 | 74.4% | ~80% | Chạy lại workers 1 |

**Ước tính Sim trung bình sau fix: ~89-91%** (tăng ~3-5% so với 86.49%)

---

## 🎯 Tóm tắt hành động

| Ưu tiên | Hành động | Dự kiến cải thiện |
|---------|-----------|-------------------|
| 🔴 P0 | **Chạy lại benchmark với `--workers 1`** để VLM không crash | +1-2% Sim toàn tập |
| 🔴 P0 | **Thêm dedup repetition** trong `_cleanup_markdown_for_rag()` | +2-3% Sim (chủ yếu cứu image_14) |
| 🟡 P1 | **Sửa Table Validator**: Không đắp thêm `reconstructed_table` khi VLM đã trả bullet có data | +1-2% Sim |
| 🟢 P2 | **Post-process** bỏ `[uncertain:]` trống (chỉ ảnh hưởng image_41) | +0.2% Sim |

> [!IMPORTANT]
> **Hành động #1 quan trọng nhất:** Chạy lại toàn bộ 50 ảnh với `--workers 1` trước, rồi mới đánh giá lại.
> Kết quả hiện tại bị "ô nhiễm" bởi các ảnh 33-36 chạy khi VLM đã crash, không phản ánh đúng sức mạnh thực sự của hệ thống.
