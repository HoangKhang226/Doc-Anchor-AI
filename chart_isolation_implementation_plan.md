# 📐 Technical Docs: Chart Isolation (Tách biệt Biểu đồ)

## 🎯 Mục tiêu

Tự động phát hiện vùng biểu đồ trong ảnh tài liệu, crop ra và lưu thành asset riêng.
**Trạng thái: Đã hoàn thành 100%. Không sửa prompt, không rẽ nhánh pipeline, giữ nguyên 100% benchmark.**

---

## 🏗️ Kiến trúc hiện tại — Hiểu trước khi sửa

Pipeline hiện tại chạy tuyến tính trong method [_extract_image_like](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/pipeline.py#L105-L191):

```
Ảnh → ImagePreprocessor → PaddleOCR → LayoutRouter → Build Prompt
  → VLM Extract → TableRegionDetector → Table Reconstruction
  → Table Validator → Normalize Markdown → ExtractionResult
```

### Các file liên quan trực tiếp

| File | Vai trò | Tình trạng |
|------|---------|---------|
| [pipeline.py](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/pipeline.py) | Luồng chính `_extract_image_like()` + `save_outputs()` | ✅ Đã tích hợp |
| [table_region_detection.py](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/table_region_detection.py) | OpenCV detect vùng bảng → trả `TableRegion` bbox | ❌ Không sửa |
| [extraction_result.py](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/schemas/extraction_result.py) | Schema output — `ExtractionResult` dataclass | ✅ Đã thêm field |
| [schemas/__init__.py](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/schemas/__init__.py) | Export symbols | ❌ Không sửa |
| [vlm_prompts.py](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/prompts/vlm_prompts.py) | 4 profile prompts cho VLM | ❌ Không sửa |
| [layout_router.py](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/routing/layout_router.py) | Router chọn prompt profile + temperature | ❌ Không sửa |
| `chart_region_detector.py` | Detect + crop biểu đồ | ✅ Đã tạo mới |

---

## 📋 Chi tiết triển khai

### Thành phần 1: `chart_region_detector.py`

> **Vị trí:** `src/ingestion/chart_region_detector.py` (cùng cấp với `table_region_detection.py`)

**Thuật toán — Phương pháp loại trừ:**

Hệ thống đã có `TableRegionDetector` trả về bbox của các vùng bảng ở [pipeline.py:131](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/pipeline.py#L131). Ta sẽ tận dụng thông tin đó:

1. Dùng OpenCV `adaptiveThreshold` + `morphologyEx(MORPH_CLOSE)` để tìm tất cả contour lớn (diện tích > 3% ảnh)
2. **Loại trừ** các contour đã trùng với `table_regions` (IoU ≥ 0.5)
3. **Loại trừ** các contour có mật độ OCR text cao (≥ 40% tổng ký tự OCR trong bbox = vùng văn bản thuần)
4. Phần còn lại = **biểu đồ** → crop bằng `cv2.imread` + array slicing → lưu PNG

**Class interface:**

```python
class ChartRegionDetector:
    def detect(
        self,
        image_path: Path,
        ocr_blocks: list[OCRBlock],
        table_bboxes: list[tuple[int,int,int,int]] | None = None,
    ) -> list[ChartRegion]:
        """Trả danh sách vùng biểu đồ. Rỗng nếu không có."""

    def crop_and_save(
        self,
        image_path: Path,
        regions: list[ChartRegion],
        output_dir: Path,
        file_stem: str,
    ) -> list[dict]:
        """Crop + lưu ảnh. Trả metadata của các chart đã lưu."""
```

> [!IMPORTANT]
> **Tại sao không dùng VLM để nhận diện biểu đồ?**
> Vì mục tiêu là KHÔNG tạo thêm tải cho Qwen2.5-VL (đã ăn hết 15GB VRAM trên T4).
> OpenCV detection là CPU-only, chạy song song mà không tranh chấp GPU với VLM.

### Thành phần 2: Schema `ExtractionResult`

> **File:** [extraction_result.py](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/schemas/extraction_result.py)

Thêm **1 dòng duy nhất** vào dataclass `ExtractionResult`:

```diff
 tables: list[ExtractedTable] = field(default_factory=list)
+chart_assets: list[dict] = field(default_factory=list)
 raw_ocr: list[OCRBlock] = field(default_factory=list)
```

Mỗi phần tử trong `chart_assets` có cấu trúc:
```json
{
  "index": 1,
  "filename": "invoice_chart_1.png",
  "path": "data/processed/assets/invoice_chart_1.png",
  "bbox": [120, 340, 580, 720],
  "score": 0.85,
  "region_type": "chart"
}
```

> [!NOTE]
> Field mới dùng `field(default_factory=list)` nên sẽ luôn là `[]` mặc định.
> 50 ảnh benchmark không có biểu đồ → `chart_assets = []` → **Benchmark không bị ảnh hưởng.**

### Thành phần 3: Tích hợp vào `pipeline.py` — `_extract_image_like()`

> **File:** [pipeline.py](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/pipeline.py#L105-L191)

Chèn **sau** dòng [131](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/pipeline.py#L131) (sau khi `TableRegionDetector` đã chạy xong):

```python
# === Chart Isolation (Phase 6b) — CPU-only, không ảnh hưởng VLM ===
chart_detector = ChartRegionDetector()
chart_regions = chart_detector.detect(
    cleaned_img_path,
    ocr_blocks,
    table_bboxes=[r.bbox for r in table_regions],
)
assets_dir = self.processed_dir / "assets"
chart_assets = chart_detector.crop_and_save(
    cleaned_img_path, chart_regions, assets_dir, path.stem,
)
```

Rồi chèn `chart_assets` vào `ExtractionResult` constructor ở [dòng 158](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/pipeline.py#L158):

```diff
 result = ExtractionResult(
     ...
     tables=reconstructed_tables,
+    chart_assets=chart_assets,
     raw_ocr=ocr_blocks,
     ...
 )
```

Và ghi chart info vào metadata:

```diff
 "table_regions": [...],
+"chart_regions": [
+    {"bbox": r.bbox, "score": r.score} for r in chart_regions
+],
+"chart_assets": chart_assets,
```

### Thành phần 4: Cập nhật `save_outputs()` — Append chart vào Markdown

> **File:** [pipeline.py](file:///d:/Project/Doc%20Anchor%20AI/src/ingestion/pipeline.py#L57-L70)

Sau khi ghi markdown chính, nối thêm link ảnh chart (nếu có):

```python
def save_outputs(self, result: ExtractionResult) -> tuple[Path, Path]:
    stem = Path(result.source_file).stem
    md_path = self.processed_dir / f"{stem}.md"
    json_path = self.processed_dir / f"{stem}.json"

    # Ghi markdown chính
    md_content = result.markdown

    # Append chart assets (nếu có)
    if result.chart_assets:
        md_content += "\n\n---\n\n## 📊 Extracted Charts\n\n"
        for chart in result.chart_assets:
            md_content += f"![Chart {chart['index']}](assets/{chart['filename']})\n\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    logger.info(f"Đã lưu output: {md_path}, {json_path}")
    return md_path, json_path
```

---

## 🛡️ Phân tích An toàn Benchmark

| Yếu tố | Tác động |
|---------|----------|
| VLM Prompt | ❌ Không sửa — 4 profile giữ nguyên |
| LayoutRouter logic | ❌ Không sửa — decision tree y nguyên |
| PaddleOCR flow | ❌ Không sửa |
| Table Validator | ❌ Không sửa |
| Normalize Markdown | ❌ Không sửa |
| `result.markdown` | ❌ Không bị thay đổi (chart chỉ append lúc `save_outputs()`) |
| Benchmark eval script | ❌ Không sửa — so sánh `extraction.markdown` trước khi save |

> [!TIP]
> Điểm quan trọng nhất: Benchmark script [run_ocr_eval.py:140](file:///d:/Project/Doc%20Anchor%20AI/evaluation/ocr/scripts/run_ocr_eval.py#L140) lấy `extraction.markdown` để chấm điểm.
> Chart chỉ được append khi gọi `save_outputs()`, mà benchmark **không gọi** `save_outputs()`.
> → **Điểm số hoàn toàn không bị ảnh hưởng.**

---

## 📁 Tóm tắt thay đổi

| Hành động | File | Mô tả |
|-----------|------|-------|
| **Tạo mới** | `src/ingestion/chart_region_detector.py` | OpenCV detect + crop |
| **Sửa** | `src/ingestion/schemas/extraction_result.py` | +1 field `chart_assets` |
| **Sửa** | `src/ingestion/pipeline.py` | Gọi detector và lưu chart_assets |

**Tổng code thay đổi: ~240 dòng mới, 0 dòng cũ bị sửa/xóa.**
