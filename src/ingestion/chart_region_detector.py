"""
Doc Anchor AI — Chart Region Detection
Dùng morphology OpenCV kết hợp với trừu tượng hóa để cô lập biểu đồ.
"""

from __future__ import annotations

import cv2
import uuid
import math
from dataclasses import dataclass
from pathlib import Path

from src.config import get_logger
from src.ingestion.schemas.extraction_result import OCRBlock

logger = get_logger(__name__)

@dataclass
class ChartRegion:
    bbox: tuple[int, int, int, int]
    score: float = 1.0


class ChartRegionDetector:
    """Detect vùng biểu đồ bằng phương pháp loại trừ."""

    def detect(
        self,
        image_path: str | Path,
        ocr_blocks: list[OCRBlock],
        table_bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> list[ChartRegion]:
        """Trả về danh sách vùng biểu đồ."""
        table_bboxes = table_bboxes or []
        path = str(image_path)
        image = cv2.imread(path)
        if image is None:
            logger.warning(f"Không thể đọc ảnh để detect biểu đồ: {path}")
            return []

        # 1. Tìm contours lớn
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2
        )
        
        # Xóa đen các vùng đã được nhận diện là Bảng để không dính vào Biểu đồ
        for tx1, ty1, tx2, ty2 in table_bboxes:
            cv2.rectangle(thresh, (max(0, tx1-5), max(0, ty1-5)), (min(img_w, tx2+5), min(img_h, ty2+5)), 0, -1)
            
        # Xóa đen toàn bộ chữ (OCR Blocks) để chữ không làm "keo dính" các biểu đồ hoặc dính vào đoạn văn
        for b in ocr_blocks:
            if not getattr(b, "bbox", None): continue
            try:
                if isinstance(b.bbox[0], (list, tuple)) and len(b.bbox[0]) >= 2:
                    bx1 = int(min(p[0] for p in b.bbox))
                    bx2 = int(max(p[0] for p in b.bbox))
                    by1 = int(min(p[1] for p in b.bbox))
                    by2 = int(max(p[1] for p in b.bbox))
                elif len(b.bbox) == 4 and isinstance(b.bbox[0], (int, float)):
                    bx1, by1, bx2, by2 = int(b.bbox[0]), int(b.bbox[1]), int(b.bbox[2]), int(b.bbox[3])
                else:
                    continue
                cv2.rectangle(thresh, (bx1, by1), (bx2, by2), 0, -1)
            except Exception:
                pass

        # Dùng kernel to hơn để nối các nét rời rạc của biểu đồ (ví dụ các cột bar chart)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_area = img_w * img_h
        min_chart_area = img_area * 0.03  # Tối thiểu 3% diện tích ảnh
        
        chart_regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            if area < min_chart_area:
                continue
                
            # 1.5 Loại trừ các hình hộp quá dẹt (đường kẻ dọc hoặc kẻ ngang)
            if w < 60 or h < 60 or w / max(1, h) > 8 or h / max(1, w) > 8:
                continue
                
            # 2. Loại trừ vùng đã bị nhận diện là bảng (IoU or Inter > 30%)
            is_table = False
            for tx1, ty1, tx2, ty2 in table_bboxes:
                ix1 = max(x, tx1)
                iy1 = max(y, ty1)
                ix2 = min(x+w, tx2)
                iy2 = min(y+h, ty2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    if inter_area / area > 0.3 or inter_area / ((tx2-tx1)*(ty2-ty1)) > 0.3:
                        is_table = True
                        break
            if is_table:
                continue
                
            # 3. Loại trừ vùng có mật độ text thuần cao
            text_area = 0
            char_count = 0
            for b in ocr_blocks:
                if not getattr(b, "bbox", None):
                    continue
                
                try:
                    if isinstance(b.bbox[0], (list, tuple)) and len(b.bbox[0]) >= 2:
                        bx1 = int(min(p[0] for p in b.bbox))
                        bx2 = int(max(p[0] for p in b.bbox))
                        by1 = int(min(p[1] for p in b.bbox))
                        by2 = int(max(p[1] for p in b.bbox))
                    elif len(b.bbox) == 4 and isinstance(b.bbox[0], (int, float)):
                        bx1, by1, bx2, by2 = int(b.bbox[0]), int(b.bbox[1]), int(b.bbox[2]), int(b.bbox[3])
                    else:
                        continue
                except Exception:
                    continue
                
                ix1 = max(x, bx1)
                iy1 = max(y, by1)
                ix2 = min(x+w, bx2)
                iy2 = min(y+h, by2)
                
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    block_area = (bx2 - bx1) * (by2 - by1)
                    if block_area > 0 and inter_area / block_area > 0.5:
                        text_area += inter_area
                        char_count += len(b.text.strip())
            
            # Nếu lượng chữ quá nhiều (> 250 ký tự) hoặc chữ chiếm > 30% diện tích -> Text block
            if text_area / area > 0.30 or char_count > 250:
                continue
                
            # Nếu khối nằm ở tít trên cùng (y < 80) và rất dẹt (Header/Title)
            if y < 80 and h < 100 and w > 200:
                continue
                
            chart_regions.append(ChartRegion(bbox=(x, y, x+w, y+h)))
            
        # 4. Gộp/Lọc các vùng nằm lồng nhau
        final_regions = []
        for r in chart_regions:
            x1, y1, x2, y2 = r.bbox
            is_inside = False
            for o in chart_regions:
                if o is r: continue
                ox1, oy1, ox2, oy2 = o.bbox
                if ox1 <= x1 and oy1 <= y1 and ox2 >= x2 and oy2 >= y2:
                    is_inside = True
                    break
            if not is_inside:
                final_regions.append(r)
                
        logger.info(f"Detected {len(final_regions)} chart region(s) from: {path}")
        return final_regions

    def crop_and_save(
        self,
        image_path: str | Path,
        regions: list[ChartRegion],
        output_dir: Path,
        file_stem: str,
    ) -> list[dict]:
        """Crop ảnh biểu đồ và lưu thành file PNG."""
        if not regions:
            return []
            
        image = cv2.imread(str(image_path))
        if image is None:
            return []
            
        output_dir.mkdir(parents=True, exist_ok=True)
        assets = []
        
        for idx, r in enumerate(regions):
            x1, y1, x2, y2 = r.bbox
            
            # Padding nhẹ 10px để cắt không bị sát viền
            h, w = image.shape[:2]
            px1 = max(0, x1 - 10)
            py1 = max(0, y1 - 10)
            px2 = min(w, x2 + 10)
            py2 = min(h, y2 + 10)
            
            crop = image[py1:py2, px1:px2]
            filename = f"{file_stem}_chart_{idx+1}_{uuid.uuid4().hex[:6]}.png"
            out_path = output_dir / filename
            cv2.imwrite(str(out_path), crop)
            
            assets.append({
                "index": idx + 1,
                "filename": filename,
                "path": str(out_path).replace("\\", "/"),
                "bbox": [x1, y1, x2, y2],
                "score": r.score,
                "region_type": "chart"
            })
            
        return assets
