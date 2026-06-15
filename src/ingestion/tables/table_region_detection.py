"""
Doc Anchor AI — Heuristic table region detection from cleaned images.

Module này tìm vùng bảng bằng morphology OpenCV để cô lập từng bảng trước
khi dựng lại Markdown từ OCR blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config import get_logger

logger = get_logger(__name__)


@dataclass
class TableRegion:
    """Vùng bảng được phát hiện trên ảnh."""

    bbox: tuple[int, int, int, int]
    score: float = 0.0
    source: str = "morphology"

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]


class TableRegionDetector:
    """Detect vùng bảng từ ảnh sử dụng PP-Structure (Deep Learning) thay vì OpenCV."""
    
    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                import os
                os.environ["FLAGS_use_mkldnn"] = "0"
                os.environ["FLAGS_enable_pir_api"] = "0"
                os.environ["PADDLE_DISABLE_MKLDNN"] = "1"
                from paddleocr import PPStructure
                self._engine = PPStructure(lang="en", show_log=False, image_orientation=False, ocr=False, table=False, recovery=False)
            except ImportError:
                logger.debug("Không tìm thấy module PPStructure. Sẽ bỏ qua và dùng thuật toán fallback.")
                return None
        return self._engine

    def detect(self, image_path: str | Path) -> list[TableRegion]:
        path = str(image_path)
        image = cv2.imread(path)
        if image is None:
            logger.warning(f"Không thể đọc ảnh để detect vùng bảng: {path}")
            return []

        regions = self.detect_from_image(image)
        logger.info(f"Detected {len(regions)} table region(s) from: {path} using PP-Structure")
        return regions

    def detect_from_image(self, image: np.ndarray) -> list[TableRegion]:
        if image is None or image.size == 0:
            return []

        engine = self._get_engine()
        if not engine:
            return []

        try:
            if hasattr(engine, "predict"):
                result_iter = engine.predict(image)
                # predict() returns an iterator/generator in some versions, or a single result
                result = next(result_iter) if hasattr(result_iter, "__next__") else result_iter
                # V3 result usually has 'res' property which contains layout
                if hasattr(result, "json"):
                    result_dict = result.json()
                elif hasattr(result, "res"):
                    result_dict = result.res
                else:
                    result_dict = result
                # Thường trả về layout bbox trong dt_polys hoặc layouts
                # Tạm thời tương thích với cả list format
                if isinstance(result_dict, dict) and "layout" in result_dict:
                    result = result_dict["layout"]
                elif isinstance(result_dict, list):
                    result = result_dict
                elif isinstance(result_dict, dict):
                    # Khám phá cấu trúc dict
                    if "res" in result_dict and isinstance(result_dict["res"], list):
                        result = result_dict["res"]
                    else:
                        result = [result_dict]
            else:
                result = engine(image)
        except Exception as e:
            logger.error(f"Lỗi khi chạy PP-Structure: {e}")
            return []

        regions: list[TableRegion] = []
        for res in result:
            if res.get('type') == 'table':
                bbox = res.get('bbox')
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    regions.append(TableRegion(bbox=(x1, y1, x2, y2), score=0.95, source="ppstructure"))

        regions.sort(key=lambda region: (region.y1, region.x1))
        return self._merge_overlapping_regions(regions)

    def _merge_overlapping_regions(self, regions: list[TableRegion]) -> list[TableRegion]:
        if not regions:
            return []

        merged: list[TableRegion] = [regions[0]]
        for region in regions[1:]:
            last = merged[-1]
            if self._iou(last.bbox, region.bbox) >= 0.25:
                merged[-1] = TableRegion(
                    bbox=(
                        min(last.x1, region.x1),
                        min(last.y1, region.y1),
                        max(last.x2, region.x2),
                        max(last.y2, region.y2),
                    ),
                    score=max(last.score, region.score),
                    source=last.source,
                )
            else:
                merged.append(region)
        return merged

    def _iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter_area / max(1.0, float(area_a + area_b - inter_area))


def get_bbox_bounds(bbox: Any) -> tuple[float, float, float, float]:
    """Trích xuất x_min, y_min, x_max, y_max một cách an toàn từ nhiều định dạng bbox."""
    if not bbox:
        return 0.0, 0.0, 0.0, 0.0
    
    if isinstance(bbox, str):
        import json
        try:
            bbox = json.loads(bbox)
        except Exception:
            return 0.0, 0.0, 0.0, 0.0
            
    if isinstance(bbox, (list, tuple)):
        # Nếu là mảng 1 chiều 4 phần tử [x1, y1, x2, y2]
        if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
            return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        # Nếu là mảng 2 chiều [[x1, y1], [x2, y2], ...]
        if len(bbox) > 0 and isinstance(bbox[0], (list, tuple)):
            try:
                xs = [float(p[0]) for p in bbox]
                ys = [float(p[1]) for p in bbox]
                return min(xs), min(ys), max(xs), max(ys)
            except (IndexError, TypeError):
                pass
    return 0.0, 0.0, 0.0, 0.0

def crop_ocr_blocks_to_region(ocr_blocks: list[Any], region: TableRegion) -> list[Any]:
    """Lấy các OCR blocks có tâm nằm trong bbox region."""
    selected: list[Any] = []
    x1, y1, x2, y2 = region.bbox
    for block in ocr_blocks:
        if not getattr(block, "bbox", None):
            continue
        
        bx_min, by_min, bx_max, by_max = get_bbox_bounds(block.bbox)
        center_x = (bx_min + bx_max) / 2.0
        center_y = (by_min + by_max) / 2.0
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            selected.append(block)
    return selected
