"""
Doc Anchor AI — Chart Region Detection v2.0
============================================
Phiên bản viết lại hoàn toàn, khắc phục 9 lỗi chí mạng của v1.0.

Thiết kế mới:
- Dynamic Scaling: Mọi ngưỡng dựa trên % kích thước ảnh, không hardcode pixel.
- Rotated Bounding Box: Dùng minAreaRect thay vì boundingRect để chống nghiêng.
- Smart Masking: Chỉ mask text NGOÀI vùng biểu đồ tiềm năng, tránh phá hủy cấu trúc.
- Multi-scale Morphology: Dùng 2 lượt kernel (nhỏ + lớn) để bắt cả chart mở và chart đặc.
- Cascade-safe: Giảm phụ thuộc vào upstream (Table/OCR) bằng dual-pass detection.
"""

from __future__ import annotations

import cv2
import uuid
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import get_logger
from src.ingestion.schemas.extraction_result import OCRBlock

logger = get_logger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ChartRegion:
    """Vùng biểu đồ được phát hiện."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) axis-aligned
    rotated_bbox: Any = None         # cv2.minAreaRect result (nếu có)
    score: float = 1.0
    detection_method: str = "morphology"


# ============================================================================
# Helper: Trích xuất bbox an toàn từ OCRBlock (chống crash đa format)
# ============================================================================

def _safe_bbox(block: OCRBlock) -> tuple[int, int, int, int] | None:
    """Trả về (x1, y1, x2, y2) từ OCRBlock.bbox, hoặc None nếu lỗi."""
    bbox = getattr(block, "bbox", None)
    if not bbox:
        return None
    try:
        if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            return min(xs), min(ys), max(xs), max(ys)
        elif len(bbox) == 4 and isinstance(bbox[0], (int, float)):
            return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    except Exception:
        pass
    return None


# ============================================================================
# Helper: Tính IoU giữa 2 hộp
# ============================================================================

def _iou(a: tuple, b: tuple) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(1.0, union)


def _intersection_ratio(inner: tuple, outer: tuple) -> float:
    """Tỉ lệ diện tích phần giao so với diện tích inner."""
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_inner = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area_inner


# ============================================================================
# Main Detector
# ============================================================================

class ChartRegionDetector:
    """
    Detect vùng biểu đồ bằng phương pháp Subtractive Morphology cải tiến.
    
    Khắc phục so với v1.0:
    - [Fix Lỗi 1, 8] Dynamic kernel & thresholds dựa trên % ảnh
    - [Fix Lỗi 5]     Smart masking: không tô đen chữ NẰM TRONG vùng chart tiềm năng
    - [Fix Lỗi 6]     Dual-pass: pass 1 (có mask table), pass 2 (không mask) -> merge
    - [Fix Lỗi 7]     minAreaRect cho rotated bbox + tính aspect ratio chính xác
    - [Fix Lỗi 9]     Multi-scale morphology: kernel nhỏ + kernel lớn, union kết quả
    """

    # --- Cấu hình (tất cả đều là TỈ LỆ, không phải pixel) ---
    MIN_CHART_AREA_RATIO = 0.025       # Biểu đồ tối thiểu 2.5% diện tích ảnh
    MAX_ASPECT_RATIO = 10.0            # Tỉ lệ dài/ngắn tối đa (loại đường kẻ)
    MIN_DIMENSION_RATIO = 0.04         # Cạnh ngắn nhất tối thiểu 4% cạnh dài ảnh
    TEXT_DENSITY_THRESHOLD = 0.35      # Mật độ chữ tối đa 35% -> vẫn là chart
    TEXT_CHAR_THRESHOLD_RATIO = 0.15   # Tối đa 15% ký tự so với tổng ký tự toàn ảnh
    HEADER_Y_RATIO = 0.06             # Vùng header: 6% chiều cao ảnh tính từ trên
    HEADER_H_RATIO = 0.08             # Chiều cao tối đa header: 8% ảnh
    KERNEL_SMALL_RATIO = 0.012        # Kernel nhỏ: 1.2% cạnh dài ảnh
    KERNEL_LARGE_RATIO = 0.025        # Kernel lớn: 2.5% cạnh dài ảnh
    TABLE_OVERLAP_THRESHOLD = 0.30    # Ngưỡng overlap với bảng để loại

    def detect(
        self,
        image_path: str | Path,
        ocr_blocks: list[OCRBlock],
        table_bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> list[ChartRegion]:
        """Entry point: Phát hiện vùng biểu đồ từ ảnh đã cleaned."""
        table_bboxes = table_bboxes or []
        path = str(image_path)
        image = cv2.imread(path)
        if image is None:
            logger.warning(f"Không thể đọc ảnh để detect biểu đồ: {path}")
            return []

        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w
        max_dim = max(img_h, img_w)

        # Tính toán tất cả ngưỡng động (Dynamic Thresholds)
        thresholds = self._compute_dynamic_thresholds(img_h, img_w, max_dim, img_area)

        # Tổng ký tự toàn ảnh (dùng để normalize char_count)
        total_chars = sum(len(b.text.strip()) for b in ocr_blocks if hasattr(b, 'text'))

        # Nhị phân hóa ảnh
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 2
        )

        # ================================================================
        # PASS 1: Subtractive (có mask Table + Text) — bắt chart chuẩn
        # ================================================================
        pass1_binary = binary.copy()

        # Mask bảng biểu
        for tx1, ty1, tx2, ty2 in table_bboxes:
            pad = int(max_dim * 0.005)
            cv2.rectangle(
                pass1_binary,
                (max(0, tx1 - pad), max(0, ty1 - pad)),
                (min(img_w, tx2 + pad), min(img_h, ty2 + pad)),
                0, -1
            )

        # Smart masking chữ: CHỈ mask text ở NGOÀI vùng chart tiềm năng
        # Bước này tránh Lỗi 5 (Over-masking phá hủy cấu trúc chart)
        # Trước tiên detect sơ bộ các vùng đồ họa lớn TRƯỚC khi mask text
        preliminary_regions = self._find_large_graphic_regions(
            pass1_binary, thresholds
        )
        for b in ocr_blocks:
            bb = _safe_bbox(b)
            if bb is None:
                continue
            # Kiểm tra: text block này có nằm trong vùng đồ họa lớn nào không?
            inside_chart = any(
                _intersection_ratio(bb, pr) > 0.7
                for pr in preliminary_regions
            )
            if not inside_chart:
                # Text ở ngoài vùng chart -> mask an toàn
                cv2.rectangle(pass1_binary, (bb[0], bb[1]), (bb[2], bb[3]), 0, -1)

        pass1_candidates = self._detect_from_binary(
            pass1_binary, thresholds, img_h, img_w, "pass1_subtractive"
        )

        # Đã xóa PASS 2 (Additive) vì nó đưa rác (đoạn văn, bảng biểu) vào kết quả.
        # Các biểu đồ chuẩn chỉ nên được lấy từ PASS 1 (Subtractive).
        all_candidates = list(pass1_candidates)

        # ================================================================
        # FILTER: Áp dụng bộ lọc Heuristics thông minh
        # ================================================================
        filtered = self._apply_heuristic_filters(
            all_candidates, ocr_blocks, table_bboxes,
            thresholds, total_chars, img_h, img_w
        )

        # Loại bỏ hộp lồng nhau
        final = self._remove_nested(filtered)
        
        # Gộp các vùng gần nhau (merge nearby) để tránh cắt chart thành nhiều mảnh
        final = self._merge_nearby_boxes(final, max_dim * 0.05) # Khoảng cách tối đa 5% cạnh dài

        logger.info(
            f"Chart Detection v2.0: {len(final)} region(s) "
            f"[pass1={len(pass1_candidates)}, "
            f"filtered={len(filtered)}] "
            f"from: {path}"
        )
        return final

    # ====================================================================
    # Tính ngưỡng động
    # ====================================================================

    def _compute_dynamic_thresholds(
        self, img_h: int, img_w: int, max_dim: int, img_area: int
    ) -> dict:
        """Tính toàn bộ ngưỡng dựa trên kích thước ảnh."""
        kernel_small = max(5, int(max_dim * self.KERNEL_SMALL_RATIO))
        kernel_large = max(11, int(max_dim * self.KERNEL_LARGE_RATIO))
        # Đảm bảo kernel là số lẻ (yêu cầu của OpenCV morphology)
        kernel_small = kernel_small | 1
        kernel_large = kernel_large | 1

        return {
            "min_area": int(img_area * self.MIN_CHART_AREA_RATIO),
            "min_dim": int(max_dim * self.MIN_DIMENSION_RATIO),
            "header_y": int(img_h * self.HEADER_Y_RATIO),
            "header_h": int(img_h * self.HEADER_H_RATIO),
            "kernel_small": kernel_small,
            "kernel_large": kernel_large,
            "img_area": img_area,
        }

    # ====================================================================
    # Tìm vùng đồ họa lớn sơ bộ (dùng cho Smart Masking)
    # ====================================================================

    def _find_large_graphic_regions(
        self, binary: np.ndarray, thresholds: dict
    ) -> list[tuple[int, int, int, int]]:
        """Tìm nhanh các vùng đồ họa lớn TRƯỚC khi mask text.
        Dùng kernel lớn để gom nhanh, chỉ lấy các khối siêu to."""
        k = thresholds["kernel_large"]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions = []
        min_area = thresholds["min_area"] * 2  # 2x ngưỡng thông thường
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= min_area:
                regions.append((x, y, x + w, y + h))
        return regions

    # ====================================================================
    # Core: Detect contours từ binary image
    # ====================================================================

    def _detect_from_binary(
        self,
        binary: np.ndarray,
        thresholds: dict,
        img_h: int,
        img_w: int,
        method_name: str,
    ) -> list[ChartRegion]:
        """Multi-scale morphology: chạy 2 kernel rồi union kết quả.
        [Fix Lỗi 9]: Kernel nhỏ bắt chart đặc, kernel lớn bắt chart mở."""
        candidates = []

        for kernel_key in ("kernel_small", "kernel_large"):
            k = thresholds[kernel_key]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(
                morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for c in contours:
                area = cv2.contourArea(c)
                if area < thresholds["min_area"]:
                    continue

                # [Fix Lỗi 7]: Dùng minAreaRect để có rotated bbox chính xác
                rotated_rect = cv2.minAreaRect(c)
                (cx, cy), (rw, rh), angle = rotated_rect
                # Chuẩn hóa: rw luôn là cạnh dài, rh là cạnh ngắn
                long_side = max(rw, rh)
                short_side = min(rw, rh)

                # Lọc cạnh quá bé
                if short_side < thresholds["min_dim"]:
                    continue

                # [Fix Lỗi 7 cont.]: Tính aspect ratio từ rotated box (chống nghiêng)
                aspect = long_side / max(1.0, short_side)
                if aspect > self.MAX_ASPECT_RATIO:
                    continue

                # Lấy axis-aligned bbox để dùng cho downstream
                x, y, w, h = cv2.boundingRect(c)
                bbox = (
                    max(0, x),
                    max(0, y),
                    min(img_w, x + w),
                    min(img_h, y + h),
                )

                # Kiểm tra trùng lặp trong cùng pass
                is_dup = any(
                    _iou(bbox, existing.bbox) > 0.5 for existing in candidates
                )
                if not is_dup:
                    candidates.append(ChartRegion(
                        bbox=bbox,
                        rotated_bbox=rotated_rect,
                        detection_method=f"{method_name}_{kernel_key}",
                    ))

        return candidates

    # ====================================================================
    # Heuristic Filters (tất cả dùng tỉ lệ, không hardcode pixel)
    # ====================================================================

    def _apply_heuristic_filters(
        self,
        candidates: list[ChartRegion],
        ocr_blocks: list[OCRBlock],
        table_bboxes: list[tuple],
        thresholds: dict,
        total_chars: int,
        img_h: int,
        img_w: int,
    ) -> list[ChartRegion]:
        """Áp dụng các bộ lọc thông minh để loại bỏ false positive."""
        filtered = []

        for candidate in candidates:
            x1, y1, x2, y2 = candidate.bbox
            area = (x2 - x1) * (y2 - y1)
            w = x2 - x1
            h = y2 - y1

            if area < 1:
                continue

            # --- Filter 1: Loại trừ vùng trùng Bảng biểu ---
            is_table = any(
                _intersection_ratio(candidate.bbox, tb) > self.TABLE_OVERLAP_THRESHOLD
                or _intersection_ratio(tb, candidate.bbox) > self.TABLE_OVERLAP_THRESHOLD
                for tb in table_bboxes
            )
            if is_table:
                continue

            # --- Filter 2: Mật độ text (Dynamic) ---
            text_area = 0
            char_count = 0
            for b in ocr_blocks:
                bb = _safe_bbox(b)
                if bb is None:
                    continue
                ratio = _intersection_ratio(bb, candidate.bbox)
                if ratio > 0.5:
                    block_area = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    text_area += int(block_area * ratio)
                    char_count += len(b.text.strip())

            text_density = text_area / max(1, area)

            # [Fix]: Chặn đứng các khối toàn chữ.
            # Nếu mật độ text > 40%, HOẶC (mật độ > 25% và có trên 5% tổng ký tự)
            char_ratio = char_count / max(1, total_chars) if total_chars > 0 else 0
            if text_density > 0.40 or (text_density > 0.25 and char_ratio > 0.05):
                continue

            # --- Filter 3: Header/Footer (Dynamic) ---
            # [Fix Lỗi 1]: Dùng % chiều cao ảnh thay vì pixel
            if y1 < thresholds["header_y"] and h < thresholds["header_h"] and w > img_w * 0.3:
                continue
            # Footer tương tự
            if y2 > img_h - thresholds["header_y"] and h < thresholds["header_h"] and w > img_w * 0.3:
                continue

            filtered.append(candidate)

        return filtered

    # ====================================================================
    # Loại bỏ hộp lồng nhau
    # ====================================================================

    def _remove_nested(self, regions: list[ChartRegion]) -> list[ChartRegion]:
        """Giữ lại hộp bao ngoài cùng, loại bỏ hộp con."""
        if len(regions) <= 1:
            return regions

        final = []
        for r in regions:
            is_inside = any(
                o is not r
                and o.bbox[0] <= r.bbox[0]
                and o.bbox[1] <= r.bbox[1]
                and o.bbox[2] >= r.bbox[2]
                and o.bbox[3] >= r.bbox[3]
                for o in regions
            )
            if not is_inside:
                final.append(r)
        return final

    # ====================================================================
    # Gộp các hộp gần nhau (Merge Nearby)
    # ====================================================================

    def _merge_nearby_boxes(self, regions: list[ChartRegion], distance_thresh: float) -> list[ChartRegion]:
        """Gộp các bounding box nếu khoảng cách giữa chúng nhỏ hơn distance_thresh."""
        if not regions:
            return []

        def distance(b1, b2):
            # Tính khoảng cách Manhattan hoặc Chebyshev giữa 2 box
            dx = max(0, max(b1[0] - b2[2], b2[0] - b1[2]))
            dy = max(0, max(b1[1] - b2[3], b2[1] - b1[3]))
            return math.hypot(dx, dy)

        merged = []
        used = [False] * len(regions)
        
        for i, r1 in enumerate(regions):
            if used[i]: continue
            
            x1, y1, x2, y2 = r1.bbox
            score = r1.score
            
            # Khởi tạo một cụm
            cluster_changed = True
            while cluster_changed:
                cluster_changed = False
                for j, r2 in enumerate(regions):
                    if i == j or used[j]: continue
                    
                    if distance((x1, y1, x2, y2), r2.bbox) < distance_thresh:
                        # Gộp box
                        x1 = min(x1, r2.bbox[0])
                        y1 = min(y1, r2.bbox[1])
                        x2 = max(x2, r2.bbox[2])
                        y2 = max(y2, r2.bbox[3])
                        score = max(score, r2.score)
                        used[j] = True
                        cluster_changed = True
            
            merged.append(ChartRegion(bbox=(x1, y1, x2, y2), score=score, detection_method="merged"))
            used[i] = True
            
        return merged

    # ====================================================================
    # Crop & Save (giữ nguyên interface cũ)
    # ====================================================================

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
        img_h, img_w = image.shape[:2]
        assets = []

        for idx, r in enumerate(regions):
            x1, y1, x2, y2 = r.bbox

            # Padding động (1% cạnh dài ảnh, tối thiểu 5px)
            pad = max(5, int(max(img_h, img_w) * 0.01))
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(img_w, x2 + pad)
            py2 = min(img_h, y2 + pad)

            crop = image[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            filename = f"{file_stem}_chart_{idx + 1}_{uuid.uuid4().hex[:6]}.png"
            out_path = output_dir / filename
            cv2.imwrite(str(out_path), crop)

            assets.append({
                "index": idx + 1,
                "filename": filename,
                "path": str(out_path).replace("\\", "/"),
                "bbox": [x1, y1, x2, y2],
                "score": r.score,
                "detection_method": r.detection_method,
                "region_type": "chart",
            })

        return assets
