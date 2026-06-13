"""
Doc Anchor AI — PaddleOCR Extractor
Local multilingual OCR engine optimized as a companion to VLM extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os

from src.config import settings, get_logger
from src.ingestion.schemas import OCRBlock

# PaddleOCR 3.x + paddlepaddle 3.x trên Windows CPU có thể lỗi oneDNN/PIR:
# "ConvertPirAttribute2RuntimeAttribute not support ...".
# Tắt các optimization này trước khi import/khởi tạo PaddleOCR để ưu tiên đường chạy ổn định.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_DISABLE_MKLDNN", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

logger = get_logger(__name__)


class PaddleOCRExtractor:
    """Wrapper nhẹ quanh PaddleOCR, trả về OCRBlock chuẩn hóa."""

    def __init__(self, languages: list[str] | None = None, use_gpu: bool | None = None):
        self.languages = languages or settings.get("ingestion.ocr.languages", ["vi", "en"])
        self.use_gpu = settings.get("ingestion.ocr.use_gpu", False) if use_gpu is None else use_gpu
        self._ocr = None

    def _load_engine(self):
        """Lazy-load PaddleOCR để tránh tăng thời gian import app.

        PaddleOCR 3.x official API:
        - PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, ...)
        - ocr.predict(image_path)

        PaddleOCR 2.x fallback:
        - PaddleOCR(use_angle_cls=True, lang=...)
        - ocr.ocr(image_path, cls=True)
        """
        if self._ocr is not None:
            return self._ocr

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR chưa được cài. Hãy cài dependencies trong requirements.txt."
            ) from exc

        lang = self._select_primary_lang(self.languages)
        device = "gpu" if self.use_gpu else "cpu"
        logger.info(f"Khởi tạo PaddleOCR lang={lang}, device={device}")

        import os
        os.environ["FLAGS_enable_pir_api"] = "0"

        try:
            self._ocr = PaddleOCR(
                lang=lang,
                device=device,
                use_angle_cls=True,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )
            self._api_version = 3
            return self._ocr
        except (TypeError, ValueError) as exc:
            logger.warning(f"PaddleOCR 3.x constructor lỗi tham số, fallback 2.x: {exc}")

        try:
            self._ocr = PaddleOCR(use_angle_cls=True, use_textline_orientation=False, enable_mkldnn=False, lang=lang)
        except (TypeError, ValueError):
            try:
                self._ocr = PaddleOCR(lang=lang, enable_mkldnn=False)
            except Exception as inner_exc:
                raise RuntimeError(f"Không thể khởi tạo PaddleOCR: {inner_exc}")
        self._api_version = 2
        return self._ocr

    @staticmethod
    def _select_primary_lang(languages: list[str]) -> str:
        """Map config language list sang PaddleOCR lang hợp lệ."""
        lowered = {lang.lower() for lang in languages}
        if "ch" in lowered or "zh" in lowered or "chinese" in lowered:
            return "ch"
        if "japan" in lowered or "ja" in lowered or "japanese" in lowered:
            return "japan"
        if "korean" in lowered or "ko" in lowered:
            return "korean"
        if "vi" in lowered or "vietnamese" in lowered:
            return "vi"
        return "en"

    def extract(self, image_path: str | Path) -> list[OCRBlock]:
        """Chạy OCR trên ảnh và trả danh sách text blocks."""
        path = Path(image_path)
        logger.info(f"Bắt đầu PaddleOCR: {path.name}")
        ocr = self._load_engine()

        # Thử gọi predict (PaddleX/V3) trước, nếu không có thì gọi ocr()
        if hasattr(ocr, "predict"):
            raw_result = ocr.predict(str(path))
        else:
            try:
                raw_result = ocr.ocr(str(path), cls=True)
            except TypeError:
                raw_result = ocr.ocr(str(path))
                
        # Determine format dynamically
        blocks = []
        is_v3 = False
        
        # Generator or list of dicts (PaddleX/V3)
        if hasattr(raw_result, "__next__") or (isinstance(raw_result, list) and len(raw_result) > 0 and (isinstance(raw_result[0], dict) or hasattr(raw_result[0], "json"))):
            is_v3 = True
            
        if is_v3:
            blocks = self._parse_paddle3_result(raw_result)
        else:
            blocks = self._parse_legacy_ocr_result(raw_result)
            
        # --- FIX FORMS KIE: GỘP CÁC KHỐI CHỮ NẰM NGANG CÙNG DÒNG ---
        # Ngăn chặn việc Key và Value bị đứt gãy do khoảng trắng dài hoặc dấu chấm `......`
        blocks = self._merge_horizontal_blocks(blocks)

        logger.info(f"PaddleOCR hoàn tất: {len(blocks)} blocks (sau khi gộp ngang)")
        return blocks
        
    def _merge_horizontal_blocks(self, blocks: list[OCRBlock]) -> list[OCRBlock]:
        """Gộp các block nằm trên cùng một dòng ngang, nối bằng khoảng trắng."""
        if not blocks:
            return []
            
        def get_cy(b):
            return sum(p[1] for p in b.bbox) / 4.0 if b.bbox and len(b.bbox) >= 4 else 0
            
        def get_cx(b):
            return sum(p[0] for p in b.bbox) / 4.0 if b.bbox and len(b.bbox) >= 4 else 0
            
        def get_h(b):
            ys = [p[1] for p in b.bbox] if b.bbox and len(b.bbox) >= 4 else [0, 0]
            return max(ys) - min(ys)
            
        def get_x_bounds(b):
            xs = [p[0] for p in b.bbox] if b.bbox and len(b.bbox) >= 4 else [0, 0]
            return min(xs), max(xs)

        # Sort theo Y (top-to-bottom), trên cùng 1 dòng thì sort theo X (left-to-right)
        # Nhóm thành các dòng (Sai số Y < 0.5 chiều cao)
        sorted_blocks = sorted(blocks, key=lambda b: get_cy(b))
        
        lines = []
        current_line = []
        for b in sorted_blocks:
            if not current_line:
                current_line.append(b)
                continue
            prev = current_line[0]
            # Nếu tâm Y chênh lệch quá ít (cùng dòng)
            if abs(get_cy(b) - get_cy(prev)) < get_h(prev) * 0.6:
                current_line.append(b)
            else:
                lines.append(sorted(current_line, key=lambda cb: get_cx(cb)))
                current_line = [b]
        if current_line:
            lines.append(sorted(current_line, key=lambda cb: get_cx(cb)))
            
        merged_blocks = []
        for line in lines:
            if not line: continue
            merged = line[0]
            for next_b in line[1:]:
                _, xmax_prev = get_x_bounds(merged)
                xmin_next, xmax_next = get_x_bounds(next_b)
                gap = xmin_next - xmax_prev
                h_prev = get_h(merged)
                
                # Gộp nếu khoảng cách hợp lý hoặc là cùng dòng Form
                # Ta tự động nối bằng khoảng trắng. Nếu cách rất xa (> 3 lần h), dùng " ... "
                sep = " "
                if gap > h_prev * 3:
                    sep = " ... "
                    
                new_bbox = [
                    [min(p[0] for p in merged.bbox), min(p[1] for p in merged.bbox)],
                    [max(p[0] for p in next_b.bbox), min(p[1] for p in merged.bbox)],
                    [max(p[0] for p in next_b.bbox), max(p[1] for p in next_b.bbox)],
                    [min(p[0] for p in merged.bbox), max(p[1] for p in next_b.bbox)]
                ]
                merged = OCRBlock(
                    text=merged.text + sep + next_b.text,
                    confidence=(merged.confidence + next_b.confidence) / 2.0,
                    bbox=new_bbox
                )
            merged_blocks.append(merged)
            
        return merged_blocks

    def _parse_legacy_ocr_result(self, raw_result: list[Any]) -> list[OCRBlock]:
        """Parse output kiểu PaddleOCR 2.x."""
        blocks: list[OCRBlock] = []
        for page in raw_result or []:
            for line in page or []:
                if not line or len(line) < 2:
                    continue
                bbox = line[0]
                text_info = line[1]
                text = text_info[0] if text_info else ""
                confidence = float(text_info[1]) if len(text_info) > 1 else 0.0
                if text.strip():
                    blocks.append(OCRBlock(text=text.strip(), confidence=confidence, bbox=bbox))
        return blocks

    def _parse_paddle3_result(self, raw_result: list[Any]) -> list[OCRBlock]:
        """Parse output kiểu PaddleOCR 3.x/PaddleX."""
        blocks: list[OCRBlock] = []
        for item in raw_result or []:
            data = getattr(item, "json", None)
            if callable(data):
                data = data()
            if not isinstance(data, dict):
                data = item if isinstance(item, dict) else {}

            result = data.get("res", data)
            texts = result.get("rec_texts") or result.get("texts") or []
            scores = result.get("rec_scores") or result.get("scores") or []
            boxes = result.get("rec_boxes") or result.get("dt_polys") or result.get("boxes") or []
            
            if isinstance(texts, str): texts = [texts]
            if isinstance(scores, (float, int)): scores = [scores]
            # boxes might be a 3D array or a 2D array if single box
            if isinstance(boxes, list) and len(boxes) > 0 and not isinstance(boxes[0], list):
                if len(boxes) == 4 or len(boxes) == 8: # likely a single flat box
                    boxes = [boxes]

            for idx, text in enumerate(texts):
                confidence = float(scores[idx]) if idx < len(scores) else 0.0
                raw_box = boxes[idx] if idx < len(boxes) else []
                if hasattr(raw_box, "tolist"):
                    bbox = raw_box.tolist()
                elif isinstance(raw_box, str):
                    import json
                    try:
                        bbox = json.loads(raw_box)
                    except Exception:
                        bbox = []
                else:
                    bbox = raw_box
                    
                # Normalize flat bbox to 2D list [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                if bbox and isinstance(bbox[0], (int, float)):
                    if len(bbox) == 4:
                        bbox = [
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]]
                        ]
                    elif len(bbox) >= 8:
                        bbox = [
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[4], bbox[5]],
                            [bbox[6], bbox[7]]
                        ]

                if str(text).strip():
                    blocks.append(OCRBlock(text=str(text).strip(), confidence=confidence, bbox=bbox))
        return blocks
