"""
Doc Anchor AI — OCR Layout Router
Tự động chọn hướng xử lý tối ưu cho ảnh/tài liệu sau bước OCR thô.

Router này không thay đổi, không dịch, không chuẩn hóa dữ liệu OCR.
Nó chỉ đọc signal từ OCR blocks để quyết định strategy:
- clean_text: tài liệu chủ yếu là chữ
- table_rich: tài liệu chủ yếu là bảng/số liệu
- mixed_layout: vừa text vừa bảng
- noisy_scan: ảnh/ocr rủi ro nhưng vẫn xử lý được
- critical_fail: gần như không đủ tín hiệu OCR để tin output

Đồng thời router trả về recommended_temperature cho VLM dựa trên đặc thù layout:
- Bảng biểu / scan nhiễu: temperature = 0.0 (cấm sáng tạo, bám sát OCR)
- Text thuần: temperature = 0.2 (cho phép hành văn mượt mà cho RAG)
- Hỗn hợp: temperature = 0.1 (cân bằng)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from src.ingestion.schemas import OCRBlock


# Chiến lược nhiệt độ VLM theo layout mode
LAYOUT_TEMPERATURE_MAP: dict[str, float] = {
    "critical_fail": 0.0,       # Ảnh quá tệ -> bắt buộc bám sát OCR thô
    "noisy_scan": 0.0,          # Scan nhiễu -> cấm sáng tạo số liệu
    "table_rich": 0.0,          # Nhiều bảng biểu -> ép khuôn cấu trúc tuyệt đối
    "clean_text": 0.2,          # Văn bản thuần -> cho phép hành văn mượt mà cho RAG
    "mixed_layout": 0.1,        # Bố cục hỗn hợp -> mức an toàn cân bằng
}


@dataclass
class LayoutRoute:
    """Kết quả route layout/chất lượng cho một tài liệu OCR."""

    layout_mode: str
    quality_class: str
    quality_score: float
    issue_flags: list[str] = field(default_factory=list)
    prompt_profile: str = "MIXED_LAYOUT"
    recommended_strategy: str = "mixed_layout_ocr"
    recommended_temperature: float = 0.1
    metrics: dict[str, float] = field(default_factory=dict)


class LayoutRouter:
    """Router tự động chọn nhánh OCR/VLM tối ưu từ OCR blocks."""

    TABLE_MARKERS = {
        "stt",
        "st",
        "qty",
        "quantity",
        "amount",
        "total",
        "date",
        "revenue",
        "profit",
        "tax",
        "vat",
        "doanh thu",
        "lợi nhuận",
        "loi nhuan",
        "tổng",
        "tong",
        "thuế",
        "thue",
        "số tiền",
        "so tien",
        "đơn giá",
        "don gia",
        "thành tiền",
        "thanh tien",
    }

    def route(self, ocr_blocks: list[OCRBlock], image_shape: tuple | None = None) -> LayoutRoute:
        """Phân tích OCR blocks và trả về route xử lý tối ưu (bao gồm recommended_temperature)."""
        texts = [block.text.strip() for block in ocr_blocks if block.text and block.text.strip()]
        total_blocks = len(texts)
        issue_flags: list[str] = []

        if total_blocks == 0:
            return LayoutRoute(
                layout_mode="critical_fail",
                quality_class="critical_fail",
                quality_score=0.0,
                issue_flags=["no_ocr_text"],
                prompt_profile="MIXED_LAYOUT",
                recommended_strategy="manual_review_or_rescan",
                recommended_temperature=LAYOUT_TEMPERATURE_MAP["critical_fail"],
                metrics={"total_blocks": 0.0},
            )

        avg_conf = sum(block.confidence for block in ocr_blocks) / max(len(ocr_blocks), 1)
        numeric_blocks = sum(1 for text in texts if re.search(r"\d", text))
        money_like = sum(1 for text in texts if re.search(r"\d[\d.,]*", text))
        long_text_blocks = sum(1 for text in texts if len(text) >= 25)
        very_short_blocks = sum(1 for text in texts if len(text) <= 3)
        marker_blocks = sum(1 for text in texts if self._looks_like_table_marker(text))
        aligned_rows = self._estimate_aligned_rows(ocr_blocks)

        numeric_ratio = numeric_blocks / total_blocks
        money_ratio = money_like / total_blocks
        long_text_ratio = long_text_blocks / total_blocks
        short_ratio = very_short_blocks / total_blocks
        marker_ratio = marker_blocks / total_blocks
        aligned_ratio = aligned_rows / max(total_blocks, 1)

        # --- Chỉ số đa ngành mới ---
        ext_metrics = self._calculate_extended_metrics(ocr_blocks, texts, image_shape)
        density_ratio = ext_metrics["density_ratio"]
        abbreviation_ratio = ext_metrics["abbreviation_ratio"]

        if avg_conf < 0.45:
            issue_flags.append("low_ocr_confidence")
        if short_ratio > 0.45:
            issue_flags.append("many_short_tokens")
        if total_blocks <= 2:
            issue_flags.append("too_few_ocr_blocks")
        if abbreviation_ratio > 0.35:
            issue_flags.append("high_abbreviation_ratio")

        # Decision Tree logic
        # Cấp 1: Critical Check (Độ toàn vẹn)
        if avg_conf < 0.35:
            layout_mode = "critical_fail"
            quality_class = "critical_fail"
            prompt_profile = "MIXED_LAYOUT"
            strategy = "manual_review_or_rescan"
        # Cấp 2: Quality Check (Độ nhiễu)
        elif avg_conf < 0.65:
            layout_mode = "noisy_scan"
            quality_class = "noisy_scan"
            prompt_profile = "ANTI_HALLUCINATION_MIXED"
            strategy = "light_preprocess_then_preserve_source_ocr"
        # Cấp 3: Layout Check (Bố cục)
        elif marker_ratio > 0.15 and aligned_ratio > 0.4:
            layout_mode = "table_rich"
            quality_class = "table_rich"
            prompt_profile = "TABLE_PRIORITY"
            strategy = "table_first_ocr_with_source_preservation"
        elif numeric_ratio < 0.1 and long_text_ratio > 0.7:
            layout_mode = "clean_text"
            quality_class = "clean_text"
            prompt_profile = "TEXT_PRIORITY"
            strategy = "text_first_ocr_with_source_preservation"
        else:
            layout_mode = "mixed_layout"
            quality_class = "mixed_layout"
            prompt_profile = "MIXED_LAYOUT"
            strategy = "split_text_and_table_then_vlm_structure"

        # --- Dynamic Temperature ---
        # Lấy temperature từ bảng mapping dựa trên layout_mode
        recommended_temp = LAYOUT_TEMPERATURE_MAP.get(layout_mode, 0.1)
        # Override: Nếu tài liệu chứa nhiều từ viết tắt/mã (y tế, kỹ thuật),
        # ép temperature = 0.0 bất kể layout_mode để chống VLM đoán mò.
        if abbreviation_ratio > 0.35:
            recommended_temp = 0.0

        quality_score = self._score_quality(
            avg_conf=avg_conf,
            total_blocks=total_blocks,
            layout_mode=layout_mode,
            issue_count=len(issue_flags),
        )

        return LayoutRoute(
            layout_mode=layout_mode,
            quality_class=quality_class,
            quality_score=quality_score,
            issue_flags=issue_flags,
            prompt_profile=prompt_profile,
            recommended_strategy=strategy,
            recommended_temperature=recommended_temp,
            metrics={
                "total_blocks": float(total_blocks),
                "avg_ocr_confidence": round(avg_conf, 3),
                "numeric_ratio": round(numeric_ratio, 3),
                "money_ratio": round(money_ratio, 3),
                "long_text_ratio": round(long_text_ratio, 3),
                "short_ratio": round(short_ratio, 3),
                "marker_ratio": round(marker_ratio, 3),
                "aligned_ratio": round(aligned_ratio, 3),
                "density_ratio": round(density_ratio, 3),
                "abbreviation_ratio": round(abbreviation_ratio, 3),
            },
        )

    def _calculate_extended_metrics(
        self,
        ocr_blocks: list[OCRBlock],
        texts: list[str],
        image_shape: tuple | None,
    ) -> dict[str, float]:
        """
        Tính các chỉ số đa ngành mở rộng:
        - density_ratio: Mật độ ký tự trên diện tích ảnh (nhận diện hợp đồng/văn bản pháp lý dày đặc).
        - abbreviation_ratio: Tỷ lệ từ viết hoa hoàn toàn (nhận diện bệnh án, hóa đơn kỹ thuật, mã phụ tùng).
        """
        total_chars = sum(len(t) for t in texts)

        # 1. Density ratio: chars per 1000 pixel² diện tích ảnh
        if image_shape and len(image_shape) >= 2:
            img_h, img_w = image_shape[0], image_shape[1]
            img_area = max(img_h * img_w, 1)
            density_ratio = total_chars / (img_area / 1000)
        else:
            # Fallback: ước lượng density từ tổng block và ký tự
            density_ratio = total_chars / max(len(texts), 1)

        # 2. Abbreviation ratio: % từ viết hoa hoàn toàn (>1 ký tự, chỉ chữ cái)
        words: list[str] = []
        for t in texts:
            words.extend(t.split())
        uppercase_words = [w for w in words if w.isupper() and len(w) > 1 and w.isalpha()]
        abbreviation_ratio = len(uppercase_words) / max(len(words), 1)

        return {
            "density_ratio": density_ratio,
            "abbreviation_ratio": abbreviation_ratio,
        }

    def _looks_like_table_marker(self, text: str) -> bool:
        lowered = text.lower()
        for marker in self.TABLE_MARKERS:
            # Use word boundaries to avoid matching substrings like 'station' for 'st'
            if re.search(r'\b' + re.escape(marker) + r'\b', lowered, re.UNICODE):
                return True
        return False

    def _estimate_aligned_rows(self, ocr_blocks: list[OCRBlock]) -> int:
        """Ước lượng số block nằm trên các hàng gần nhau bằng bbox y-center."""
        centers: list[float] = []
        heights: list[float] = []
        for block in ocr_blocks:
            if not block.bbox:
                continue
            ys = [point[1] for point in block.bbox if len(point) >= 2]
            if ys:
                centers.append(sum(ys) / len(ys))
                heights.append(max(ys) - min(ys))

        if len(centers) < 4:
            return 0

        # Tính khoảng cách động dựa trên chiều cao trung bình của block
        dynamic_epsilon = (sum(heights) / len(heights)) * 0.4 if heights else 12.0

        centers.sort()
        grouped = 0
        current_group = [centers[0]]
        for center in centers[1:]:
            if abs(center - current_group[-1]) <= dynamic_epsilon:
                current_group.append(center)
            else:
                if len(current_group) >= 2:
                    grouped += len(current_group)
                current_group = [center]
        if len(current_group) >= 2:
            grouped += len(current_group)
        return grouped

    def _score_quality(
        self,
        *,
        avg_conf: float,
        total_blocks: int,
        layout_mode: str,
        issue_count: int,
    ) -> float:
        score = avg_conf * 0.55
        score += 0.15 if total_blocks >= 4 else 0.0
        
        # Thưởng điểm cho layout hợp lệ rõ ràng
        if layout_mode in ["clean_text", "table_rich", "mixed_layout"]:
            score += 0.15
            
        score -= min(issue_count * 0.08, 0.24)
        return round(max(0.0, min(1.0, score)), 3)
