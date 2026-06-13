"""
Doc Anchor AI — Local Hybrid Ingestion Pipeline
Kết hợp PaddleOCR, OpenCV Morphology, VLM và validator cho trích xuất hình ảnh tài liệu.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from src.config import settings, get_logger

from src.ingestion.image_preprocessor import ImagePreprocessor
from src.ingestion.vlm_ocr import VLMOCRProcessor
from src.ingestion.extractors import PaddleOCRExtractor
from src.ingestion.prompts import build_financial_extraction_prompt
from src.ingestion.routing import LayoutRoute, LayoutRouter
from src.ingestion.schemas import ConfidenceReport, ExtractedTable, ExtractionResult, OCRBlock
from src.ingestion.table_reconstruction import reconstruct_tables_from_ocr
from src.ingestion.table_region_detection import TableRegionDetector, crop_ocr_blocks_to_region
from src.ingestion.chart_region_detector import ChartRegionDetector

logger = get_logger(__name__)


class IngestionPipeline:
    """Pipeline local-first cho OCR/VLM extraction + save output + indexing."""

    def __init__(self):
        self.processed_dir = Path(settings.get("storage.data.processed_dir", "data/processed"))
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.layout_router = LayoutRouter()

    def run(self, file_path: str | Path) -> ExtractionResult:
        """Chạy extraction end-to-end cho file ảnh và trả ExtractionResult."""
        start_time = time.time()
        path = Path(file_path)
        logger.info(f"[IngestionPipeline] Processing image: {path.name}")

        result = self._extract_image_like(path)

        result.quality_class, result.quality_score, result.issue_flags, result.recommended_action = self._classify_quality(result)
        result.confidence = self._calculate_confidence(result)
        result.requires_human_review = self._requires_human_review(result)
        result.metadata["elapsed_seconds"] = round(time.time() - start_time, 2)
        result.metadata["input_type"] = "image"
        return result

    def save_outputs(self, result: ExtractionResult) -> tuple[Path, Path]:
        """Lưu Markdown và JSON cạnh nhau trong data/processed."""
        stem = Path(result.source_file).stem
        md_path = self.processed_dir / f"{stem}.md"
        json_path = self.processed_dir / f"{stem}.json"

        md_content = result.markdown
        
        # === Chart Isolation (Phase 6b) ===
        if result.chart_assets:
            md_content += "\n\n## 📊 Biểu đồ trong tài liệu\n\n"
            for asset in result.chart_assets:
                filename = Path(asset['path']).name
                md_content += f"![Biểu đồ {asset['index']}](assets/{filename})\n\n"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Đã lưu output: {md_path}, {json_path}")
        return md_path, json_path

    def _extract_image_like(self, path: Path) -> ExtractionResult:
        """Ảnh/PDF scan: preprocess nhẹ -> PaddleOCR -> VLM JSON-first."""
        preprocessor = ImagePreprocessor()
        cleaned_img_path = self.processed_dir / f"cleaned_{path.name}"
        preprocessor.process(path, cleaned_img_path)

        ocr_blocks: list[OCRBlock] = []
        if settings.get("ingestion.ocr.enabled", True):
            try:
                ocr_blocks = PaddleOCRExtractor().extract(cleaned_img_path)
            except Exception as exc:
                logger.warning(f"PaddleOCR failed, tiếp tục chỉ với VLM: {exc}")

        route = self.layout_router.route(ocr_blocks)
        prompt = build_financial_extraction_prompt(
            ocr_blocks,
            prompt_profile=route.prompt_profile,
        )
        # Tính năng 3: Dynamic Temperature — truyền temperature từ LayoutRouter
        vlm_response = VLMOCRProcessor().extract(
            cleaned_img_path,
            system_prompt=prompt,
            ocr_blocks=ocr_blocks,
            temperature=route.recommended_temperature,
        )

        table_regions = TableRegionDetector().detect(cleaned_img_path) if ocr_blocks else []
        reconstructed_tables, reconstructed_table_md, table_reconstruction_metrics = self._reconstruct_tables_with_regions(
            ocr_blocks,
            table_regions,
        )

        # === Chart Isolation (Phase 6b) ===
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

        # Tính năng 4: OpenCV-VLM Table Validator — validate cấu trúc bảng trước khi normalize
        vlm_response = self._validate_and_fallback_tables(vlm_response, reconstructed_tables)

        markdown, sanitize_warnings = self._normalize_markdown(
            vlm_response,
            tables=reconstructed_tables,
            ocr_blocks=ocr_blocks,
            fallback_text=vlm_response,
        )
        # CHỈ dùng reconstructed_table (OpenCV) nếu VLM hoàn toàn không trả về text.
        if not markdown.strip() and reconstructed_table_md:
            markdown = reconstructed_table_md
            sanitize_warnings.append("VLM rỗng, fallback dùng text thô của OCR bbox.")
            
        issue_flags = [*route.issue_flags, *sanitize_warnings]

        table_confidence = float(table_reconstruction_metrics.get("table_confidence", 0.0) or 0.0)
        table_threshold = 0.5
        final_representation = "table" if reconstructed_tables else "linear_text"
        table_reconstruction_status = "success" if reconstructed_tables else "linear_fallback"

        result = ExtractionResult(
            source_file=path.name,
            document_type="unknown",
            languages=[],
            markdown=markdown,
            fields={},
            tables=reconstructed_tables,
            chart_assets=chart_assets,
            raw_ocr=ocr_blocks,
            uncertain_tokens=self._extract_uncertain_tokens(markdown),
            quality_class=route.quality_class,
            quality_score=route.quality_score,
            issue_flags=issue_flags,
            recommended_action=route.recommended_strategy,
            metadata={
                "engine": "paddleocr+vlm_markdown",
                "cleaned_image": str(cleaned_img_path),
                "vlm_raw_response": vlm_response,
                "sanitize_warnings": sanitize_warnings,
                "layout_mode": route.layout_mode,
                "prompt_profile": route.prompt_profile,
                "recommended_strategy": route.recommended_strategy,
                "layout_metrics": route.metrics,
                "table_reconstruction": table_reconstruction_metrics,
                "table_reconstruction_status": table_reconstruction_status,
                "table_confidence_score": round(table_confidence, 3),
                "table_confidence_threshold": table_threshold,
                "final_representation": final_representation,
                "table_regions": [
                    {"bbox": region.bbox, "score": region.score, "source": region.source}
                    for region in table_regions
                ],
                "chart_regions": [
                    {"bbox": r.bbox, "score": r.score} for r in chart_regions
                ],
                "chart_assets": chart_assets,
            },
        )
        return result


    def _reconstruct_tables_with_regions(
        self,
        ocr_blocks: list[OCRBlock],
        table_regions: list[Any],
    ) -> tuple[list[ExtractedTable], str, dict[str, Any]]:
        """Ưu tiên reconstruct trong từng vùng bảng đã detect, fallback toàn trang nếu fail."""
        all_tables: list[ExtractedTable] = []
        markdown_parts: list[str] = []
        region_metrics: list[dict[str, Any]] = []

        for idx, region in enumerate(table_regions, start=1):
            region_blocks = crop_ocr_blocks_to_region(ocr_blocks, region)
            tables, markdown, metrics = reconstruct_tables_from_ocr(region_blocks)
            metrics = {
                **metrics,
                "region_index": idx,
                "region_bbox": region.bbox,
                "region_score": region.score,
                "ocr_blocks": len(region_blocks),
            }
            region_metrics.append(metrics)
            if not tables or not markdown.strip():
                continue

            for table in tables:
                table.name = f"reconstructed_table_{idx}"
            all_tables.extend(tables)
            markdown_parts.append(f"## reconstructed_table_{idx}\n{markdown.strip()}")

        if all_tables and markdown_parts:
            return all_tables, "\n\n".join(markdown_parts), {
                "detected": True,
                "source": "table_region_morphology+ocr_bbox_heuristic",
                "regions": region_metrics,
                "tables": len(all_tables),
            }

        fallback_tables, fallback_md, fallback_metrics = reconstruct_tables_from_ocr(ocr_blocks)
        return fallback_tables, fallback_md, {
            **fallback_metrics,
            "source": "full_page_ocr_bbox_heuristic",
            "regions": region_metrics,
            "fallback_used": True,
        }

    def _normalize_markdown(
        self,
        raw_markdown: Any,
        *,
        tables: list[ExtractedTable],
        ocr_blocks: list[OCRBlock],
        fallback_text: str,
    ) -> tuple[str, list[str]]:
        """Đảm bảo markdown là nội dung tài liệu thật, không phải JSON/code fence."""
        warnings: list[str] = []
        markdown = str(raw_markdown or "").strip()

        if markdown.startswith("```markdown"):
            markdown = markdown[len("```markdown"):].strip()
        elif markdown.startswith("```"):
            markdown = markdown[3:].strip()
        if markdown.endswith("```"):
            markdown = markdown[:-3].strip()

        if markdown.lstrip().startswith("{"):
            warnings.append("VLM trả markdown dạng JSON; đã dựng lại markdown từ tables/OCR.")
            markdown = ""

        if not markdown and tables:
            markdown = self._tables_to_markdown(tables)

        if not markdown and ocr_blocks:
            warnings.append("Không có markdown hợp lệ; dùng raw OCR text làm fallback.")
            markdown = "\n".join(f"- {block.text}" for block in ocr_blocks if block.text.strip())

        if markdown:
            # 4. Cleanup markdown
            logger.info("Dọn dẹp Markdown đầu ra...")
            cleaned_markdown, cleanup_warnings = self._cleanup_markdown(markdown, ocr_blocks)
            markdown = cleaned_markdown
            warnings.extend(cleanup_warnings)

        if not markdown:
            warnings.append("Không có markdown/OCR fallback hợp lệ; dùng raw VLM response.")
            markdown = fallback_text.strip()

        return markdown, warnings

    def _tables_to_markdown(self, tables: list[ExtractedTable]) -> str:
        """Dựng Markdown table đơn giản từ structured tables."""
        sections: list[str] = []
        for table in tables:
            if not table.columns:
                continue
            sections.append(f"## {table.name}")
            sections.append("| " + " | ".join(table.columns) + " |")
            sections.append("| " + " | ".join("---" for _ in table.columns) + " |")
            for row in table.rows:
                sections.append("| " + " | ".join(str(row.get(col, "")) for col in table.columns) + " |")
            sections.append("")
        return "\n".join(sections).strip()

    def _extract_uncertain_tokens(self, markdown: str) -> list[str]:
        """Lấy các token chưa chắc chắn được VLM đánh dấu trong Markdown."""
        matches = re.findall(r"\[uncertain:\s*([^\]]+)\]", markdown, flags=re.IGNORECASE)
        tokens = []
        for item in matches:
            token = item.strip()
            if token:
                tokens.append(token)
        return tokens

    # ====================================================================
    # Tính năng 4: OpenCV-VLM Table Validator
    # ====================================================================

    def _analyze_vlm_table_structure(self, markdown_text: str) -> list[dict]:
        """
        Phân tích chuỗi Markdown của VLM để tìm các khối bảng và đếm số cột thực tế.
        Trả về danh sách các bảng kèm metadata cấu trúc.
        """
        # Regex tìm các khối dòng liên tiếp có chứa dấu |
        table_blocks = re.findall(r'((?:^\|.+?\|$\n?)+)', markdown_text, re.MULTILINE)

        analyzed_tables = []
        for block in table_blocks:
            lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
            if len(lines) < 2:  # Bảng phải có ít nhất tiêu đề + dòng phân cách |---|
                continue

            # Đếm số lượng cột dựa trên số dấu pipe '|' ở từng dòng
            # Một dòng chuẩn | A | B | C | sẽ có 4 dấu '|' -> 3 cột
            cols_per_row = [
                line.count('|') - 1
                for line in lines
                if not re.match(r'^\|[\s\-\|]+\|$', line)
            ]

            if not cols_per_row:
                continue

            # Kiểm tra tính đồng nhất (VLM có bị gõ thiếu dấu | ở dòng nào không)
            is_broken = len(set(cols_per_row)) > 1
            avg_cols = max(cols_per_row) if cols_per_row else 0

            analyzed_tables.append({
                "raw_block": block,
                "is_broken": is_broken,
                "detected_columns": avg_cols,
                "total_rows": len(lines),
            })

        return analyzed_tables

    def _validate_and_fallback_tables(
        self,
        vlm_markdown: str,
        reconstructed_tables: list[ExtractedTable],
    ) -> str:
        """
        Chốt chặn Validate: Nếu VLM sinh bảng lỗi hoặc lệch cấu trúc cột so với OpenCV,
        hệ thống sẽ vứt bảng VLM và swap bằng Reconstructed Table từ OCR Blocks.
        """
        if not reconstructed_tables:
            return vlm_markdown  # Không có bảng từ OpenCV để đối chiếu, tin tưởng VLM

        # 1. Quét cấu trúc bảng của VLM
        vlm_tables = self._analyze_vlm_table_structure(vlm_markdown)

        # Nếu OpenCV tìm thấy bảng nhưng VLM hoàn toàn không dựng bảng (rã thành text)
        if not vlm_tables and reconstructed_tables:
            # P0 Fix: Kiểm tra xem VLM đã rã bảng thành bullet list có data chưa.
            # Nếu VLM đã trả bullet list có chứa số liệu → tin tưởng VLM, KHÔNG đắp thêm bảng rác.
            vlm_has_structured_data = self._vlm_has_bullet_data(vlm_markdown)
            if vlm_has_structured_data:
                logger.info(
                    "[Table Validator] VLM rã bảng thành bullet list có data. "
                    "Giữ nguyên output VLM, bỏ qua reconstructed_table."
                )
                return vlm_markdown
            fallback_md = "\n\n"
            for table in reconstructed_tables:
                fallback_md += self._tables_to_markdown([table]) + "\n\n"
            logger.warning("[Table Validator] VLM không sinh bảng, fallback toàn bộ từ OpenCV.")
            return vlm_markdown + fallback_md

        # 2. Đối chiếu từng cặp bảng (mapping theo thứ tự xuất hiện từ trên xuống)
        final_markdown = vlm_markdown

        for idx, cv_table in enumerate(reconstructed_tables):
            if idx >= len(vlm_tables):
                # VLM bỏ sót bảng thứ n mà OpenCV tìm thấy -> Chèn bổ sung
                cv_md = self._tables_to_markdown([cv_table])
                final_markdown += f"\n\n{cv_md}"
                logger.warning(f"[Table Validator] Bảng {idx+1} bị VLM bỏ sót, chèn bổ sung.")
                continue

            vlm_t = vlm_tables[idx]
            cv_expected_cols = len(cv_table.columns) if cv_table.columns else 0

            # Kích hoạt Fallback nếu:
            # A: VLM bị gõ thiếu/thừa dấu pipe giữa các hàng (is_broken)
            # B: Số cột VLM lệch >= 2 so với lưới OpenCV
            if vlm_t["is_broken"] or (cv_expected_cols > 0 and abs(vlm_t["detected_columns"] - cv_expected_cols) >= 2):
                cv_secure_md = self._tables_to_markdown([cv_table])
                final_markdown = final_markdown.replace(vlm_t["raw_block"], f"\n{cv_secure_md}\n")
                logger.warning(
                    f"[Table Validator] Bảng {idx+1} triggers Fallback. "
                    f"VLM Cols: {vlm_t['detected_columns']} vs OpenCV Expected: {cv_expected_cols}"
                )

        return final_markdown

    def _vlm_has_bullet_data(self, markdown: str) -> bool:
        """Kiểm tra VLM đã rã bảng thành bullet list có chứa số liệu/data thực chưa.

        Nếu True → VLM đã nỗ lực trích xuất data dạng `- Label: value`,
        không cần đắp thêm reconstructed_table (chỉ gây nhiễu).
        """
        lines = markdown.split("\n")
        bullet_lines_with_data = 0
        for line in lines:
            stripped = line.strip()
            # Dòng bullet có chứa số liệu (ít nhất 1 con số)
            if (stripped.startswith("-") or stripped.startswith("*")) and re.search(r"\d", stripped):
                bullet_lines_with_data += 1
        # Nếu có ≥ 5 dòng bullet chứa số → VLM đã rã bảng thành cấu trúc có data
        return bullet_lines_with_data >= 5


    def _cleanup_markdown(self, markdown: str, ocr_blocks: list[OCRBlock]) -> tuple[str, list[str]]:
        """Làm sạch markdown đầu ra để giảm bullet spam và noise OCR."""
        warnings: list[str] = []
        
        # P1 Fix: Strip Model Artifacts — quét và triệt tiêu toàn bộ meta-tag tự sinh của VLM
        # Bao phủ nhiều model: Qwen ([uncertain:]), InternVL (<box>), Gemma ([unrecognized]), v.v.
        text = re.sub(r'\[uncertain:[^\]]*\]', '', markdown)
        text = re.sub(r'\[unrecognized[^\]]*\]', '', text)
        text = re.sub(r'<box>.*?</box>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'</?(?:box|ref|ocr|region)>', '', text)
        # Thu gọn dấu chấm dài (VLM hay bị ảo giác vẽ dấu chấm)
        text = re.sub(r'\.{4,}', '...', text)
        
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        raw_lines = [line.strip() for line in text.split("\n")]
        
        lines = []
        in_table = False
        for line in raw_lines:
            is_table_line = line.startswith("|")
            
            # Tự động chèn dòng trống trước bảng để tránh gộp đoạn văn
            if is_table_line and not in_table:
                if lines and lines[-1] != "":
                    lines.append("")
                    
            in_table = is_table_line
            
            # Bảo toàn tối đa 1 dòng trống
            if not line and (not lines or not lines[-1]):
                continue
                
            # Gộp các dòng bị đứt khúc trong bảng (ví dụ dấu `|` bị rơi rớt xuống dòng)
            if in_table and line == "|" and lines and lines[-1].startswith("|"):
                lines[-1] = lines[-1] + " |"
                continue
                
            # Ép xuống dòng cứng (hard line break) cho các dòng text thường để tránh gộp đoạn
            if line and not in_table and not line.startswith("#") and not line.startswith("```"):
                line = line + "  "
                
            lines.append(line)
        
        # Không tính các dòng rỗng, dòng bảng, hoặc dòng tiêu đề vào short_like để tránh nhận diện sai tài liệu ngắn
        struct_lines = [l for l in lines if l.strip() and not l.startswith("|") and not l.startswith("#")]
        bullet_like = sum(1 for line in struct_lines if line.startswith("-") or line.startswith("*") or line.startswith("•"))
        
        # Chỉ tính dòng ngắn thực sự không có cấu trúc
        short_like = sum(1 for line in struct_lines if len(line.strip()) <= 15)
        struct_line_count = len(struct_lines) or 1

        cleaned_lines: list[str] = []
        prev_line = ""
        for line in lines:
            if line == prev_line:
                continue
            prev_line = line
            if line.startswith("```"):
                continue
            cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines).strip()

        # P0 Fix: Dedup Repetition — VLM đôi khi lặp toàn bộ nội dung nhiều lần
        # (ví dụ: sao kê ngân hàng 8 cột → lặp 5 lần, từ 2.4KB → 12KB)
        deduped_text, dedup_count = self._dedup_repeated_blocks(cleaned_text)
        if dedup_count > 0:
            warnings.append(f"VLM repetition loop: đã xóa {dedup_count} khối nội dung bị lặp.")
            cleaned_text = deduped_text

        # Kiểm tra xem có bảng markdown hợp lệ không
        has_table = "|" in cleaned_text and "---" in cleaned_text

        # Chỉ rebuild từ OCR nếu không có bảng và tỷ lệ spam dòng ngắn/bullet cực kỳ cao (ví dụ: ảnh cực kỳ mờ/nhiễu)
        if not has_table and struct_line_count >= 5:
            bullet_ratio = bullet_like / struct_line_count
            short_ratio = short_like / struct_line_count
            if (bullet_ratio >= 0.70 or short_ratio >= 0.85) and ocr_blocks:
                warnings.append("Markdown quá nhiễu; đã dựng lại phiên bản gọn hơn từ OCR blocks.")
                rebuilt = self._rebuild_markdown_from_ocr(ocr_blocks)
                if rebuilt.strip():
                    cleaned_text = rebuilt.strip()

        return cleaned_text, warnings

    def _dedup_repeated_blocks(self, text: str, min_block_chars: int = 200) -> tuple[str, int]:
        """Phát hiện và loại bỏ các khối nội dung bị VLM lặp lại (repetition loop).

        Thuật toán:
          1. Chia text thành các khối bằng heading (# ...) hoặc dòng trống kép.
          2. Nếu một khối ≥ min_block_chars ký tự xuất hiện ≥ 2 lần → giữ bản đầu, xóa bản sau.

        Returns:
            (text_đã_dedup, số_khối_bị_xóa)
        """
        # Chia thành các section bởi heading markdown
        sections = re.split(r'(?=^# )', text, flags=re.MULTILINE)
        if len(sections) <= 1:
            # Không có heading → thử chia bằng dòng trống kép
            sections = re.split(r'\n\n\n+', text)

        if len(sections) <= 1:
            return text, 0

        seen_fingerprints: dict[str, int] = {}
        unique_sections: list[str] = []
        removed_count = 0

        for section in sections:
            stripped = section.strip()
            if not stripped:
                continue

            # Fingerprint: chuẩn hóa whitespace để so sánh
            fingerprint = re.sub(r'\s+', ' ', stripped).strip()

            if len(fingerprint) < min_block_chars:
                # Khối quá ngắn → giữ nguyên (không dedup khối nhỏ)
                unique_sections.append(section)
                continue

            if fingerprint in seen_fingerprints:
                # Đã thấy khối này (khớp hoàn toàn) → xóa bản lặp
                removed_count += 1
                logger.warning(
                    f"[Dedup] Xóa khối lặp ({len(fingerprint)} chars), "
                    f"bản gốc ở section #{seen_fingerprints[fingerprint]}"
                )
                continue

            # P0 Fix: Kiểm tra lặp bị cắt đứt (cutoff) do EOF
            # Cửa sổ trượt: so sánh khối hiện tại với phần đầu của các khối đã thấy
            is_cutoff_duplicate = False
            for seen_fp, sec_idx in seen_fingerprints.items():
                # Exact prefix match (nhanh)
                if seen_fp.startswith(fingerprint):
                    is_cutoff_duplicate = True
                    removed_count += 1
                    logger.warning(
                        f"[Dedup] Xóa khối lặp bị cắt cụt ({len(fingerprint)} chars), "
                        f"bản gốc ở section #{sec_idx}"
                    )
                    break
                # Fuzzy match: khối hiện tại tương đồng >80% với phần đầu của khối đã thấy
                # (bắt trường hợp VLM sinh lại nội dung hơi khác chút ít)
                if len(fingerprint) < len(seen_fp):
                    window = seen_fp[:len(fingerprint)]
                    try:
                        from rapidfuzz import fuzz
                        similarity = fuzz.ratio(fingerprint, window)
                    except ImportError:
                        similarity = 0
                    if similarity > 80:
                        is_cutoff_duplicate = True
                        removed_count += 1
                        logger.warning(
                            f"[Dedup] Xóa khối lặp fuzzy ({similarity:.0f}% match, "
                            f"{len(fingerprint)} chars), bản gốc ở section #{sec_idx}"
                        )
                        break
            
            if is_cutoff_duplicate:
                continue

            seen_fingerprints[fingerprint] = len(unique_sections) + 1
            unique_sections.append(section)

        if removed_count == 0:
            return text, 0

        return "\n\n".join(unique_sections).strip(), removed_count

    def _rebuild_markdown_from_ocr(self, ocr_blocks: list[OCRBlock]) -> str:
        """Dựng lại markdown tuyến tính từ OCR blocks đã sắp theo vị trí."""
        if not ocr_blocks:
            return ""

        def block_metrics(block: OCRBlock) -> tuple[float, float, float, float]:
            xs = [point[0] for point in block.bbox]
            ys = [point[1] for point in block.bbox]
            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)
            return x_min, x_max, y_min, y_max

        items = []
        for block in ocr_blocks:
            text = block.text.strip()
            if not text:
                continue
            x_min, x_max, y_min, y_max = block_metrics(block)
            center_y = (y_min + y_max) / 2.0
            center_x = (x_min + x_max) / 2.0
            items.append((center_y, center_x, text, y_min, y_max))

        if not items:
            return ""

        items.sort(key=lambda row: (row[0], row[1]))
        grouped_lines: list[list[tuple[float, float, str, float, float]]] = []
        current_line: list[tuple[float, float, str, float, float]] = []
        current_y: float | None = None
        y_threshold = max(12.0, sum(item[4] - item[3] for item in items) / len(items) * 0.75)

        for item in items:
            center_y = item[0]
            if current_y is None or abs(center_y - current_y) <= y_threshold:
                current_line.append(item)
                current_y = center_y if current_y is None else (current_y + center_y) / 2.0
            else:
                grouped_lines.append(sorted(current_line, key=lambda row: row[1]))
                current_line = [item]
                current_y = center_y
        if current_line:
            grouped_lines.append(sorted(current_line, key=lambda row: row[1]))

        rendered_lines: list[str] = []
        for group in grouped_lines:
            texts = [text for _, _, text, _, _ in group]
            if not texts:
                continue
            joined = " ".join(texts)
            rendered_lines.append(joined.strip())

        return "\n".join(rendered_lines).strip()

    def _classify_quality(self, result: ExtractionResult) -> tuple[str, float, list[str], str]:
        """Phân loại độ khó của tài liệu để route xử lý và review."""
        flags: list[str] = []

        ocr_conf = 0.0
        if result.raw_ocr:
            ocr_conf = sum(block.confidence for block in result.raw_ocr) / len(result.raw_ocr)
        text_len = len(result.markdown.strip())
        table_count = len(result.tables)
        token_count = len(result.uncertain_tokens)
        has_table = any(table.columns for table in result.tables) or ("|" in result.markdown and "---" in result.markdown)
        has_text = text_len > 50

        if ocr_conf < 0.45:
            flags.append("low_ocr_confidence")
        if text_len < 30:
            flags.append("very_short_output")
        if token_count > 10:
            flags.append("many_uncertain_tokens")
        if result.document_type == "unknown":
            flags.append("unknown_document_type")
        if result.metadata.get("sanitize_warnings"):
            flags.append("sanitized_vlm_output")

        routed_mode = str(result.metadata.get("layout_mode") or result.quality_class or "unknown")
        router_score = float(result.quality_score or 0.0)
        if ocr_conf < 0.35 or text_len < 30:
            quality_class = "critical_fail"
        elif ocr_conf < 0.65 or token_count > 0:
            quality_class = "noisy_scan"
        elif routed_mode in {"clean_text", "table_rich", "mixed_layout", "noisy_scan", "critical_fail"}:
            quality_class = routed_mode
        elif has_table and has_text and table_count > 0:
            quality_class = "mixed_layout"
        elif has_table and not has_text:
            quality_class = "table_rich"
        elif has_text and not has_table:
            quality_class = "clean_text"
        else:
            quality_class = "mixed_layout"

        quality_score = max(
            router_score,
            max(
                0.0,
                min(
                    1.0,
                    (ocr_conf * 0.35)
                    + (0.25 if has_text else 0.0)
                    + (0.2 if has_table else 0.0)
                    - (0.05 * min(token_count, 5))
                    - (0.1 if result.document_type == "unknown" else 0.0),
                ),
            ),
        )

        if quality_class == "critical_fail":
            action = "Quét lại hoặc review thủ công; nguồn ảnh quá kém để tin output."
        elif result.recommended_action:
            action = result.recommended_action
        elif quality_class == "table_rich":
            action = "Ưu tiên bbox table reconstruction rồi VLM normalize, giữ nguyên chữ gốc."
        elif quality_class == "mixed_layout":
            action = "Tách text/table, giữ OCR gốc, VLM chỉ ráp cấu trúc."
        elif quality_class == "noisy_scan":
            action = "Preprocess nhẹ, OCR giữ nguyên dữ liệu gốc, review nếu còn thiếu."
        else:
            action = "OCR + VLM nhẹ, không dịch, không chuẩn hóa nội dung."

        return quality_class, round(quality_score, 3), flags, action

    def _calculate_confidence(self, result: ExtractionResult) -> ConfidenceReport:
        """Tính confidence nhẹ dựa trên OCR và layout/table."""
        ocr_conf = 0.0
        if result.raw_ocr:
            ocr_conf = sum(block.confidence for block in result.raw_ocr) / len(result.raw_ocr)

        has_table = "|" in result.markdown and "---" in result.markdown
        table_conf = 0.9 if has_table and result.tables else (0.65 if has_table or result.tables else 0.25)
        layout_conf = 0.8 if len(result.markdown.strip()) > 100 else 0.3

        if result.metadata.get("sanitize_warnings"):
            layout_conf = min(layout_conf, 0.55)

        overall = (
            ocr_conf * 0.35
            + layout_conf * 0.35
            + table_conf * 0.30
        )
        return ConfidenceReport(
            ocr_confidence=round(ocr_conf, 3),
            layout_confidence=round(layout_conf, 3),
            table_confidence=round(table_conf, 3),
            overall=round(overall, 3),
        )

    def _requires_human_review(self, result: ExtractionResult) -> bool:
        """Flag review nếu output còn dấu hiệu không đáng tin."""
        threshold = settings.get("ingestion.extraction.confidence_threshold", 0.75)
        if result.quality_class == "critical_fail":
            return True
        if result.quality_score < threshold:
            return True
        if result.confidence.overall < threshold:
            return True
        if result.document_type == "unknown":
            return True
        if result.metadata.get("sanitize_warnings"):
            return True
            return True
        if not result.markdown.strip():
            return True
        return False
