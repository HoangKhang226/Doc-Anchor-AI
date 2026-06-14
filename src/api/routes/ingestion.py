import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.ingestion.pipeline import IngestionPipeline

router = APIRouter(tags=["Ingestion"])

# Instantiate the pipeline globally. 
# In a production environment with high concurrency, you might want a pool of workers
# or a message queue (like Celery) rather than blocking the HTTP request thread.
pipeline = IngestionPipeline()

class ExtractedTableResponse(BaseModel):
    name: str
    columns: List[str]
    rows: List[Dict[str, Any]]

class OCRBlockResponse(BaseModel):
    text: str
    confidence: float
    bbox: List[List[float]]
    language: str | None

class ExtractionResponse(BaseModel):
    source_file: str
    document_type: str
    languages: List[str]
    markdown: str
    fields: Dict[str, Any]
    tables: List[ExtractedTableResponse]
    chart_assets: List[Dict[str, Any]]
    raw_ocr: List[OCRBlockResponse]
    uncertain_tokens: List[str]
    quality_class: str
    quality_score: float
    issue_flags: List[str]
    recommended_action: str
    requires_human_review: bool
    layout_mode: str
    processing_time_ms: float
    metadata: Dict[str, Any]

@router.post("/extract", response_model=ExtractionResponse)
async def extract_document(file: UploadFile = File(...)):
    """
    Tải file ảnh/tài liệu lên và chạy qua toàn bộ Ingestion Pipeline.
    Trả về nội dung Markdown đã được chuẩn hóa và các thông tin Metadata liên quan.
    """
    if not file.content_type.startswith("image/") and file.content_type not in ["application/pdf"]:
         raise HTTPException(status_code=400, detail="Chỉ hỗ trợ file ảnh (jpeg, png, webp, v.v.) và PDF.")
    
    tmp_path = None
    try:
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            
        # Run pipeline
        start_time = time.time()
        result = pipeline.run(tmp_path)
        
        # Optionally save outputs locally for debugging
        pipeline.save_outputs(result)
        
        # Convert dataclasses to dicts for FastAPI response
        return {
            "source_file": file.filename,
            "document_type": result.document_type,
            "languages": result.languages,
            "markdown": result.markdown,
            "fields": result.fields,
            "tables": [{"name": t.name, "columns": t.columns, "rows": t.rows} for t in result.tables],
            "chart_assets": result.chart_assets,
            "raw_ocr": [
                {"text": b.text, "confidence": b.confidence, "bbox": b.bbox, "language": b.language} 
                for b in result.raw_ocr
            ],
            "uncertain_tokens": result.uncertain_tokens,
            "quality_class": result.quality_class,
            "quality_score": result.quality_score,
            "issue_flags": result.issue_flags,
            "recommended_action": result.recommended_action,
            "requires_human_review": result.requires_human_review,
            "layout_mode": result.metadata.get("layout_mode", "unknown"),
            "processing_time_ms": result.metadata.get("elapsed_seconds", 0) * 1000,
            "metadata": result.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
