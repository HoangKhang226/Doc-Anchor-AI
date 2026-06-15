import pytest

def test_pipeline_import():
    """Test xem IngestionPipeline có import và khởi tạo được không."""
    from src.ingestion.pipeline import IngestionPipeline
    
    # Chỉ test khởi tạo (kiểm tra liên kết file), không gọi .run() để tránh tốn thời gian
    pipeline = IngestionPipeline()
    assert pipeline is not None
    assert hasattr(pipeline, 'run')
    assert hasattr(pipeline, 'save_outputs')

def test_preprocessor_import():
    """Test thư mục core."""
    from src.ingestion.core.image_preprocessor import ImagePreprocessor
    preprocessor = ImagePreprocessor()
    assert preprocessor is not None

def test_paddle_extractor_import():
    """Test thư mục extractors - PaddleOCR."""
    from src.ingestion.extractors.paddle_ocr_extractor import PaddleOCRExtractor
    # Khởi tạo dạng lazy-load, không tốn tài nguyên
    extractor = PaddleOCRExtractor()
    assert extractor is not None

def test_vlm_extractor_import():
    """Test thư mục extractors - VLM."""
    from src.ingestion.extractors.vlm_ocr_extractor import VLMOCRProcessor
    vlm = VLMOCRProcessor()
    assert vlm is not None

def test_table_region_detector_import():
    """Test thư mục tables."""
    from src.ingestion.tables.table_region_detection import TableRegionDetector
    detector = TableRegionDetector()
    assert detector is not None

def test_schemas_data_models():
    """Test các Data Models cơ bản có hoạt động đúng chuẩn Pydantic/Dataclass không."""
    from src.ingestion.schemas.extraction_result import OCRBlock
    
    block = OCRBlock(text="Doc Anchor AI", confidence=0.99, bbox=[[0,0], [10,0], [10,10], [0,10]])
    assert block.text == "Doc Anchor AI"
    assert block.confidence == 0.99
    assert len(block.bbox) == 4
