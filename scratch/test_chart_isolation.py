"""Test script for Chart Isolation."""
import sys
import json
from pathlib import Path

# Thêm thư mục gốc của dự án vào PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.image_preprocessor import ImagePreprocessor
from src.ingestion.extractors import PaddleOCRExtractor
from src.ingestion.table_region_detection import TableRegionDetector
from src.ingestion.chart_region_detector import ChartRegionDetector

def test_chart_isolation():
    image_path = Path("d:/Project/Doc Anchor AI/ed7dcef3-be10-4ac4-b1e2-0814d3b0f125.jpg")
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        return
        
    print(f"Testing Chart Isolation on: {image_path}")
    
    # 1. Clean image
    scratch_dir = Path("scratch")
    scratch_dir.mkdir(exist_ok=True)
    cleaned_path = scratch_dir / f"cleaned_{image_path.name}"
    
    preprocessor = ImagePreprocessor()
    preprocessor.process(image_path, cleaned_path)
    print(f"1. Image cleaned -> {cleaned_path}")
    
    # 2. Extract OCR
    print("2. Running PaddleOCR...")
    try:
        ocr_extractor = PaddleOCRExtractor()
        ocr_blocks = ocr_extractor.extract(cleaned_path)
        print(f"   -> Extracted {len(ocr_blocks)} OCR blocks.")
    except Exception as e:
        print(f"   -> Lỗi PaddleOCR: {e}")
        return
    
    # 3. Detect Table Regions
    print("3. Detecting table regions...")
    table_detector = TableRegionDetector()
    table_regions = table_detector.detect(cleaned_path) if ocr_blocks else []
    print(f"   -> Detected {len(table_regions)} table regions: {[r.bbox for r in table_regions]}")
    
    # 4. Detect Chart Regions
    print("4. Detecting chart regions...")
    chart_detector = ChartRegionDetector()
    chart_regions = chart_detector.detect(
        cleaned_path,
        ocr_blocks,
        table_bboxes=[r.bbox for r in table_regions]
    )
    print(f"   -> Detected {len(chart_regions)} chart regions: {[r.bbox for r in chart_regions]}")
    
    # 5. Crop and Save
    assets_dir = scratch_dir / "assets"
    print(f"5. Cropping and saving to {assets_dir}...")
    assets = chart_detector.crop_and_save(
        cleaned_path,
        chart_regions,
        assets_dir,
        image_path.stem
    )
    
    print("\n--- Final Chart Assets Metadata ---")
    print(json.dumps(assets, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    test_chart_isolation()
