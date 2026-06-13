import os
import sys
from pathlib import Path

# Thêm đường dẫn project
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline

def test_pipeline():
    print("Khởi tạo Pipeline...")
    pipeline = IngestionPipeline()
    
    # Dùng ảnh test có sẵn của anh
    test_image = Path("d:/Project/Doc Anchor AI/ed7dcef3-be10-4ac4-b1e2-0814d3b0f125.jpg")
    
    if not test_image.exists():
        print(f"Không tìm thấy ảnh: {test_image}")
        return
        
    print(f"\nBắt đầu xử lý ảnh: {test_image}")
    try:
        result = pipeline.run(test_image)
        
        print("\n=== KẾT QUẢ ===")
        print(f"Quality Class: {result.quality_class}")
        print(f"Overall Confidence: {result.confidence.overall:.2f}")
        print(f"Tables: {len(result.tables)}")
        
        # In 500 ký tự đầu của Markdown
        preview = result.markdown[:500] + ("..." if len(result.markdown) > 500 else "")
        print("\n--- Markdown Preview ---")
        print(preview)
        
        print("\nTest thành công! Code không có lỗi syntax/runtime.")
        
    except Exception as e:
        import traceback
        print(f"\n[LỖI] Pipeline crash:")
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
