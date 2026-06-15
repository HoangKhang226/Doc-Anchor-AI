import zipfile
from pathlib import Path

def main():
    pred_files = list(Path("evaluation/ocr/data").rglob("*.pred.md"))
    if pred_files:
        zip_path = Path("evaluation/ocr/doc_anchor_ai_ocr_predictions.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for file in pred_files:
                # Giữ nguyên cấu trúc thư mục con tương ứng
                arcname = file.relative_to(Path("evaluation/ocr/data"))
                zipf.write(file, arcname=arcname)
        print(f"Đã nén {len(pred_files)} file .pred.md thành công vào: {zip_path}")
    else:
        print("Không tìm thấy file .pred.md nào để nén!")

if __name__ == "__main__":
    main()
