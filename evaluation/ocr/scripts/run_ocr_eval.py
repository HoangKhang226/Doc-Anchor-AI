import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz
from src.ingestion.pipeline import IngestionPipeline
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from evaluation_metrics import calculate_metrics_by_category

def main():
    parser = argparse.ArgumentParser(description="Colab OCR Evaluation Suite")
    parser.add_argument("--workers", type=int, default=1, help="Số lượng luồng chạy song song (batch processing). Khuyến nghị: 4-8 trên Colab.")
    args = parser.parse_args()

    data_dir = Path("evaluation/ocr/data")
    report_file = Path("evaluation/ocr/eval_report.json")
    
    # Auto-resume logic: Đọc kết quả cũ nếu có
    results = {}
    if report_file.exists():
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"🔄 Auto-Resume: Tìm thấy {len(results)} kết quả cũ. Sẽ BỎ QUA các file này.")
        except json.JSONDecodeError:
            print("⚠️ File eval_report.json bị lỗi, sẽ ghi đè từ đầu.")
            
    pipeline = IngestionPipeline()
    
    print("="*80)
    print("🚀 DOC ANCHOR AI: COLAB OCR EVALUATION SUITE")
    print("="*80)
    
    print("🔥 Đang khởi tạo (Warm-up) PaddleOCR để tải model (chống lỗi Multi-thread)...")
    try:
        from src.ingestion.extractors.paddle_ocr_extractor import PaddleOCRExtractor
        from src.ingestion.table_region_detection import TableRegionDetector
        PaddleOCRExtractor()._load_engine()
        TableRegionDetector()._get_engine()
        print("✅ Khởi tạo PaddleOCR thành công.")
    except Exception as e:
        print(f"⚠️ Cảnh báo khởi tạo PaddleOCR: {e}")
        
    # Tìm tất cả ảnh trong các thư mục con (cord_v2, sroie, custom_finsight)
    all_images = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
        all_images.extend(list(data_dir.rglob(ext)))
        
    all_images = sorted(all_images)
    
    if not all_images:
        print("❌ Không tìm thấy ảnh nào trong evaluation/ocr/data/")
        return
        
    print(f"Tổng số ảnh trong thư mục: {len(all_images)}")
    
    # Bọc việc ghi file bằng Lock để tránh race condition khi chạy multi-thread
    write_lock = threading.Lock()
    
    def process_image(img_path: Path):
        dataset_name = img_path.parent.name
        img_id = f"{dataset_name}/{img_path.name}"
        
        if img_id in results:
            return None # Bỏ qua nếu đã chấm điểm
            
        # Tìm file Ground Truth với 2 trường hợp phổ biến:
        # ƯU TIÊN 1: Giữ nguyên đuôi ảnh (vd: ảnh.jpg -> ảnh.jpg.gt.txt) — chính xác nhất
        # ƯU TIÊN 2: Bỏ đuôi ảnh (vd: ảnh.jpg -> ảnh.gt.txt) — dễ bị xung đột nếu có 2 ảnh cùng tên
        gt_path_exact = img_path.with_name(img_path.name + '.gt.txt')
        gt_path_stem = img_path.with_suffix('.gt.txt')
        
        gt_path = gt_path_exact if gt_path_exact.exists() else gt_path_stem
        if not gt_path.exists():
            return f"⚠️  Bỏ qua {img_id} (Không có Ground Truth .gt.txt)"
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = f.read().strip()
            
        t0 = time.time()
        try:
            extraction = pipeline.run(img_path)
            latency = time.time() - t0
            
            prediction = (extraction.markdown or "").strip()
            
            # Ghi output ra file để sau này dễ dàng debug/kiểm tra nếu điểm thấp
            pred_path = img_path.with_name(img_path.name + ".pred.md")
            with open(pred_path, 'w', encoding='utf-8') as f:
                f.write(prediction)

            # Gọi Metric Router từ evaluation_metrics.py
            metrics = calculate_metrics_by_category(dataset_name, ground_truth, prediction)
            
            with write_lock:
                results[img_id] = {
                    "dataset": dataset_name,
                    "cer": metrics["cer"],
                    "wer": metrics["wer"],
                    "sim": metrics["sim"],
                    "sim_token": metrics["sim_token"],
                    "table_f1": metrics.get("table_f1"),
                    "kie_f1": metrics.get("kie_f1"),
                    "latency": round(latency, 2),
                    "mode": extraction.metadata.get('layout_mode')
                }
                # Ghi file liên tục (Checkpointing)
                with open(report_file, "w", encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
            
            # Xây dựng chuỗi kết quả
            log_msg = f"✅ Xong {img_id} ({latency:.2f}s) | CER: {metrics['cer']:.3f} | WER: {metrics['wer']:.3f} | Sim: {metrics['sim']:.2%}"
            if "table_f1" in metrics:
                log_msg += f" | Table F1: {metrics['table_f1']:.2%}"
            if "kie_f1" in metrics:
                log_msg += f" | Form F1: {metrics['kie_f1']:.2%}"
                
            return log_msg
        except Exception as e:
            return f"❌ LỖI tại {img_id}: {e}"

    processed_this_session = 0
    if args.workers > 1:
        print(f"⚡ Bật chế độ chạy song song với {args.workers} workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_image, img_path) for img_path in all_images]
            for future in as_completed(futures):
                msg = future.result()
                if msg:
                    print(msg)
                    processed_this_session += 1
    else:
        print("Chạy tuần tự (1 worker)...")
        for img_path in all_images:
            msg = process_image(img_path)
            if msg:
                print(msg)
                processed_this_session += 1
            
    print("\n" + "="*80)
    print("📈 TỔNG KẾT ĐÁNH GIÁ (EVALUATION REPORT)")
    print("="*80)
    
    if results:
        # Nhóm kết quả theo dataset/thư mục
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in results.values():
            grouped[r["dataset"]].append(r)

        total_cer = sum(r["cer"] for r in results.values())
        total_wer = sum(r["wer"] for r in results.values())
        total_sim = sum(r["sim"] for r in results.values())
        n = len(results)
        
        total_sim_token = sum(r.get("sim_token", r["sim"]) for r in results.values())
        
        # --- GHI RA FILE TXT ĐỂ DỄ DÀNG COPY/PASTE VÀ ĐÁNH GIÁ ---
        txt_report_file = Path("evaluation/ocr/eval_report.txt")
        with open(txt_report_file, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("📊 BÁO CÁO BENCHMARK THEO LOẠI TÀI LIỆU (CATEGORIZED)\n")
            f.write("="*80 + "\n\n")
            
            for cat, items in sorted(grouped.items()):
                cat_n = len(items)
                cat_cer = sum(r["cer"] for r in items) / cat_n
                cat_wer = sum(r["wer"] for r in items) / cat_n
                cat_sim = sum(r["sim"] for r in items) / cat_n
                
                # Tính các chỉ số phụ nếu có
                special_metrics = ""
                if cat == "tables":
                    cat_table_f1 = sum(r.get("table_f1", 0.0) or 0.0 for r in items) / cat_n
                    special_metrics = f" | Table Cell F1: {cat_table_f1:.2%}"
                elif cat == "forms":
                    cat_kie_f1 = sum(r.get("kie_f1", 0.0) or 0.0 for r in items) / cat_n
                    special_metrics = f" | Form KIE F1: {cat_kie_f1:.2%}"
                else:
                    cat_sim_token = sum(r.get("sim_token", r["sim"]) for r in items) / cat_n
                    special_metrics = f" | TokenSim: {cat_sim_token:.2%}"
                
                f.write(f"📁 Nhóm: {cat.upper()} ({cat_n} file)\n")
                f.write(f"   -> CER: {cat_cer:.3f} | WER: {cat_wer:.3f} | Sim: {cat_sim:.2%}{special_metrics}\n\n")

            f.write("="*80 + "\n")
            f.write("🔥 TỔNG KẾT TOÀN HỆ THỐNG (OVERALL)\n")
            f.write("="*80 + "\n")
            f.write(f"Tổng số file đã hoàn thành : {n}\n")
            f.write(f"CER trung bình toàn tập    : {total_cer/n:.3f}\n")
            f.write(f"WER trung bình toàn tập    : {total_wer/n:.3f}\n")
            f.write(f"Sim trung bình (normalized): {total_sim/n:.2%}\n")
            f.write(f"TokenSim trung bình        : {total_sim_token/n:.2%}\n\n")

            f.write("="*80 + "\n")
            f.write("BÁO CÁO ĐÁNH GIÁ CHI TIẾT TỪNG ẢNH\n")
            f.write("="*80 + "\n")
            for img_id, r in sorted(results.items()):
                st = r.get('sim_token', r['sim'])
                f.write(f"[{img_id}] | CER: {r['cer']:.3f} | WER: {r['wer']:.3f} | Sim: {r['sim']:.2%} | TokenSim: {st:.2%}\n")

        # In thẳng ra màn hình Console cho người dùng xem
        with open(txt_report_file, "r", encoding="utf-8") as f:
            print(f.read())
            
        print(f"\n📁 Đã lưu file báo cáo chi tiết tại: {txt_report_file}")
        
        # --- TỰ ĐỘNG NÉN FILE ZIP TRÊN COLAB ---
        import shutil
        zip_path = Path("evaluation/ocr/benchmark_results")
        print("\n📦 Đang nén toàn bộ kết quả thành file ZIP...")
        # Tạo thư mục tạm để gom các file báo cáo
        temp_dir = Path("evaluation/ocr/temp_zip")
        temp_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(txt_report_file, temp_dir / "eval_report.txt")
        if report_file.exists():
            shutil.copy(report_file, temp_dir / "eval_report.json")
            
        # Gom thêm toàn bộ file markdown dự đoán (.pred.md)
        pred_dir = temp_dir / "predictions"
        pred_dir.mkdir(exist_ok=True)
        pred_count = 0
        for pred_file in data_dir.rglob("*.pred.md"):
            # Giữ nguyên cấu trúc thư mục (pure_text, forms, v.v.)
            dest_dir = pred_dir / pred_file.parent.name
            dest_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(pred_file, dest_dir / pred_file.name)
            pred_count += 1
        print(f"   Đã gom {pred_count} file markdown dự đoán vào ZIP.")
        
        # Nén thành benchmark_results.zip
        shutil.make_archive(str(zip_path), 'zip', str(temp_dir))
        
        # Xóa thư mục tạm
        shutil.rmtree(temp_dir)
        print(f"✅ Đã nén xong! File ZIP nằm tại: {zip_path}.zip (Hãy tải file này về máy)")
        
    else:
        print("Chưa có kết quả nào được ghi nhận.")

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
    main()
