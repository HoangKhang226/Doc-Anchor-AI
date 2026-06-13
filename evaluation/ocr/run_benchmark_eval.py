import os
import glob
import re
from collections import defaultdict

try:
    import jiwer
except ImportError:
    print("Vui lòng cài đặt jiwer: pip install jiwer")
    exit(1)

def get_tokens(text):
    # Tách từ, bỏ dấu câu để tính F1
    return set(re.findall(r'\w+', text.lower()))

def calculate_f1(ref_text, pred_text):
    ref_tokens = get_tokens(ref_text)
    pred_tokens = get_tokens(pred_text)
    
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
        
    common_tokens = ref_tokens.intersection(pred_tokens)
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_dataset(gt_dir, pred_dir):
    """
    gt_dir: Thư mục chứa ground truth, có các sub-folders (forms, mixed_layouts,...)
    pred_dir: Thư mục chứa kết quả dự đoán từ model (cấu trúc tương tự)
    """
    categories = ['forms', 'mixed_layouts', 'pure_text', 'tables']
    
    # Dictionary lưu trữ điểm số theo category
    results = defaultdict(lambda: {'cer': [], 'wer': [], 'f1': [], 'count': 0})
    
    for category in categories:
        gt_category_path = os.path.join(gt_dir, category)
        pred_category_path = os.path.join(pred_dir, category)
        
        if not os.path.exists(gt_category_path):
            continue
            
        gt_files = glob.glob(os.path.join(gt_category_path, "*.gt.txt"))
        
        for gt_file in gt_files:
            filename = os.path.basename(gt_file)
            # Giả sử file dự đoán có tên tương tự, thay đuôi .gt.txt bằng .pred.txt
            pred_filename = filename.replace('.gt.txt', '.pred.txt')
            pred_file = os.path.join(pred_category_path, pred_filename)
            
            with open(gt_file, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
                
            # Nếu chưa có file pred (model chưa chạy xong), bỏ qua hoặc tính là rỗng
            if os.path.exists(pred_file):
                with open(pred_file, 'r', encoding='utf-8') as f:
                    pred_text = f.read().strip()
            else:
                pred_text = ""
                
            # Xử lý text rỗng để tránh lỗi chia cho 0 trong jiwer
            ref_for_jiwer = ref_text if ref_text else "EMPTY"
            pred_for_jiwer = pred_text if pred_text else "EMPTY_PRED"
            
            # Tính toán Metrics
            try:
                cer = jiwer.cer(ref_for_jiwer, pred_for_jiwer)
                wer = jiwer.wer(ref_for_jiwer, pred_for_jiwer)
            except ValueError:
                cer = 1.0
                wer = 1.0
                
            f1 = calculate_f1(ref_text, pred_text)
            
            # Giới hạn lỗi tối đa là 1.0 (100%) để tránh nhiễu
            cer = min(cer, 1.0)
            wer = min(wer, 1.0)
            
            results[category]['cer'].append(cer)
            results[category]['wer'].append(wer)
            results[category]['f1'].append(f1)
            results[category]['count'] += 1

    # In Báo cáo tổng hợp
    print("="*60)
    print(f"{'BÁO CÁO KẾT QUẢ ĐÁNH GIÁ BENCHMARK OCR':^60}")
    print("="*60)
    print(f"{'Category':<18} | {'Count':<6} | {'CER (%)':<8} | {'WER (%)':<8} | {'F1-Score':<8}")
    print("-" * 60)
    
    total_cer, total_wer, total_f1, total_count = 0, 0, 0, 0
    
    for category in categories:
        res = results[category]
        if res['count'] > 0:
            avg_cer = sum(res['cer']) / res['count']
            avg_wer = sum(res['wer']) / res['count']
            avg_f1 = sum(res['f1']) / res['count']
            
            total_cer += sum(res['cer'])
            total_wer += sum(res['wer'])
            total_f1 += sum(res['f1'])
            total_count += res['count']
            
            print(f"{category:<18} | {res['count']:<6} | {avg_cer*100:>5.2f}% | {avg_wer*100:>5.2f}% | {avg_f1*100:>5.2f}%")
            
    if total_count > 0:
        print("=" * 60)
        print(f"{'OVERALL (TRUNG BÌNH)':<18} | {total_count:<6} | {(total_cer/total_count)*100:>5.2f}% | {(total_wer/total_count)*100:>5.2f}% | {(total_f1/total_count)*100:>5.2f}%")
        print("=" * 60)

if __name__ == "__main__":
    # Thay đổi đường dẫn cho phù hợp
    GT_DIR = r"d:\Project\Doc Anchor AI\evaluation\ocr\data\categorized"
    PRED_DIR = r"d:\Project\Doc Anchor AI\evaluation\ocr\data\predictions\model_a" # Thay đổi tên model
    
    evaluate_dataset(GT_DIR, PRED_DIR)
