from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
import re
def normalize_text(text: str) -> str:
    """Chuẩn hóa text: Loại bỏ tất cả noise từ Markdown formatting."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'^\|[\s:|-]+\|\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = text.replace("|", " ").replace("*", " ")
    text = re.sub(r'-{3,}', ' ', text)
    text = re.sub(r'(?<!\d)-(?!\d)', ' ', text)
    text = re.sub(r'\[\s*[xX]?\s*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_cer(reference: str, hypothesis: str) -> float:
    # Tránh chia 0
    if not reference:
        return 1.0 if hypothesis else 0.0
    return min(1.0, Levenshtein.distance(reference, hypothesis) / len(reference))

def calculate_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    return min(1.0, Levenshtein.distance(ref_words, hyp_words) / len(ref_words))

def eval_standard_logic(gt: str, pred: str) -> dict:
    """Đánh giá cơ bản cho Pure Text và Mixed Layouts."""
    norm_gt = normalize_text(gt)
    norm_pred = normalize_text(pred)
    
    return {
        "cer": calculate_cer(norm_gt, norm_pred),
        "wer": calculate_wer(norm_gt, norm_pred),
        "sim": fuzz.ratio(norm_gt, norm_pred) / 100.0,
        "sim_token": fuzz.token_sort_ratio(norm_gt, norm_pred) / 100.0
    }

def extract_markdown_tables(md_text: str) -> list[list[str]]:
    """Trích xuất tất cả các ô (cells) từ Markdown Table."""
    cells = []
    lines = md_text.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('|') and line.endswith('|'):
            # Bỏ qua dòng phân cách bảng (---|---|---)
            if re.match(r'^\|[\s\-\|]+\|$', line):
                continue
            # Tách các ô, bỏ ô rỗng ở đầu và cuối do ký tự |
            row_cells = [c.strip() for c in line.split('|')[1:-1]]
            cells.extend(row_cells)
    return [normalize_text(c) for c in cells if c.strip()]

def eval_table_logic(gt: str, pred: str) -> dict:
    """Đánh giá chuyên sâu cho Bảng biểu (Tables)."""
    base_metrics = eval_standard_logic(gt, pred)
    
    # Tiền xử lý: Bóc tách riêng các ô trong bảng
    gt_cells = extract_markdown_tables(gt)
    pred_cells = extract_markdown_tables(pred)
    
    # Tính Table Cell F1 Score (Độ chính xác của việc bóc từng ô)
    if not gt_cells and not pred_cells:
        cell_f1 = 1.0
    elif not gt_cells or not pred_cells:
        cell_f1 = 0.0
    else:
        # Đếm số ô trùng khớp
        gt_set = set(gt_cells)
        pred_set = set(pred_cells)
        
        true_positives = len(gt_set.intersection(pred_set))
        precision = true_positives / len(pred_set) if pred_set else 0.0
        recall = true_positives / len(gt_set) if gt_set else 0.0
        
        if precision + recall == 0:
            cell_f1 = 0.0
        else:
            cell_f1 = 2 * (precision * recall) / (precision + recall)
            
    base_metrics["table_f1"] = cell_f1
    return base_metrics

def extract_key_values(md_text: str) -> dict[str, str]:
    """Trích xuất cặp Key-Value từ định dạng **Key**: Value hoặc - Key: Value"""
    kv_pairs = {}
    # Match: **Key**: Value OR - Key: Value
    pattern = r'(?:\*\*([^*]+)\*\*\s*:\s*(.+))|(?:-\s*([^:\n]+)\s*:\s*(.+))'
    for match in re.finditer(pattern, md_text):
        if match.group(1): # Pattern 1
            key, val = match.group(1), match.group(2)
        else: # Pattern 2
            key, val = match.group(3), match.group(4)
        kv_pairs[normalize_text(key)] = normalize_text(val)
    return kv_pairs

def eval_form_logic(gt: str, pred: str) -> dict:
    """Đánh giá chuyên sâu cho Biểu mẫu (Forms / KIE)."""
    base_metrics = eval_standard_logic(gt, pred)
    
    gt_kvs = extract_key_values(gt)
    pred_kvs = extract_key_values(pred)
    
    if not gt_kvs and not pred_kvs:
        kie_f1 = 1.0
    elif not gt_kvs or not pred_kvs:
        kie_f1 = 0.0
    else:
        # Tính điểm dựa trên Key khớp và Value khớp
        matched_keys = 0
        matched_full = 0
        for k, v in pred_kvs.items():
            if k in gt_kvs:
                matched_keys += 1
                # Nếu value khớp > 80% thì coi như lấy đúng
                if fuzz.ratio(v, gt_kvs[k]) > 80:
                    matched_full += 1
                    
        precision = matched_full / len(pred_kvs) if pred_kvs else 0.0
        recall = matched_full / len(gt_kvs) if gt_kvs else 0.0
        
        if precision + recall == 0:
            kie_f1 = 0.0
        else:
            kie_f1 = 2 * (precision * recall) / (precision + recall)
            
    base_metrics["kie_f1"] = kie_f1
    return base_metrics

def calculate_metrics_by_category(dataset: str, gt: str, pred: str) -> dict:
    """Router định tuyến logic chấm điểm theo thể loại dataset."""
    if dataset == "tables":
        return eval_table_logic(gt, pred)
    elif dataset == "forms":
        return eval_form_logic(gt, pred)
    else:
        # pure_text, mixed_layouts, v.v.
        return eval_standard_logic(gt, pred)
