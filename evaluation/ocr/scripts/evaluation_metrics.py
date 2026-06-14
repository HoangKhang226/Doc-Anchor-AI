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
    text = re.sub(r'\.{2,}', ' ', text)
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

def extract_markdown_tables(md_text: str) -> list[str]:
    """Trích xuất tất cả các ô (cells) từ Markdown Table (bao gồm cả open tables)."""
    cells = []
    lines = md_text.split('\n')
    for line in lines:
        line = line.strip()
        if '|' in line:
            # Bỏ qua dòng phân cách bảng (ví dụ: ---|---|--- hoặc |---|---|)
            if re.match(r'^[\s\-\|]+$', line) and '-' in line:
                continue
            
            raw_cells = line.split('|')
            if len(raw_cells) > 1:
                row_cells = [c.strip() for c in raw_cells]
                # Nếu bảng chuẩn có viền (bắt đầu/kết thúc bằng |), các ô ngoài cùng sẽ trống -> loại bỏ
                if line.startswith('|') and not row_cells[0]:
                    row_cells.pop(0)
                if line.endswith('|') and row_cells and not row_cells[-1]:
                    row_cells.pop(-1)
                
                cells.extend(row_cells)
    return [normalize_text(c) for c in cells if c.strip()]

def eval_table_logic(gt: str, pred: str) -> dict:
    """Đánh giá chuyên sâu cho Bảng biểu (Tables)."""
    base_metrics = eval_standard_logic(gt, pred)
    
    # Tiền xử lý: Bóc tách riêng các ô trong bảng
    gt_cells = extract_markdown_tables(gt)
    pred_cells = extract_markdown_tables(pred)
    
    # Tính Table Cell F1 Score bằng Fuzzy Matching
    if not gt_cells and not pred_cells:
        cell_f1 = 1.0
    elif not gt_cells or not pred_cells:
        cell_f1 = 0.0
    else:
        matched_cells = 0
        available_gt_cells = gt_cells.copy()
        
        for pred_c in pred_cells:
            best_match_idx = -1
            best_score = 0
            
            for idx, gt_c in enumerate(available_gt_cells):
                score = fuzz.ratio(pred_c, gt_c)
                if score > best_score:
                    best_score = score
                    best_match_idx = idx
                    
            # Ngưỡng 80% để chấp nhận lỗi nhỏ từ OCR (vd: sai 1 dấu phẩy)
            if best_score >= 80 and best_match_idx != -1:
                matched_cells += 1
                available_gt_cells.pop(best_match_idx)
                
        precision = matched_cells / len(pred_cells) if pred_cells else 0.0
        recall = matched_cells / len(gt_cells) if gt_cells else 0.0
        
        if precision + recall == 0:
            cell_f1 = 0.0
        else:
            cell_f1 = 2 * (precision * recall) / (precision + recall)
            
    base_metrics["table_f1"] = cell_f1
    return base_metrics

def extract_key_values(md_text: str) -> dict[str, str]:
    """Trích xuất cặp Key-Value từ mọi định dạng (Bullet, Bold, Table, Flat)"""
    kv_pairs = {}
    
    # 1. Quét theo Regex cho các format có cấu trúc (Bullet, Bold)
    pattern_structured = r'(?:\*\*([^*:\n]+?)\*\*\s*:\s*([^**\n]*))|(?:\*\*([^*:\n]+?):\*\*\s*([^**\n]*))|(?:(?:^|\n)[ \t]*[-*][ \t]+([^:\n]+?)\s*:\s*([^\n]+))'
    for match in re.finditer(pattern_structured, md_text):
        groups = [g for g in match.groups() if g is not None]
        if len(groups) >= 2:
            key, val = groups[0], groups[1]
            kv_pairs[normalize_text(key)] = normalize_text(val)
        
    # 2. Xử lý các dòng phẳng (Flat Text), dải chấm, hoặc ô trong Table
    # Thay thế | và dải dấu chấm thành khoảng trắng lớn
    text_cleaned = re.sub(r'\|', '  ', md_text)
    text_cleaned = re.sub(r'\.{3,}', '  ', text_cleaned)
    
    # Tách thành các block dựa trên xuống dòng hoặc khoảng trắng kép
    segments = re.split(r'\n|\s{2,}', text_cleaned)
    
    for segment in segments:
        segment = segment.strip()
        if not segment or ':' not in segment:
            continue
            
        parts = segment.split(':', 1)
        if len(parts) == 2:
            key = normalize_text(parts[0])
            val = normalize_text(parts[1])
            
            # Bỏ qua các key nhiễu (quá dài hoặc trống)
            if key and len(key.split()) < 15 and key not in kv_pairs:
                kv_pairs[key] = val
                
    # 3. Hỗ trợ bóc tách từ Markdown Table (VLM thường xuyên trả về dạng Bảng 2 cột cho Form)
    lines = md_text.split('\n')
    for line in lines:
        line = line.strip()
        if '|' in line and not re.match(r'^[\s\-\|]+$', line):
            raw_cells = line.split('|')
            if len(raw_cells) > 2:
                row_cells = [c.strip() for c in raw_cells]
                if line.startswith('|') and not row_cells[0]:
                    row_cells.pop(0)
                if line.endswith('|') and row_cells and not row_cells[-1]:
                    row_cells.pop(-1)
                
                # Nếu dòng bảng có đúng 2 cột, đây khả năng cao là Key-Value (vd: | Họ Tên | Nguyễn Văn A |)
                if len(row_cells) == 2:
                    key = normalize_text(row_cells[0]).replace(':', '').strip()
                    val = normalize_text(row_cells[1])
                    if key and len(key.split()) < 15 and key not in kv_pairs:
                        # Bỏ qua các dòng tiêu đề bảng phổ biến
                        if key.lower() not in ["key", "value", "chỉ tiêu", "nội dung", "thông tin", "thuộc tính", "giá trị"]:
                            kv_pairs[key] = val
                            
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
        # Tính điểm dựa trên Key khớp và Value khớp (Fuzzy matching cho cả Key và Value)
        matched_keys = 0
        matched_full = 0
        
        # Tạo bản copy để không match 1 key nhiều lần
        available_gt_kvs = gt_kvs.copy()
        
        for pred_k, pred_v in pred_kvs.items():
            best_match_k = None
            best_score = 0
            
            # Tìm key trong GT giống nhất (Fuzzy Match > 85% để chịu lỗi gõ sai của người dán nhãn)
            for gt_k in available_gt_kvs.keys():
                score = fuzz.ratio(pred_k, gt_k)
                if score > best_score:
                    best_score = score
                    best_match_k = gt_k
                    
            if best_score > 85 and best_match_k is not None:
                matched_keys += 1
                # Nếu value khớp > 80% thì coi như lấy đúng
                if fuzz.ratio(pred_v, available_gt_kvs[best_match_k]) > 80:
                    matched_full += 1
                # Xóa key đã match để tránh duplicate
                del available_gt_kvs[best_match_k]
                
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
