import streamlit as st
import requests
import os
from pathlib import Path

API_URL = "http://127.0.0.1:8000/api/v1/extract"

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h1 style='color: #2D3748; font-weight: 800; letter-spacing: -1px; margin-bottom: 0;'>DOC ANCHOR</h1>
            <p style='color: #718096; font-size: 0.9rem; margin-top: 0;'>AI DOCUMENT INTELLIGENCE</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation
        view_mode = st.radio("Navigation", ["OCR Upload", "History"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("<h4 style='color: #4A5568; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;'>Active Pipelines</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div class="pipeline-item">
            <div class="status-dot"></div> Document Ingestion
        </div>
        <div class="pipeline-item">
            <div class="status-dot"></div> Table Extraction
        </div>
        <div class="pipeline-item inactive">
            <div class="status-dot inactive-dot"></div> Chart Analysis (Crop)
        </div>
        """, unsafe_allow_html=True)
        
        return view_mode

def render_history():
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'><span>📁</span> Processed History</div>", unsafe_allow_html=True)
    
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        st.info("Chưa có dữ liệu xử lý nào (No history found).")
        st.markdown("</div>", unsafe_allow_html=True)
        return
        
    md_files = list(processed_dir.glob("*.md"))
    if not md_files:
        st.info("Chưa có file Markdown nào được lưu.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
        
    # Sort files by modification time, newest first
    md_files.sort(key=os.path.getmtime, reverse=True)
    
    for md_file in md_files:
        stem = md_file.stem
        
        # Tìm file ảnh đã được tiền xử lý (cleaned_...)
        image_files = list(processed_dir.glob(f"cleaned_{stem}.*"))
        img_path = image_files[0] if image_files else None
        
        with st.expander(f"📄 {md_file.name}"):
            img_col, txt_col = st.columns([1, 1.2])
            
            with img_col:
                st.markdown("**Source Image**")
                img_tab, = st.tabs(["🖼️ Image View"])
                with img_tab:
                    with st.container(height=800):
                        if img_path:
                            st.image(str(img_path), use_container_width=True)
                        else:
                            st.warning("Original image not found.")
                    
            with txt_col:
                st.markdown("**Extracted Markdown**")
                with open(md_file, "r", encoding="utf-8") as f:
                    md_content = f.read()
                    
                tab1, tab2 = st.tabs(["Preview", "Edit Markdown"])
                
                with tab1:
                    with st.container(height=800):
                        st.markdown(md_content)
                    
                    st.write("")
                    
                    # Nút download ngay dưới markdown
                    st.download_button(
                        label="Download Markdown",
                        data=md_content,
                        file_name=md_file.name,
                        mime="text/markdown",
                        use_container_width=True,
                        key=f"dl_hist_{stem}"
                    )
                    
                with tab2:
                    edited_md = st.text_area("Chỉnh sửa Markdown", value=md_content, height=800, label_visibility="collapsed", key=f"edit_hist_{stem}")
                    if st.button("Lưu thay đổi (Save to file)", key=f"save_hist_{stem}", type="primary", use_container_width=True):
                        with open(md_file, "w", encoding="utf-8") as f:
                            f.write(edited_md)
                        st.success("Đã cập nhật file gốc thành công!")
                        st.rerun()
                    
    st.markdown("</div>", unsafe_allow_html=True)

def render_dashboard():
    # Premium Custom CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp { 
            background-color: #F7FAFC; 
            background-image: radial-gradient(#E2E8F0 1px, transparent 1px);
            background-size: 20px 20px;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }
        
        .pipeline-item {
            display: flex;
            align-items: center;
            font-size: 0.9rem;
            color: #4A5568;
            margin: 8px 0;
        }
        .pipeline-item.inactive {
            color: #A0AEC0;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #48BB78;
            border-radius: 50%;
            margin-right: 10px;
            box-shadow: 0 0 5px rgba(72, 187, 120, 0.5);
        }
        .inactive-dot {
            background-color: #CBD5E0;
            box-shadow: none;
        }
        
        /* Card Styling */
        .premium-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
            border: 1px solid rgba(226, 232, 240, 0.8);
            margin-bottom: 24px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .premium-card:hover {
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02);
        }
        
        /* Metric Styling */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            border: 1px solid #E2E8F0;
            border-top: 4px solid #3182CE;
        }
        div[data-testid="metric-container"] label {
            color: #718096 !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.8rem;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #2D3748 !important;
            font-weight: 800 !important;
            font-size: 2rem !important;
        }
        
        /* Custom Headers */
        .section-header {
            color: #2D3748;
            font-weight: 700;
            font-size: 1.25rem;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
        }
        .section-header span {
            margin-right: 10px;
        }
        
        /* Fix Streamlit default layout spacing */
        div.block-container {
            padding-top: 2rem;
            max-width: 1400px;
        }

        /* Khống chế cỡ chữ của Markdown bằng px tĩnh để đảm bảo nó nhỏ gọn */
        [data-testid="stMarkdownContainer"] p, 
        [data-testid="stMarkdownContainer"] li, 
        [data-testid="stMarkdownContainer"] td,
        [data-testid="stMarkdownContainer"] th {
            font-size: 11px !important;
            line-height: 1.4 !important;
        }
        [data-testid="stMarkdownContainer"] h1 { font-size: 16px !important; }
        [data-testid="stMarkdownContainer"] h2 { font-size: 14px !important; }
        [data-testid="stMarkdownContainer"] h3 { font-size: 12px !important; }
        
        /* Chỉnh lại font bảng để không bị chiếm chỗ */
        [data-testid="stMarkdownContainer"] table {
            font-size: 11px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    view_mode = render_sidebar()

    st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h1 style="color: #1A202C; font-weight: 800; font-size: 2.5rem; margin-bottom: 0.5rem;">Document Extraction Hub</h1>
            <p style="color: #718096; font-size: 1.1rem;">Enterprise-grade OCR and layout reconstruction pipeline</p>
        </div>
    """, unsafe_allow_html=True)

    if view_mode == "History":
        render_history()
    else:
        # Vùng Upload File
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><span>📥</span> Upload Target Document</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop your image or PDF here...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
    
        # Nếu người dùng đổi file khác, xóa kết quả cũ
        if uploaded_file is not None:
            if "processed_filename" in st.session_state and st.session_state["processed_filename"] != uploaded_file.name:
                if "extraction_result" in st.session_state:
                    del st.session_state["extraction_result"]
                st.session_state["processed_filename"] = uploaded_file.name
    
            # Nút chạy Pipeline
            if st.button("🚀 Execute Pipeline", use_container_width=True, type="primary"):
                with st.spinner("Analyzing document structure & extracting data via Local AI..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(API_URL, files=files)
                        
                        if response.status_code == 200:
                            st.session_state["extraction_result"] = response.json()
                            st.session_state["processed_filename"] = uploaded_file.name
                        else:
                            st.error(f"Lỗi từ Server: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Không thể kết nối đến Backend FastAPI: {e}")
    
            # Chỉ hiển thị kết quả nếu đã có trong session_state cho file hiện tại
            if "extraction_result" in st.session_state and st.session_state.get("processed_filename") == uploaded_file.name:
                result = st.session_state["extraction_result"]
                st.success(f"✨ Successfully processed in {result.get('processing_time_ms', 0)/1000:.2f}s!")
                
                # KPI Cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Detected Type", value=result.get("document_type", "Unknown").title().replace("_", " "))
                with col2:
                    quality = result.get("quality_class", "N/A").title().replace("_", " ")
                    score = result.get('quality_score', 0) * 100
                    st.metric(label="Layout Score", value=f"{score:.1f}%", delta=quality, delta_color="normal" if score > 70 else "inverse")
                with col3:
                    tables_count = len(result.get("tables", []))
                    blocks_count = len(result.get("raw_ocr", []))
                    st.metric(label="Data Elements", value=f"{tables_count} Tables", delta=f"{blocks_count} Text Blocks", delta_color="off")
                
                file_stem = Path(uploaded_file.name).stem
                md_filename = f"{file_stem}.md"
                md_file_path = Path(f"data/processed/{md_filename}")
                
                # Đưa toàn bộ kết quả vào một expander để giống hệt UI bên History
                with st.expander(f"📄 {md_filename}", expanded=True):
                    # Chia 2 cột (Trái: Hình ảnh gốc | Phải: Kết quả Markdown)
                    img_col, txt_col = st.columns([1, 1.2])
                    
                    with img_col:
                        st.markdown("**Source Image**")
                        img_tab, = st.tabs(["🖼️ Image View"])
                        with img_tab:
                            with st.container(height=800):
                                st.image(uploaded_file, use_container_width=True)
                        
                    with txt_col:
                        st.markdown("**Extracted Markdown**")
                        
                        current_md = result.get("markdown", "No text extracted.")
                        
                        tab1, tab2 = st.tabs(["👁️ Preview", "✏️ Edit Markdown"])
                        
                        with tab1:
                            # Hiển thị Markdown qua component chuyên dụng (Tăng height bằng ảnh)
                            with st.container(height=800):
                                st.markdown(current_md)
                            
                            st.write("")
                            
                            # Nút tải xuống dài nằm ngang dưới bảng markdown
                            st.download_button(
                                label="📥 Download Markdown",
                                data=current_md,
                                file_name=md_filename,
                                mime="text/markdown",
                                use_container_width=True,
                                type="primary"
                            )
                            
                        with tab2:
                            edited_md = st.text_area("Chỉnh sửa Markdown", value=current_md, height=800, label_visibility="collapsed", key=f"edit_up_{file_stem}")
                            if st.button("💾 Lưu thay đổi (Save to file)", type="primary", use_container_width=True, key=f"save_up_{file_stem}"):
                                # Lưu vào session_state để Preview tự nhận diện
                                st.session_state["extraction_result"]["markdown"] = edited_md
                                
                                # Cố gắng lưu vào file gốc trên ổ cứng nếu nó tồn tại
                                if md_file_path.exists():
                                    with open(md_file_path, "w", encoding="utf-8") as f:
                                        f.write(edited_md)
                                else:
                                    # Nếu file chưa tồn tại (do pipeline chưa save kịp?), cố gắng tạo mới
                                    md_file_path.parent.mkdir(parents=True, exist_ok=True)
                                    with open(md_file_path, "w", encoding="utf-8") as f:
                                        f.write(edited_md)
                                        
                                st.success("Đã cập nhật dữ liệu gốc thành công!")
                                st.rerun()
                
                st.write("---")
                
                # Chi tiết Tables & OCR Blocks
                with st.expander("🛠️ View System Diagnostics & Metadata"):
                    st.json(result.get("metadata", {}))
