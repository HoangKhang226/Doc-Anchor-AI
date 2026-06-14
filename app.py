import streamlit as st
from src.ui.dashboard_view import render_dashboard

# Thiết lập cấu hình trang
st.set_page_config(page_title="Doc Anchor AI", page_icon="📄", layout="wide")

if __name__ == "__main__":
    render_dashboard()
