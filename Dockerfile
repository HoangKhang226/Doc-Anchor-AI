# Sử dụng Python 3.11 mỏng nhẹ làm base image
FROM python:3.11-slim

# Cài đặt các thư viện hệ thống (bắt buộc phải có để chạy OpenCV và PaddleOCR)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy file cấu hình trước để tận dụng Docker Cache cho bước cài thư viện
COPY requirements.txt .

# Cài đặt thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào Container
COPY . .

# Expose các port cần thiết
EXPOSE 8000 8501
