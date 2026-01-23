# Use Python 3.11 slim - Debian Bookworm
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# Split into separate RUN commands for better debugging and layer caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV dependencies (correct package names for Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgthread-2.0-0 \
    libgl1 \
    libgtk-3-0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download additional Tesseract language data
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata/ && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/mar.traineddata -O /usr/share/tesseract-ocr/5/tessdata/mar.traineddata && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/tam.traineddata -O /usr/share/tesseract-ocr/5/tessdata/tam.traineddata && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/tel.traineddata -O /usr/share/tesseract-ocr/5/tessdata/tel.traineddata && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/guj.traineddata -O /usr/share/tesseract-ocr/5/tessdata/guj.traineddata && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/urd.traineddata -O /usr/share/tesseract-ocr/5/tessdata/urd.traineddata

# Copy requirements and install Python packages
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploaded_images

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT}/ || exit 1

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 1