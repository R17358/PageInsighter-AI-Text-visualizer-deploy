# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopencv-dev \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download additional language data files directly from GitHub
RUN wget -q https://github.com/tesseract-ocr/tessdata/raw/main/mar.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/ || true && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/tam.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/ || true && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/tel.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/ || true && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/guj.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/ || true && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/urd.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/ || true

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Render will override this with PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run the application
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
