# Ultra-minimal - guaranteed to work
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Install ONLY what we absolutely need
RUN apt-get update && \
    apt-get install -y tesseract-ocr wget && \
    rm -rf /var/lib/apt/lists/*

# Download language files
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata/ && \
    cd /usr/share/tesseract-ocr/5/tessdata/ && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata && \
    wget -q https://github.com/tesseract-ocr/tessdata/raw/main/hin.traineddata

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
