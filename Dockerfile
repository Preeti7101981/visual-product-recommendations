# Dockerfile for deploying this app on Hugging Face Spaces (Docker SDK).
# Hugging Face Spaces expects the app to listen on port 7860 by default.

FROM python:3.11-slim

WORKDIR /app

# System dependency needed by Pillow for some image formats
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["gunicorn", "src.app:app", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1"]
