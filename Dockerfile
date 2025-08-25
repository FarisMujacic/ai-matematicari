# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# sistemski paketi (curl + build deps po potrebi)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# radni dir
WORKDIR /app

# deps
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# app
COPY . /app

# Cloud Run koristi PORT env var (default 8080)
ENV PORT=8080

# gunicorn (bolje za proizvodnju nego flask built-in)
CMD exec gunicorn -w 2 -k gthread -t 0 -b :$PORT app:app
