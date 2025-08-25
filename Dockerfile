# Dockerfile
FROM python:3.11-slim

# Sistem dependencije (psycopg2, etc. – ako ti ne trebaju, možeš skratiti)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

# Najvažnije: ovdje forsiramo da app sluša na 0.0.0.0:8080
CMD exec gunicorn -w 1 --threads 8 -k gthread -t 0 -b 0.0.0.0:${PORT} app:app
