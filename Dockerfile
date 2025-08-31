FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OS paketi potrebni za build (psycopg2, Pillow, itd.)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
      curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# --- CMD (dinamički, koristi ENV var-ove) ---
# Defaulti: 1 worker, 8 threads, timeout 120s
CMD exec gunicorn \
  --bind :${PORT:-8080} \
  --workers ${WEB_CONCURRENCY:-1} \
  --threads ${THREADS:-8} \
  --timeout ${GUNICORN_TIMEOUT:-120} \
  --graceful-timeout 90 \
  --access-logfile - \
  --error-logfile - \
  app:app
