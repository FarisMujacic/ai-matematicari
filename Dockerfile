# Dockerfile (prod, Gunicorn + veći timeout)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app
COPY . .

# Cloud Run sluša $PORT
ENV PORT=8080
# Veći timeout + thread worker (dobro za IO-bound pozive ka OpenAI)
ENV GUNICORN_CMD_ARGS="--bind 0.0.0.0:${PORT} --timeout 600 --graceful-timeout 90 -k gthread --threads 8 --workers 2"

# start
CMD ["gunicorn", "app:app"]
