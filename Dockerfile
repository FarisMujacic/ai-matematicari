# Dockerfile (prod, Gunicorn, Cloud Run)
FROM python:3.11-slim

# Brži/čistiji logovi
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Sistemske zavisnosti (psycopg2/libpq, build tools, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Cloud Run prosljeđuje PORT (ne hardkodirati)
# Start preko gunicorna: 2 radnika, 8 threadova, timeouti prilagođeni OpenAI pozivima
CMD ["gunicorn",
     "-b", "0.0.0.0:${PORT}",
     "-w", "2",
     "-k", "gthread",
     "--threads", "8",
     "--timeout", "120",
     "--graceful-timeout", "90",
     "app:app"]
