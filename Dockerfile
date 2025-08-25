# Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instaliraj pakete
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App fajlovi
COPY . .

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

# Najjednostavnije: startaj Flask app direktno (sluša 0.0.0.0:8080 u app.py)
CMD ["python", "app.py"]
