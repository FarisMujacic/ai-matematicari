cat > ~/matbot-refresh.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

# --- Parametri ---
PROJECT_ID="$(gcloud config get-value project)"
REGION="europe-west1"
SERVICE="matbot"
REPO="https://github.com/CORBiH/ai-matematicari.git"

echo "Project: $PROJECT_ID | Region: $REGION | Service: $SERVICE"

# --- Svježi klon u /tmp (bez lokalnih repoa, bez konflikata) ---
rm -rf /tmp/ai-matematicari
git clone --depth 1 "$REPO" /tmp/ai-matematicari
cd /tmp/ai-matematicari
echo "Git rev: $(git rev-parse --short HEAD)"

# --- Build+Deploy preko cloudbuild.yaml (u CI okruženju) ---
# cloudbuild.yaml već radi: build image -> push -> gcloud run deploy -> update traffic
gcloud builds submit --config cloudbuild.yaml .

# --- URL i smoke testovi ---
RUN_URL="$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.address.url)')"
echo "Run URL: $RUN_URL"

echo "---- Health (/healthz) ----"
(set -x; curl -fsS "$RUN_URL/healthz" || true)

echo "---- Root (/) ----"
(set -x; curl -fsS -o /dev/null -w "%{http_code}\n" "$RUN_URL/")

echo "---- /submit quick test ----"
(set -x; curl -fsS -X POST "$RUN_URL/submit" -F 'razred=7' -F 'user_text=Koliko je 2+3?' || true)

# --- Logovi zadnjih ~10 min (bez --since koje na nekim verzijama ne radi) ---
echo "---- Logs (10m) ----"
gcloud logging read \
  'resource.type="cloud_run_revision" AND resource.labels.service_name="'"$SERVICE"'"' \
  --freshness=10m --limit=120 --format='value(textPayload)' || true

echo "Done."
SH
chmod +x ~/matbot-refresh.sh
