#!/usr/bin/env bash
set -euo pipefail

# --- PODESI OVO PO POTREBI ---
PROJECT_ID="math-bot-465513"
REGION="europe-west1"
SERVICE="matbot"
AR_REPO="matbot"   # Artifact Registry repo ime (docker)
# ----------------------------

echo "▶ Project: $PROJECT_ID  Region: $REGION  Service: $SERVICE"
gcloud config set project "$PROJECT_ID" >/dev/null

# Jedinstven tag za verziju (koristi se kao BUILD_ID)
TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE="europe-west1-docker.pkg.dev/$PROJECT_ID/$AR_REPO/$SERVICE:$TAG"

# (Opcionalno) Kreiraj AR repo ako ne postoji
if ! gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" >/dev/null 2>&1; then
  echo "ℹ️  Artifact Registry repo '$AR_REPO' ne postoji, kreiram..."
  gcloud artifacts repositories create "$AR_REPO" \
    --repository-format=docker --location="$REGION" \
    --description="Images for $SERVICE"
fi

echo "▶ Build + push -> $IMAGE"
gcloud builds submit --tag "$IMAGE" .

echo "▶ Deploy na Cloud Run"
gcloud run deploy "$SERVICE" \
  --image="$IMAGE" \
  --region="$REGION" \
  --platform=managed \
  --quiet \
  --allow-unauthenticated \
  --set-env-vars "BUILD_ID=$TAG"

echo "▶ Prebaci traffic na najnoviju spremnu reviziju"
gcloud run services update-traffic "$SERVICE" \
  --region="$REGION" --to-latest

RUN_URL="$(gcloud run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
echo "✅ Gotovo. URL:"
echo "   $RUN_URL/?v=$TAG"
