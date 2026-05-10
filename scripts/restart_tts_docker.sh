#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

COMPOSE_FILE="docker-compose.tts.yml"

echo "Stopping dedicated TTS container (if running)..."
docker compose -f "$COMPOSE_FILE" down || true

echo "Starting dedicated TTS container..."
docker compose -f "$COMPOSE_FILE" up -d --build

echo "Waiting for health endpoint on :8092..."
for i in {1..30}; do
  if curl -fsS "http://127.0.0.1:8092/health" >/dev/null 2>&1; then
    echo "TTS server is healthy."
    exit 0
  fi
  sleep 2
done

echo "TTS server did not become healthy in time."
echo "Check logs with: docker compose -f $COMPOSE_FILE logs -f tts"
exit 1
