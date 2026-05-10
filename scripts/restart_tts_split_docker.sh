#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

COMPOSE_FILE="docker-compose.tts.split.yml"

echo "Stopping split TTS stack (if running)..."
docker compose -f "$COMPOSE_FILE" down || true

echo "Starting split TTS stack..."
docker compose -f "$COMPOSE_FILE" up -d --build

echo "Waiting for proxy health on :8092..."
for i in {1..90}; do
  if curl -fsS "http://127.0.0.1:8092/health" >/dev/null 2>&1; then
    echo "TTS proxy is healthy."
    exit 0
  fi
  sleep 2
done

echo "TTS proxy did not become healthy in time."
echo "Check logs with: docker compose -f $COMPOSE_FILE logs -f tts-proxy tts-core tts-coqui tts-chatterbox tts-vibevoice"
exit 1
