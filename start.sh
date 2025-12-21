#!/bin/bash

# Voice Agent - Script di Avvio
# =============================

set -e

# Colori
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[Voice Agent]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[Warning]${NC} $1"
}

error() {
    echo -e "${RED}[Error]${NC} $1"
}

# Directory del progetto
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

log "=========================================="
log "🎙️  Voice Agent - Sistema di Avvio"
log "=========================================="

# 1. Verifica ambiente virtuale
if [ ! -d "venv" ]; then
    log "Creo ambiente virtuale Python..."
    python3 -m venv venv
fi

log "Attivo ambiente virtuale..."
source venv/bin/activate

# 2. Installa dipendenze se necessario
if [ ! -f "venv/.deps_installed" ]; then
    log "Installo dipendenze Python..."
    pip install -r requirements.txt
    touch venv/.deps_installed
fi

# 3. Copia .env se non esiste
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        log "Creo file .env da template..."
        cp env.example .env
    else
        warn "File env.example non trovato!"
    fi
fi

# 4. Avvia Docker services
log "Avvio LiveKit e Redis..."
docker-compose up -d

# Attendi che LiveKit sia pronto
log "Attendo che LiveKit sia pronto..."
sleep 5

# 5. Verifica Ollama
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log "✓ Ollama in esecuzione"
else
    warn "Ollama non sembra essere in esecuzione"
    warn "Avvia Ollama in un altro terminale: ollama serve"
fi

# 6. Avvia i servizi Python
log "Avvio Web Server e Voice Agent..."

# Web Server in background
python server.py &
WEB_PID=$!

sleep 2

# Voice Agent in background
python -m agent.main &
AGENT_PID=$!

# Cleanup function
cleanup() {
    log "Arresto servizi..."
    kill $WEB_PID 2>/dev/null || true
    kill $AGENT_PID 2>/dev/null || true
    docker-compose down
    log "Fatto!"
    exit 0
}

trap cleanup SIGINT SIGTERM

log "=========================================="
log "🎉 Sistema pronto!"
log "📱 Apri http://localhost:8080 nel browser"
log "🛑 Premi Ctrl+C per arrestare"
log "=========================================="

# Attendi
wait





