#!/bin/bash
# Script per riavviare tutti i servizi (Docker + servizi MPS)

set -e

echo "🔄 Riavvio servizi Voice Agent"
echo ""

# Directory del progetto
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 1. Ferma i servizi Python esistenti (whisper e tts)
echo "🛑 Fermo servizi Python esistenti..."
pkill -f "whisper_server.py" 2>/dev/null || true
pkill -f "tts_server.py" 2>/dev/null || true
sleep 2

# 2. Ferma i container Docker
echo "🐳 Fermo container Docker..."
docker-compose down 2>/dev/null || true

# 3. Verifica che Docker sia in esecuzione
if ! docker info > /dev/null 2>&1; then
    echo "⚠️  Docker non è in esecuzione. Avvia Docker Desktop e riprova."
    exit 1
fi

# 4. Avvia i container Docker
echo ""
echo "🐳 Avvio container Docker..."
docker-compose up -d --build

# Attendi che i servizi siano pronti
echo "⏳ Attendo che i servizi Docker siano pronti..."
sleep 5

# 5. Determina quale Python usare
if [ -f "$DIR/venv/bin/python" ]; then
    PYTHON="$DIR/venv/bin/python"
elif [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON="$DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

echo "🐍 Usando Python: $PYTHON"

# 6. Avvia server Whisper con GPU/MPS sulla porta 8091
echo ""
echo "🔊 Avvio server Whisper (GPU/MPS) sulla porta 8091..."
export WHISPER_MODEL=small
export WHISPER_LANGUAGE=it
export WHISPER_DEVICE=auto  # Rileva automaticamente MPS
export WHISPER_PORT=8091
cd "$DIR"
nohup "$PYTHON" whisper_server.py > "$DIR/whisper_server.log" 2>&1 &
WHISPER_PID=$!
echo "   PID: $WHISPER_PID"
echo "   Log: $DIR/whisper_server.log"
sleep 5

# Verifica che Whisper sia avviato
if curl -s http://localhost:8091/ > /dev/null 2>&1; then
    echo "   ✅ Whisper server attivo"
else
    echo "   ⚠️  Whisper server potrebbe non essere ancora pronto"
fi

# 7. Avvia server TTS con GPU/MPS sulla porta 8092
echo ""
echo "🎤 Avvio server TTS (GPU/MPS) sulla porta 8092..."
cd "$DIR"
nohup "$PYTHON" tts_server.py --port 8092 > "$DIR/tts_server.log" 2>&1 &
TTS_PID=$!
echo "   PID: $TTS_PID"
echo "   Log: $DIR/tts_server.log"
sleep 5

# Verifica che TTS sia avviato
if curl -s http://localhost:8092/health > /dev/null 2>&1; then
    echo "   ✅ TTS server attivo"
else
    echo "   ⚠️  TTS server potrebbe non essere ancora pronto"
fi

echo ""
echo "✅ Tutti i servizi sono stati riavviati!"
echo ""

# Rileva IP della rete locale
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo "unknown")

echo "📌 Endpoint LOCALI (da questo computer):"
echo "   - Web UI: https://localhost:8080"
echo "   - Whisper Server: http://localhost:8091"
echo "   - TTS Server: http://localhost:8092"
echo "   - LiveKit: ws://localhost:7880"
echo ""
echo "📌 Endpoint RETE (da altri dispositivi):"
echo "   - Web UI: https://$LOCAL_IP:8080"
echo "   - LiveKit: ws://$LOCAL_IP:7880"
echo ""
echo "📋 Log:"
echo "   - Docker: docker logs -f voice-agent-web"
echo "   - Docker Agent: docker logs -f voice-agent-worker"
echo "   - Whisper: tail -f whisper_server.log"
echo "   - TTS: tail -f tts_server.log"
echo ""
echo "💡 Per fermare i servizi:"
echo "   - Docker: docker-compose down"
echo "   - Python: pkill -f 'whisper_server.py|tts_server.py'"
