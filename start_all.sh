#!/bin/bash
# Script per avviare tutti i servizi

echo "🚀 Avvio servizi Voice Agent"
echo ""

# Verifica che livekit sia in esecuzione
if ! pgrep -x "livekit-server" > /dev/null; then
    echo "⚠️  LiveKit non è in esecuzione. Avvialo con:"
    echo "   livekit-server --dev"
    echo ""
fi

# Verifica che Ollama sia in esecuzione
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama non è in esecuzione. Avvialo con:"
    echo "   ollama serve"
    echo ""
fi

# Directory del progetto
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 1. Avvia server Whisper con GPU/MPS
echo "🔊 Avvio server Whisper (GPU/MPS) sulla porta 8090..."
if [ -f "$DIR/.venv/bin/python" ]; then
    PYTHON="$DIR/.venv/bin/python"
else
    PYTHON="python3"
fi

# Verifica se il server Whisper è già in esecuzione
if curl -s http://localhost:8090/ > /dev/null 2>&1; then
    echo "✅ Server Whisper già in esecuzione"
else
    WHISPER_MODEL=small WHISPER_LANGUAGE=it nohup $PYTHON whisper_server.py > whisper_server.log 2>&1 &
    echo "   PID: $!"
    echo "   Log: whisper_server.log"
    sleep 3
fi

# 2. Avvia Docker containers (web + agent)
echo ""
echo "🐳 Avvio containers Docker..."
docker-compose up -d --build

echo ""
echo "✅ Servizi avviati!"
echo ""

# Rileva IP della rete locale
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo "unknown")

echo "📌 Endpoint LOCALI (da questo computer):"
echo "   - Web UI: https://localhost:8080"
echo "   - Whisper Server: http://localhost:8090"
echo "   - LiveKit: ws://localhost:7880"
echo ""
echo "📌 Endpoint RETE (da altri dispositivi):"
echo "   - Web UI: https://$LOCAL_IP:8080"
echo "   - LiveKit: ws://$LOCAL_IP:7880"
echo ""
echo "💡 Per connessioni da altri dispositivi nella rete:"
echo "   1. Assicurati che il firewall permetta le porte 7880, 8080"
echo "   2. Collegati a https://$LOCAL_IP:8080 dal dispositivo"
echo ""
echo "📋 Log:"
echo "   - docker logs -f voice-agent-web"
echo "   - docker logs -f voice-agent-worker"
echo "   - tail -f whisper_server.log"









