#!/bin/bash
# Script per pre-caricare il modello Ollama in GPU e mantenerlo in memoria
# Viene eseguito all'avvio dello stack Docker per garantire che il modello
# sia pronto a rispondere immediatamente alle chiamate.

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gpt-oss:20b}"
MAX_RETRIES=30
RETRY_INTERVAL=5

echo "🔄 Ollama Preload: caricamento modello ${OLLAMA_MODEL}..."
echo "   Host: ${OLLAMA_HOST}"

# Attendi che Ollama sia disponibile
echo "⏳ Attendo che Ollama sia raggiungibile..."
retries=0
while [ $retries -lt $MAX_RETRIES ]; do
    if curl -s "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; then
        echo "✅ Ollama è raggiungibile"
        break
    fi
    retries=$((retries + 1))
    echo "   Tentativo ${retries}/${MAX_RETRIES}..."
    sleep $RETRY_INTERVAL
done

if [ $retries -eq $MAX_RETRIES ]; then
    echo "❌ Ollama non raggiungibile dopo ${MAX_RETRIES} tentativi. Uscita."
    exit 1
fi

# Verifica se il modello è già caricato in GPU
echo "🔍 Verifica se il modello è già caricato..."
LOADED=$(curl -s "${OLLAMA_HOST}/api/ps" 2>/dev/null)
if echo "$LOADED" | grep -q "${OLLAMA_MODEL}"; then
    echo "✅ Modello ${OLLAMA_MODEL} già caricato in GPU. Nessuna azione necessaria."
    exit 0
fi

# Pre-carica il modello inviando una richiesta minimale con keep_alive=-1
echo "🚀 Caricamento modello ${OLLAMA_MODEL} in GPU (keep_alive=-1)..."
RESPONSE=$(curl -s -X POST "${OLLAMA_HOST}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"${OLLAMA_MODEL}\", \"prompt\": \"ciao\", \"keep_alive\": -1, \"stream\": false}" \
    --max-time 300)

if [ $? -eq 0 ]; then
    echo "✅ Modello ${OLLAMA_MODEL} caricato con successo!"
    
    # Verifica finale
    echo "🔍 Verifica caricamento..."
    curl -s "${OLLAMA_HOST}/api/ps" 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('models', [])
    if models:
        for m in models:
            name = m.get('name', 'unknown')
            size = m.get('size', 0)
            size_gb = size / (1024**3)
            expires = m.get('expires_at', 'never')
            print(f'   Modello: {name}')
            print(f'   Dimensione VRAM: {size_gb:.1f} GB')
            print(f'   Scadenza: {expires}')
    else:
        print('   ⚠️  Nessun modello trovato caricato')
except:
    print('   Impossibile leggere stato modelli')
" 2>/dev/null
    
    echo "✅ Preload completato! Il modello resterà in GPU indefinitamente."
else
    echo "❌ Errore durante il caricamento del modello!"
    exit 1
fi
