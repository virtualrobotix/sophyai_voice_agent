# 🎙️ Voice Agent - Assistente Vocale Self-Hosted

Un sistema di assistente vocale WebRTC completamente self-hosted che utilizza:
- **LiveKit** per la comunicazione WebRTC in tempo reale
- **Whisper** (faster-whisper) per la trascrizione vocale
- **Ollama** con il modello `gpt-oss` per le risposte AI
- **TTS selezionabile** per la sintesi vocale in italiano

## 🏗️ Architettura

```
┌─────────────┐     WebRTC      ┌──────────────┐
│   Browser   │◄───────────────►│    LiveKit   │
│   Client    │                 │    Server    │
└─────────────┘                 └──────┬───────┘
                                       │
                                       ▼
                               ┌───────────────┐
                               │  Voice Agent  │
                               │   (Python)    │
                               └───────┬───────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
       ┌────────────┐           ┌────────────┐           ┌────────────┐
       │  Whisper   │           │   Ollama   │           │    TTS     │
       │   (STT)    │           │   (LLM)    │           │  Engine    │
       └────────────┘           └────────────┘           └────────────┘
```

## 📋 Requisiti

- **Docker** e **Docker Compose**
- **Python 3.10+**
- **Ollama** installato e in esecuzione
- Modello `gpt-oss` caricato in Ollama (o altro modello a scelta)
- 16GB+ RAM consigliati
- Microfono e altoparlanti

## 🚀 Installazione

### 1. Clona e configura

```bash
cd livekit-test

# Copia il file di configurazione
cp env.example .env

# Modifica le variabili se necessario
nano .env
```

### 2. Avvia LiveKit Server

```bash
# Avvia LiveKit e Redis
docker-compose up -d

# Verifica che siano in esecuzione
docker-compose ps
```

### 3. Installa dipendenze Python

```bash
# Crea ambiente virtuale
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure: .\venv\Scripts\activate  # Windows

# Installa dipendenze
pip install -r requirements.txt
```

### 3b. Installazione su server con GPU NVIDIA (CUDA)

Per sfruttare la GPU NVIDIA per VibeVoice e Whisper:

```bash
# Crea ambiente virtuale
python3 -m venv venv
source venv/bin/activate

# Installa PyTorch con supporto CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Installa le altre dipendenze
pip install -r requirements-cuda.txt

# Installa VibeVoice
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice && pip install -e . && cd ..

# Verifica che CUDA sia disponibile
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Nota**: Assicurati di avere i driver NVIDIA e CUDA Toolkit installati sul sistema.

### 4. Scarica modello Piper per italiano (opzionale)

```bash
# Crea directory modelli
mkdir -p models/piper

# Scarica modello italiano
curl -L -o models/piper/it_IT-riccardo-x_low.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx

curl -L -o models/piper/it_IT-riccardo-x_low.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx.json
```

### 5. Verifica Ollama

```bash
# Assicurati che Ollama sia in esecuzione
ollama list

# Se non hai gpt-oss, puoi usare un altro modello
# Modifica OLLAMA_MODEL in .env
```

## 🎮 Avvio

### Opzione 1: Tutto insieme

```bash
# Terminal 1: LiveKit (se non già avviato)
docker-compose up -d

# Terminal 2: Web Server
python server.py

# Terminal 3: Voice Agent
python -m agent.main
```

### Opzione 2: Script di avvio

```bash
# Avvia tutto
./start.sh

# Oppure con lo script Python
python run.py
```

### Accedi all'interfaccia

Apri il browser su: **http://localhost:8080**

## 🔊 Opzioni TTS

| Engine | Self-Hosted | Qualità | Velocità | Note |
|--------|-------------|---------|----------|------|
| **Piper** | ✅ Sì | Buona | Veloce | Consigliato per uso locale |
| **Coqui** | ✅ Sì | Alta | Media | Richiede più risorse |
| **Edge** | ❌ No | Ottima | Veloce | Usa API Microsoft gratuite |
| **Kokoro** | ✅ Sì | Alta | Media | Multilingua |

Puoi cambiare il TTS in tempo reale dall'interfaccia web.

## ⚙️ Configurazione

Modifica il file `.env`:

```env
# LiveKit
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret_dev_key_change_in_production

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss  # Cambia con il tuo modello

# Whisper
WHISPER_MODEL=base  # tiny, base, small, medium, large
WHISPER_LANGUAGE=it
WHISPER_DEVICE=cpu  # o cuda per GPU

# TTS Default
DEFAULT_TTS=piper  # piper, coqui, edge, kokoro

# Server
WEB_PORT=8080
LOG_LEVEL=INFO
```

## 📁 Struttura Progetto

```
livekit-test/
├── docker-compose.yml      # LiveKit + Redis
├── livekit.yaml           # Configurazione LiveKit
├── requirements.txt       # Dipendenze Python
├── env.example           # Template configurazione
├── server.py             # Web server FastAPI
├── run.py                # Script avvio completo
├── agent/
│   ├── __init__.py
│   ├── config.py         # Gestione configurazione
│   ├── main.py           # Agent principale
│   ├── stt/
│   │   ├── __init__.py
│   │   └── whisper_stt.py    # Whisper STT
│   ├── llm/
│   │   ├── __init__.py
│   │   └── ollama_llm.py     # Ollama LLM
│   └── tts/
│       ├── __init__.py
│       ├── base.py           # Interfaccia TTS
│       ├── piper_tts.py      # Piper TTS
│       ├── coqui_tts.py      # Coqui TTS
│       ├── edge_tts_engine.py # Edge TTS
│       └── kokoro_tts.py     # Kokoro TTS
├── web/
│   ├── index.html        # Frontend
│   └── app.js            # Client JavaScript
└── models/
    └── piper/            # Modelli Piper locali
```

## 🔧 Risoluzione Problemi

### LiveKit non si avvia
```bash
# Verifica i log
docker-compose logs livekit

# Riavvia
docker-compose down && docker-compose up -d
```

### Errore connessione Ollama
```bash
# Verifica che Ollama sia in esecuzione
curl http://localhost:11434/api/tags

# Se non risponde, avvia Ollama
ollama serve
```

### Whisper lento
- Usa un modello più piccolo: `WHISPER_MODEL=tiny`
- Se hai GPU NVIDIA: `WHISPER_DEVICE=cuda`

### TTS non funziona
- **Piper**: Assicurati di aver scaricato i modelli
- **Coqui**: Potrebbe richiedere download al primo avvio
- **Edge**: Richiede connessione internet

## 📝 API Endpoints

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/` | GET | Frontend web |
| `/api/health` | GET | Health check |
| `/api/token` | POST | Genera token LiveKit |
| `/api/tts/engines` | GET | Lista TTS disponibili |
| `/api/tts/{engine}/voices` | GET | Voci per engine |
| `/api/config` | GET | Configurazione pubblica |

## 🤝 Contribuire

1. Fork del repository
2. Crea un branch (`git checkout -b feature/nuova-feature`)
3. Commit (`git commit -am 'Aggiunge nuova feature'`)
4. Push (`git push origin feature/nuova-feature`)
5. Apri una Pull Request

## 📄 Licenza

MIT License

## 🙏 Crediti

- [LiveKit](https://livekit.io/) - WebRTC infrastructure
- [Whisper](https://github.com/openai/whisper) - Speech recognition
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Piper](https://github.com/rhasspy/piper) - Fast TTS
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Neural TTS
- [Edge TTS](https://github.com/rany2/edge-tts) - Microsoft Edge TTS





