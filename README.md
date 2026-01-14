# 🎙️ Voice Agent - Assistente Vocale Self-Hosted

Un sistema di assistente vocale WebRTC completamente self-hosted che utilizza:
- **LiveKit** per la comunicazione WebRTC in tempo reale
- **Whisper** (faster-whisper) per la trascrizione vocale
- **Multi-LLM**: Ollama (locale) o OpenRouter (cloud con 100+ modelli)
- **TTS selezionabile** per la sintesi vocale in italiano
- **PostgreSQL** per persistenza chat e configurazione

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
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
  ┌────────────┐              ┌────────────────┐             ┌────────────┐
  │  Whisper   │              │  LLM Provider  │             │    TTS     │
  │   (STT)    │              │ Ollama/OpenRT  │             │  Engines   │
  └────────────┘              └────────────────┘             └────────────┘
                                       │
                                       ▼
                               ┌───────────────┐
                               │  PostgreSQL   │
                               │  (Database)   │
                               └───────────────┘
```

## 📋 Requisiti

- **Docker** e **Docker Compose**
- **Python 3.10+**
- **Ollama** installato e in esecuzione (per LLM locale)
- Modello `gpt-oss` caricato in Ollama (o altro modello a scelta)
- 16GB+ RAM consigliati (32GB per modelli TTS avanzati)
- Microfono e altoparlanti
- **GPU NVIDIA** (opzionale, consigliato per VibeVoice/Chatterbox)

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

Per sfruttare la GPU NVIDIA per VibeVoice, Chatterbox e Whisper:

```bash
# Crea ambiente virtuale
python3 -m venv venv
source venv/bin/activate

# Installa PyTorch con supporto CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Installa le altre dipendenze
pip install -r requirements-cuda.txt

# Installa VibeVoice (opzionale)
git clone https://github.com/microsoft/VibeVoice.git vibevoice_repo
cd vibevoice_repo && pip install -e . && cd ..

# Installa Chatterbox (opzionale)
pip install chatterbox-tts

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

| Engine | Self-Hosted | Qualità | Velocità | Lingue | Note |
|--------|-------------|---------|----------|--------|------|
| **Piper** | ✅ Sì | Buona | Veloce | Multi | Consigliato per uso locale, leggero |
| **Coqui** | ✅ Sì | Alta | Media | Multi | Richiede più risorse |
| **Edge** | ❌ No | Ottima | Veloce | Multi | Usa API Microsoft gratuite |
| **Kokoro** | ✅ Sì | Alta | Media | Multi | Multilingua, buona qualità |
| **VibeVoice** | ✅ Sì | Eccellente | Veloce | 6 | Microsoft, streaming real-time, multi-speaker |
| **Chatterbox** | ✅ Sì | Eccellente | Media | 23 | Resemble AI, voice cloning, emotion control |

Puoi cambiare il TTS in tempo reale dall'interfaccia web.

### VibeVoice (Microsoft)

TTS espressivo con streaming in tempo reale (~300ms latenza):

- **Modelli**: `realtime` (bassa latenza) o `longform` (alta qualità)
- **Speaker**: 4 speaker disponibili
- **Lingue**: Italiano, Inglese, Cinese, Spagnolo, Francese, Tedesco
- **Richiede GPU** per prestazioni ottimali

```env
VIBEVOICE_MODEL=realtime
VIBEVOICE_LANGUAGE=it
VIBEVOICE_SPEAKER=speaker_1
VIBEVOICE_SPEED=1.0
VIBEVOICE_GPU=true
```

### Chatterbox (Resemble AI)

TTS state-of-the-art con voice cloning e emotion control:

- **Modelli**: `standard`, `multilingual`, `turbo`
- **Lingue**: 23 lingue supportate (incluso Italiano)
- **Voice Cloning**: Clona voce da file audio di riferimento
- **Emotion Control**: Controllo esagerazione e CFG weight

```env
CHATTERBOX_MODEL=multilingual
CHATTERBOX_LANGUAGE=it
CHATTERBOX_DEVICE=auto
# Opzionale: voice cloning
CHATTERBOX_AUDIO_PROMPT_PATH=/path/to/voice.wav
# Opzionale: emotion control
CHATTERBOX_EXAGGERATION=0.5
```

## 🤖 Opzioni LLM

| Provider | Locale | Modelli | Note |
|----------|--------|---------|------|
| **Ollama** | ✅ Sì | Locali | LLM locale, privacy totale |
| **OpenRouter** | ❌ No | 100+ | Accesso a GPT-4, Claude, Gemini, ecc. |

### OpenRouter

Per usare OpenRouter (accesso a GPT-4, Claude, ecc.):

1. Registrati su [openrouter.ai](https://openrouter.ai)
2. Crea una API key
3. Configura nel file `.env`:

```env
OPENROUTER_API_KEY=sk-or-xxx...
OPENROUTER_MODEL=openai/gpt-4-turbo
```

Modelli consigliati per OpenRouter:
- `openai/gpt-4-turbo` - Veloce e intelligente
- `anthropic/claude-3-opus` - Alta qualità
- `google/gemini-pro` - Buon rapporto qualità/prezzo
- `mistralai/mistral-7b-instruct` - Economico e veloce

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

# OpenRouter (opzionale, alternativa a Ollama)
# OPENROUTER_API_KEY=sk-or-xxx...
# OPENROUTER_MODEL=openai/gpt-3.5-turbo

# Whisper
WHISPER_MODEL=base  # tiny, base, small, medium, large
WHISPER_LANGUAGE=it
WHISPER_DEVICE=cpu  # o cuda per GPU

# TTS Default (piper, coqui, edge, kokoro, vibevoice, chatterbox)
DEFAULT_TTS=piper

# Piper TTS
PIPER_MODEL=it_IT-riccardo-x_low
PIPER_SPEAKER=0

# Edge TTS
EDGE_VOICE=it-IT-DiegoNeural

# Coqui TTS
COQUI_MODEL=tts_models/it/mai_female/glow-tts

# Kokoro TTS
KOKORO_VOICE=it_sara

# VibeVoice TTS (Microsoft)
VIBEVOICE_MODEL=realtime
VIBEVOICE_LANGUAGE=it
VIBEVOICE_SPEAKER=speaker_1
VIBEVOICE_SPEED=1.0
VIBEVOICE_GPU=true

# Chatterbox TTS (Resemble AI)
CHATTERBOX_MODEL=multilingual
CHATTERBOX_LANGUAGE=it
CHATTERBOX_DEVICE=auto

# Video/Vision Analysis (opzionale)
VIDEO_ANALYSIS_ENABLED=true
OPENROUTER_VISION_MODEL=openai/gpt-4-vision-preview
OLLAMA_VISION_MODEL=llava

# Server
WEB_PORT=8080
LOG_LEVEL=INFO
```

## 📁 Struttura Progetto

```
livekit-test/
├── docker-compose.yml       # LiveKit + Redis + PostgreSQL
├── Dockerfile              # Docker build principale
├── Dockerfile.agent        # Docker build per agent
├── livekit.yaml            # Configurazione LiveKit
├── livekit-host.yaml       # Config LiveKit per host
├── livekit-local.yaml      # Config LiveKit locale
├── sip-config.yaml         # Configurazione SIP
├── requirements.txt        # Dipendenze Python
├── requirements-cuda.txt   # Dipendenze con CUDA
├── env.example             # Template configurazione
├── server.py               # Web server FastAPI
├── run.py                  # Script avvio completo
├── tts_server.py           # Server TTS dedicato
├── whisper_server.py       # Server Whisper dedicato
├── start.sh                # Script avvio singolo
├── start_all.sh            # Script avvio tutti i servizi
├── setup_cuda.sh           # Setup CUDA
├── agent/
│   ├── __init__.py
│   ├── config.py           # Gestione configurazione
│   ├── main.py             # Agent principale
│   ├── stt/
│   │   ├── __init__.py
│   │   └── whisper_stt.py      # Whisper STT
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama_llm.py       # Ollama LLM
│   │   └── openrouter_llm.py   # OpenRouter LLM
│   └── tts/
│       ├── __init__.py
│       ├── base.py             # Interfaccia TTS base
│       ├── piper_tts.py        # Piper TTS
│       ├── coqui_tts.py        # Coqui TTS
│       ├── edge_tts_engine.py  # Edge TTS (Microsoft)
│       ├── kokoro_tts.py       # Kokoro TTS
│       ├── vibevoice_tts.py    # VibeVoice TTS (Microsoft)
│       └── chatterbox_tts.py   # Chatterbox TTS (Resemble AI)
├── db/
│   ├── __init__.py
│   ├── database.py         # Gestione PostgreSQL
│   └── schema.sql          # Schema database
├── config/
│   └── tts_config.json     # Configurazione TTS
├── web/
│   ├── index.html          # Frontend principale
│   ├── debug.html          # Pagina debug
│   └── app.js              # Client JavaScript
├── vibevoice_repo/         # Repository VibeVoice (Microsoft)
│   └── ...
└── models/
    └── piper/              # Modelli Piper locali
```

## 🗄️ Database

Il sistema utilizza PostgreSQL per:
- **Persistenza chat**: Salvataggio conversazioni
- **Configurazione dinamica**: Settings modificabili da UI
- **Cronologia messaggi**: Storico completo delle interazioni

### Schema Database

| Tabella | Descrizione |
|---------|-------------|
| `settings` | Configurazione key-value |
| `chats` | Sessioni di conversazione |
| `messages` | Messaggi individuali |

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
- **VibeVoice**: Richiede GPU e installazione da repo Microsoft
- **Chatterbox**: `pip install chatterbox-tts`

### VibeVoice: CUDA non disponibile
```bash
# Verifica CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Se False, installa PyTorch CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Chatterbox: Errore caricamento modello
```bash
# Reinstalla con supporto corretto
pip uninstall chatterbox-tts
pip install chatterbox-tts

# Per macOS con MPS
CHATTERBOX_DEVICE=mps  # nel .env
```

## 📝 API Endpoints

| Endpoint | Metodo | Descrizione |
|----------|--------|-------------|
| `/` | GET | Frontend web |
| `/api/health` | GET | Health check |
| `/api/token` | POST | Genera token LiveKit |
| `/api/tts/engines` | GET | Lista TTS disponibili |
| `/api/tts/{engine}/voices` | GET | Voci per engine |
| `/api/config` | GET | Configurazione pubblica |
| `/api/settings` | GET/POST | Gestione impostazioni |
| `/api/chats` | GET/POST | Gestione chat |
| `/api/chats/{id}/messages` | GET | Messaggi di una chat |

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
- [OpenRouter](https://openrouter.ai/) - Multi-model LLM API
- [Piper](https://github.com/rhasspy/piper) - Fast TTS
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Neural TTS
- [Edge TTS](https://github.com/rany2/edge-tts) - Microsoft Edge TTS
- [VibeVoice](https://github.com/microsoft/VibeVoice) - Microsoft Real-time TTS
- [Chatterbox](https://github.com/resemble-ai/chatterbox) - Resemble AI TTS
- [Kokoro](https://github.com/hexgrad/kokoro) - Multilingual TTS


