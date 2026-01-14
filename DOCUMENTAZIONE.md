# Documentazione Sistema Voice Agent - SophyAI Live Server

## Indice

1. [Introduzione e Panoramica](#1-introduzione-e-panoramica)
2. [Architettura del Sistema](#2-architettura-del-sistema)
3. [Moduli Dettagliati](#3-moduli-dettagliati)
4. [Flussi di Dati](#4-flussi-di-dati)
5. [Deployment e Infrastruttura](#5-deployment-e-infrastruttura)
6. [API Reference](#6-api-reference)
7. [Configurazione Avanzata](#7-configurazione-avanzata)

---

## 1. Introduzione e Panoramica

### 1.1 Descrizione del Sistema

**SophyAI Live Server** è un sistema di assistente vocale completamente self-hosted che utilizza WebRTC per comunicazioni in tempo reale. Il sistema permette conversazioni vocali bidirezionali con un assistente AI, supportando:

- **Speech-to-Text (STT)**: Trascrizione vocale in tempo reale tramite Whisper
- **Large Language Model (LLM)**: Generazione di risposte tramite Ollama (locale) o OpenRouter (cloud)
- **Text-to-Speech (TTS)**: Sintesi vocale con multiple engine (Piper, Coqui, Edge, Kokoro, VibeVoice, Chatterbox)
- **Video Analysis**: Analisi di immagini/video tramite modelli vision (opzionale)
- **Multi-user Support**: Supporto per più utenti nella stessa room
- **Persistenza**: Salvataggio conversazioni e configurazioni in PostgreSQL

### 1.2 Tecnologie Utilizzate

| Componente | Tecnologia | Versione/Note |
|------------|-----------|---------------|
| **WebRTC** | LiveKit | Server WebRTC per comunicazioni real-time |
| **Web Server** | FastAPI | Framework Python per API REST |
| **STT** | faster-whisper | Trascrizione vocale basata su Whisper |
| **LLM Locale** | Ollama | Runtime per modelli LLM locali |
| **LLM Cloud** | OpenRouter | API gateway per 100+ modelli (GPT-4, Claude, Gemini, ecc.) |
| **TTS** | Multiple engines | Piper, Coqui, Edge, Kokoro, VibeVoice, Chatterbox |
| **Database** | PostgreSQL | Persistenza dati e configurazioni |
| **Cache** | Redis | Cache per LiveKit |
| **Container** | Docker | Containerizzazione servizi |
| **Frontend** | HTML/JavaScript | Interfaccia web client |

### 1.3 Requisiti di Sistema

#### Requisiti Minimi
- **CPU**: 4 core
- **RAM**: 8GB
- **Storage**: 20GB liberi
- **Network**: Connessione internet (per Edge TTS e OpenRouter)

#### Requisiti Consigliati
- **CPU**: 8+ core
- **RAM**: 16GB+ (32GB per modelli TTS avanzati)
- **Storage**: 50GB+ (per modelli Whisper, TTS e LLM)
- **GPU**: NVIDIA GPU con CUDA (opzionale, consigliato per VibeVoice/Chatterbox/Whisper)
- **Network**: Connessione stabile

#### Software Richiesto
- Docker e Docker Compose
- Python 3.10+
- Ollama (per LLM locale)
- Node.js (opzionale, per sviluppo)

### 1.4 Diagramma Architetturale ad Alto Livello

```mermaid
graph TB
    subgraph Client["Client Browser"]
        Web[Web Interface<br/>HTML/JS]
    end
    
    subgraph Infrastructure["Infrastructure Layer"]
        LiveKit[LiveKit Server<br/>WebRTC]
        Redis[Redis<br/>Cache]
        Postgres[PostgreSQL<br/>Database]
    end
    
    subgraph Application["Application Layer"]
        WebServer[Web Server<br/>FastAPI]
        Agent[Voice Agent<br/>Worker]
    end
    
    subgraph Services["AI Services"]
        STT[Whisper STT<br/>Speech-to-Text]
        LLM_Ollama[Ollama LLM<br/>Local]
        LLM_OpenRouter[OpenRouter LLM<br/>Cloud]
        TTS[TTS Engines<br/>Multiple]
        Vision[Vision Models<br/>Optional]
    end
    
    subgraph External["External Services"]
        OllamaHost[Ollama Host<br/>localhost:11434]
        TTSHost[TTS Server<br/>localhost:8092]
    end
    
    Web -->|HTTPS/HTTP| WebServer
    Web -->|WebRTC| LiveKit
    LiveKit <-->|WebRTC| Agent
    LiveKit --> Redis
    WebServer --> Postgres
    Agent --> STT
    Agent --> LLM_Ollama
    Agent --> LLM_OpenRouter
    Agent --> TTS
    Agent --> Vision
    Agent --> Postgres
    LLM_Ollama --> OllamaHost
    TTS --> TTSHost
    WebServer --> LiveKit
```

---

## 2. Architettura del Sistema

### 2.1 Componenti Principali

Il sistema è composto da 5 componenti principali:

#### 2.1.1 Web Server (`server.py`)
Server FastAPI che gestisce:
- **Frontend**: Serve file HTML/JS statici
- **API REST**: Endpoints per configurazione, chat, TTS, LLM
- **Token Generation**: Genera token LiveKit per autenticazione client
- **Database Management**: Connessione e operazioni su PostgreSQL
- **HTTPS/HTTP**: Supporto per entrambi i protocolli

**File**: [`server.py`](server.py)

#### 2.1.2 Voice Agent (`agent/main.py`)
Worker LiveKit che orchestra:
- **STT Processing**: Riceve audio, invia a Whisper per trascrizione
- **LLM Integration**: Invia testo a LLM (Ollama o OpenRouter), riceve risposta
- **TTS Synthesis**: Converte risposta testuale in audio
- **Session Management**: Gestisce sessioni conversazionali multi-user
- **Video Analysis**: Estrae frame video e li analizza con modelli vision

**File**: [`agent/main.py`](agent/main.py)

#### 2.1.3 LiveKit Server
Server WebRTC che gestisce:
- **Media Streaming**: Audio/video in tempo reale
- **Room Management**: Gestione stanze e partecipanti
- **Agent Dispatch**: Dispatch automatico di agent workers

**Configurazione**: [`livekit.yaml`](livekit.yaml), [`livekit-local.yaml`](livekit-local.yaml)

#### 2.1.4 Database (PostgreSQL)
Persistenza per:
- **Settings**: Configurazioni key-value
- **Chats**: Sessioni conversazionali
- **Messages**: Storico messaggi

**Schema**: [`db/schema.sql`](db/schema.sql)

#### 2.1.5 Redis
Cache per LiveKit:
- Session state
- Room metadata
- Temporary data

### 2.2 Comunicazione tra Componenti

```mermaid
sequenceDiagram
    participant Browser
    participant WebServer
    participant LiveKit
    participant Agent
    participant STT
    participant LLM
    participant TTS
    participant DB
    
    Browser->>WebServer: GET / (Frontend)
    Browser->>WebServer: POST /api/token
    WebServer->>LiveKit: Create Room + Dispatch Agent
    WebServer-->>Browser: Token + LiveKit URL
    
    Browser->>LiveKit: WebRTC Connect (con token)
    LiveKit->>Agent: Agent Connected
    
    Browser->>LiveKit: Audio Stream (microfono)
    LiveKit->>Agent: Audio Frame
    Agent->>STT: Transcribe Audio
    STT-->>Agent: Text
    
    Agent->>DB: Load Settings
    DB-->>Agent: Settings
    Agent->>LLM: Generate Response
    LLM-->>Agent: Response Text
    
    Agent->>TTS: Synthesize Audio
    TTS-->>Agent: Audio Data
    Agent->>LiveKit: Audio Stream
    LiveKit->>Browser: Audio Playback
```

### 2.3 Stack Tecnologico

#### Backend
- **Python 3.10+**: Linguaggio principale
- **FastAPI**: Web framework asincrono
- **LiveKit Agents SDK**: Framework per agent WebRTC
- **asyncpg**: Driver PostgreSQL asincrono
- **aiohttp**: Client HTTP asincrono

#### AI/ML
- **faster-whisper**: STT engine
- **Ollama**: LLM runtime locale
- **OpenRouter**: LLM API gateway
- **PyTorch**: Framework ML (per TTS avanzati)
- **Transformers**: Libreria HuggingFace

#### Infrastructure
- **Docker**: Containerizzazione
- **Docker Compose**: Orchestrazione
- **PostgreSQL 16**: Database
- **Redis 7**: Cache

#### Frontend
- **HTML5**: Markup
- **JavaScript (ES6+)**: Logica client
- **WebRTC API**: Comunicazione real-time
- **Web Audio API**: Gestione audio

---

## 3. Moduli Dettagliati

### 3.1 Web Server (`server.py`)

Il Web Server è implementato in [`server.py`](server.py) e fornisce:

#### 3.1.1 FastAPI Application

```python
app = FastAPI(
    title="Voice Agent API",
    description="API per il Voice Agent WebRTC",
    version="1.0.0"
)
```

**Caratteristiche**:
- CORS abilitato per tutte le origini
- Middleware per logging
- Static files serving per frontend
- HTTPS/HTTP dual mode

#### 3.1.2 Endpoints Principali

##### Autenticazione e Token
- **`POST /api/token`**: Genera token LiveKit per partecipante
  - Verifica unicità nome utente nella room
  - Crea room se non esiste
  - Dispatch agent se non presente
  - Ritorna token JWT e URL WebSocket

##### Configurazione
- **`GET /api/config`**: Configurazione pubblica (LiveKit URL, TTS default, ecc.)
- **`GET /api/settings`**: Tutte le impostazioni dal database
- **`POST /api/settings`**: Aggiorna multiple impostazioni
- **`GET /api/settings/{key}`**: Singola impostazione
- **`PUT /api/settings/{key}`**: Aggiorna singola impostazione

##### LLM Management
- **`GET /api/ollama/models`**: Lista modelli Ollama disponibili
- **`POST /api/ollama/select`**: Seleziona modello Ollama
- **`GET /api/openrouter/models`**: Lista modelli OpenRouter (100+)
- **`POST /api/openrouter/select`**: Seleziona modello OpenRouter
- **`POST /api/openrouter/key`**: Salva API key OpenRouter

##### TTS Management
- **`GET /api/tts/engines`**: Lista TTS engines disponibili
- **`GET /api/tts/{engine}/voices`**: Voci disponibili per engine
- **`GET /api/tts/current`**: Configurazione TTS corrente
- **`POST /api/tts/current`**: Imposta TTS engine/voce
- **`POST /api/tts/test`**: Test sintesi TTS (ritorna audio WAV)

##### Chat Management
- **`GET /api/chats`**: Lista tutte le chat
- **`POST /api/chats`**: Crea nuova chat
- **`GET /api/chats/{id}`**: Dettagli chat con messaggi
- **`DELETE /api/chats/{id}`**: Elimina chat
- **`POST /api/chats/{id}/messages`**: Aggiungi messaggio

##### System Prompt
- **`GET /api/prompt`**: Prompt di sistema corrente
- **`POST /api/prompt`**: Aggiorna prompt di sistema

##### Status e Health
- **`GET /api/health`**: Health check semplice
- **`GET /api/status`**: Status dettagliato di tutti i servizi (LiveKit, Ollama, Agent, Whisper, SIP)
- **`GET /api/timing`**: Statistiche timing (STT, LLM, TTS, latenza)
- **`POST /api/timing`**: Aggiorna timing stats (chiamato dall'agent)

#### 3.1.3 Database Connection Management

Il server gestisce connessioni PostgreSQL tramite pool asincrono:

```python
async def get_database():
    """Get database instance, initialize if needed."""
    global _db
    if _db is None:
        from db.database import get_db
        _db = await get_db()
    return _db
```

**Lifecycle**:
- Connessione al startup
- Pool di connessioni (min 2, max 10)
- Chiusura graceful allo shutdown

#### 3.1.4 HTTPS/HTTP Configuration

Il server supporta entrambi i protocolli:
- **HTTPS** (porta 8443): Con certificati SSL self-signed
- **HTTP** (porta 8080): Per app mobile che non supportano certificati self-signed

**Certificati**: Cercati in `certs/key.pem` e `certs/cert.pem`

### 3.2 Voice Agent (`agent/main.py`)

Il Voice Agent è il cuore del sistema, implementato in [`agent/main.py`](agent/main.py).

#### 3.2.1 LiveKit Agent Worker

L'agent si registra come worker LiveKit e viene dispatchato automaticamente quando un client si connette:

```python
async def entrypoint(ctx: JobContext):
    """Entrypoint per l'agent LiveKit"""
    await ctx.connect()
    agent = VoiceAgent(ctx)
    await agent.start()
```

#### 3.2.2 Orchestrazione STT → LLM → TTS

Il flusso principale:

1. **Ricezione Audio**: L'agent riceve frame audio da LiveKit
2. **STT Processing**: Accumula audio fino a silenzio, invia a Whisper
3. **Trigger Detection**: Verifica se il messaggio contiene trigger (@sophyai) o se è single-user mode
4. **LLM Generation**: Invia testo a LLM (Ollama o OpenRouter) con context
5. **TTS Synthesis**: Converte risposta in audio tramite TTS engine configurato
6. **Audio Streaming**: Invia audio a LiveKit per playback

#### 3.2.3 Gestione Sessioni Conversazionali

L'agent mantiene:
- **Conversation History**: Ultimi N messaggi per context
- **Session State**: Stato della sessione corrente
- **User Context**: Informazioni utente (lingua rilevata, preferenze)

#### 3.2.4 Multi-user Support

L'agent supporta più utenti nella stessa room:
- **Mention-based Activation**: In multi-user mode, l'agent risponde solo se menzionato (@sophyai)
- **Single-user Auto-response**: Con un solo utente, risponde sempre
- **Force Response Mode**: Flag per forzare risposta sempre (toggle da frontend)

#### 3.2.5 Video Analysis Integration

L'agent può analizzare video/webcam tramite:
- **VideoFrameExtractor**: Estrae frame da video tracks LiveKit
- **MultimodalLLM**: Wrapper per modelli vision (GPT-4 Vision, LLaVA)
- **Function Tools**: Tool functions per analisi generica, documenti, età, ambiente

**File**: [`agent/main.py`](agent/main.py) (classi `VideoFrameExtractor`, `VisionAgent`, `MultimodalLLM`)

#### 3.2.6 Timing e Metriche

L'agent traccia metriche di performance:
- **STT Time**: Tempo trascrizione
- **LLM Time**: Tempo generazione risposta (incluso TTFT - Time To First Token)
- **TTS Time**: Tempo sintesi audio
- **End-to-End Latency**: Latenza totale da input audio a output audio
- **To First Audio**: Tempo fino al primo chunk audio

Le metriche vengono inviate al web server via API.

### 3.3 Speech-to-Text (`agent/stt/`)

#### 3.3.1 Whisper STT Implementation

Il modulo STT è implementato in [`agent/stt/whisper_stt.py`](agent/stt/whisper_stt.py).

**Classe Principale**: `WhisperSTT`

**Caratteristiche**:
- Utilizza `faster-whisper` per performance ottimali
- Supporta modelli: `tiny`, `base`, `small`, `medium`, `large`
- VAD (Voice Activity Detection) integrato
- Language detection automatica
- Streaming transcription support

**Inizializzazione**:
```python
stt = WhisperSTT(
    model_size="base",
    language="it",
    device="cpu",  # o "cuda" per GPU
    compute_type="int8"
)
```

**Metodi Principali**:
- `transcribe(audio_data, sample_rate)`: Trascrizione sincrona
- `transcribe_stream(audio_stream)`: Trascrizione streaming asincrona

#### 3.3.2 Modelli Supportati

| Modello | Dimensione | Velocità | Qualità | RAM Richiesta |
|---------|-----------|----------|---------|---------------|
| `tiny` | ~39MB | Molto veloce | Bassa | ~1GB |
| `base` | ~74MB | Veloce | Media | ~1GB |
| `small` | ~244MB | Media | Buona | ~2GB |
| `medium` | ~769MB | Lenta | Alta | ~5GB |
| `large` | ~1550MB | Molto lenta | Eccellente | ~10GB |

#### 3.3.3 Language Detection

Whisper rileva automaticamente la lingua se `language=None`. Il sistema usa la lingua rilevata per:
- TTS language selection (se auto_language è attivo)
- Context injection per LLM

#### 3.3.4 Performance e Ottimizzazioni

- **VAD Filtering**: Filtra silenzio per ridurre processing
- **Beam Search**: Configurabile (default beam_size=5)
- **Quantization**: Supporto int8 per CPU, float16 per GPU
- **Streaming**: Processing incrementale per bassa latenza

### 3.4 Large Language Model (`agent/llm/`)

Il sistema supporta due provider LLM:

#### 3.4.1 Ollama LLM (`agent/llm/ollama_llm.py`)

**Classe**: `OllamaLLM`

**Caratteristiche**:
- LLM completamente locale (privacy totale)
- Supporta qualsiasi modello Ollama (gpt-oss, llama2, mistral, ecc.)
- Streaming responses
- Conversation history management
- System prompt configurabile

**Inizializzazione**:
```python
llm = OllamaLLM(
    model="gpt-oss",
    host="http://localhost:11434",
    system_prompt="Sei un assistente..."
)
```

**Metodi**:
- `generate(user_message)`: Generazione sincrona
- `generate_stream(user_message)`: Generazione streaming asincrona
- `clear_history()`: Pulisce cronologia
- `set_system_prompt(prompt)`: Aggiorna prompt sistema

**Integrazione LiveKit**:
La classe `OllamaLLM` e `OllamaLLMStream` in [`agent/main.py`](agent/main.py) implementano l'interfaccia LiveKit Agents SDK.

#### 3.4.2 OpenRouter LLM (`agent/llm/openrouter_llm.py`)

**Classe**: `OpenRouterLLM`

**Caratteristiche**:
- Accesso a 100+ modelli (GPT-4, Claude, Gemini, ecc.)
- API compatibile OpenAI
- Supporto vision models
- Function calling support
- JSON mode support

**Modelli Popolari**:
- `openai/gpt-4-turbo`: Veloce e intelligente
- `anthropic/claude-3-opus`: Alta qualità
- `google/gemini-pro`: Buon rapporto qualità/prezzo
- `openai/gpt-4-vision-preview`: Supporto immagini

**Inizializzazione**:
```python
llm = OpenRouterLLM(
    api_key="sk-or-xxx...",
    model="openai/gpt-4-turbo"
)
```

**Integrazione LiveKit**:
Usa il plugin OpenAI di LiveKit con base URL OpenRouter.

#### 3.4.3 Conversation History Management

Entrambi i provider mantengono:
- **Message History**: Ultimi N messaggi (default 20)
- **System Prompt**: Prompt di sistema configurabile
- **Context Injection**: Contesto aggiuntivo iniettabile

#### 3.4.4 System Prompt Configuration

Il system prompt è configurabile via API (`/api/prompt`). Default:

```
Sei Sophy, assistente vocale ultra-veloce. PRIORITA ASSOLUTA: VELOCITA E SINTESI.

REGOLE FONDAMENTALI:
1. RISPOSTE ULTRA-BREVI: massimo 1-2 frasi, mai piu di 30 parole
2. VAI DRITTO AL PUNTO: niente preamboli, saluti inutili o ripetizioni
3. LINGUA: rispondi nella stessa lingua dell utente
```

#### 3.4.5 Function Calling Support

I modelli che supportano function calling (GPT-4, Claude) possono usare tool functions per:
- Analisi video (`analyze_video`)
- Estrazione dati documenti (`extract_document_data`)
- Stima età (`estimate_age`)
- Descrizione ambiente (`describe_environment`)

### 3.5 Text-to-Speech (`agent/tts/`)

Il sistema supporta 6 TTS engines diversi, tutti implementati come sottoclassi di `BaseTTS`.

#### 3.5.1 Base TTS Interface (`agent/tts/base.py`)

**Classe Astratta**: `BaseTTS`

**Interfaccia**:
```python
class BaseTTS(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """Sintesi sincrona"""
    
    @abstractmethod
    async def synthesize_async(self, text: str) -> TTSResult:
        """Sintesi asincrona"""
    
    @abstractmethod
    def get_available_voices(self) -> list[dict]:
        """Lista voci disponibili"""
```

**Factory Function**: `get_tts_engine(engine_name, **kwargs)` in [`agent/tts/__init__.py`](agent/tts/__init__.py)

#### 3.5.2 Piper TTS (`agent/tts/piper_tts.py`)

**Caratteristiche**:
- ✅ Self-hosted
- Veloce e leggero
- Ottimo supporto italiano
- Modelli ONNX (efficienti)

**Configurazione**:
```python
tts = PiperTTS(
    model="it_IT-riccardo-x_low",
    speaker=0
)
```

**Voci**: Dipendono dal modello scaricato

#### 3.5.3 Coqui TTS (`agent/tts/coqui_tts.py`)

**Caratteristiche**:
- ✅ Self-hosted
- Alta qualità neurale
- Richiede più risorse
- Supporto multilingua

**Configurazione**:
```python
tts = CoquiTTS(
    model="tts_models/it/mai_female/glow-tts"
)
```

#### 3.5.4 Edge TTS (`agent/tts/edge_tts_engine.py`)

**Caratteristiche**:
- ❌ Cloud-based (Microsoft Edge TTS)
- Qualità eccellente
- Veloce
- Multilingua (100+ voci)
- **Fallback di default** se altri engine non disponibili

**Configurazione**:
```python
tts = EdgeTTS(
    voice="it-IT-DiegoNeural"
)
```

**Voci Popolari**:
- `it-IT-DiegoNeural` (Italiano maschile)
- `it-IT-ElsaNeural` (Italiano femminile)
- `en-US-GuyNeural` (Inglese maschile)
- `en-US-AriaNeural` (Inglese femminile)

#### 3.5.5 Kokoro TTS (`agent/tts/kokoro_tts.py`)

**Caratteristiche**:
- ✅ Self-hosted
- Multilingua (82M parametri)
- Alta qualità
- Supporto speed control

**Configurazione**:
```python
tts = KokoroTTS(
    voice="it_sara",
    speed=1.0
)
```

#### 3.5.6 VibeVoice TTS (`agent/tts/vibevoice_tts.py`)

**Caratteristiche**:
- ✅ Self-hosted (Microsoft)
- Streaming real-time (~300ms latenza)
- Multi-speaker (4 voci)
- Espressivo e naturale
- **Richiede GPU** per performance ottimali

**Modelli**:
- `realtime`: Bassa latenza (0.5B parametri)
- `longform`: Alta qualità (1.6B parametri)

**Configurazione**:
```python
tts = VibeVoiceTTS(
    model="realtime",
    language="it",
    speaker="speaker_1",
    speed=1.0
)
```

**TTS Server**: VibeVoice viene eseguito su server esterno (`tts_server.py`) per accesso GPU/MPS.

#### 3.5.7 Chatterbox TTS (`agent/tts/chatterbox_tts.py`)

**Caratteristiche**:
- ✅ Self-hosted (Resemble AI)
- State-of-the-art quality
- Voice cloning support
- Emotion control
- 23 lingue supportate

**Modelli**:
- `standard`: Qualità standard
- `multilingual`: Multilingua
- `turbo`: Veloce

**Configurazione**:
```python
tts = ChatterboxTTS(
    model="multilingual",
    language="it",
    device="auto",  # auto, cuda, cpu, mps
    exaggeration=0.5,  # 0.0-1.0
    audio_prompt_path="/path/to/voice.wav"  # Voice cloning
)
```

**TTS Server**: Chatterbox viene eseguito su server esterno per accesso GPU/MPS.

#### 3.5.8 TTS Server Esterno (`tts_server.py`)

Server FastAPI dedicato che gira sul host (non in Docker) per:
- Accesso a GPU/MPS (Apple Silicon)
- Esecuzione TTS pesanti (VibeVoice, Chatterbox, Piper, Kokoro)
- API REST per sintesi

**Endpoints**:
- `GET /health`: Health check
- `GET /status`: Stato TTS server
- `POST /synthesize`: Sintetizza testo (ritorna PCM 16-bit 24kHz)

**File**: [`tts_server.py`](tts_server.py)

### 3.6 Database (`db/`)

#### 3.6.1 PostgreSQL Schema

Lo schema è definito in [`db/schema.sql`](db/schema.sql).

**Tabelle**:

1. **`settings`**: Key-value store per configurazioni
   ```sql
   CREATE TABLE settings (
       key VARCHAR(255) PRIMARY KEY,
       value TEXT NOT NULL,
       updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
   );
   ```

2. **`chats`**: Sessioni conversazionali
   ```sql
   CREATE TABLE chats (
       id SERIAL PRIMARY KEY,
       title VARCHAR(255) NOT NULL DEFAULT 'Nuova Chat',
       created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
   );
   ```

3. **`messages`**: Messaggi individuali
   ```sql
   CREATE TABLE messages (
       id SERIAL PRIMARY KEY,
       chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
       role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
       content TEXT NOT NULL,
       created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
   );
   ```

**Indici**:
- `idx_messages_chat_id`: Per query veloci per chat
- `idx_messages_created_at`: Per ordinamento temporale

#### 3.6.2 Database Service (`db/database.py`)

**Classe**: `DatabaseService`

**Caratteristiche**:
- Connection pooling asincrono (min 2, max 10)
- Operazioni asincrone con `asyncpg`
- Gestione automatica connessioni

**Metodi Principali**:

**Settings**:
- `get_setting(key)`: Recupera singola impostazione
- `set_setting(key, value)`: Imposta impostazione (upsert)
- `get_all_settings()`: Tutte le impostazioni
- `set_multiple_settings(settings)`: Aggiorna multiple

**Chats**:
- `get_chats()`: Lista tutte le chat
- `create_chat(title)`: Crea nuova chat
- `get_chat(chat_id)`: Dettagli chat
- `update_chat_title(chat_id, title)`: Aggiorna titolo
- `delete_chat(chat_id)`: Elimina chat

**Messages**:
- `get_messages(chat_id, limit)`: Messaggi di una chat
- `add_message(chat_id, role, content)`: Aggiungi messaggio
- `delete_message(message_id)`: Elimina messaggio

**Global Instance**:
```python
async def get_db() -> DatabaseService:
    """Get or create the global database instance."""
    # Singleton pattern
```

### 3.7 Configuration (`agent/config.py`)

Il modulo di configurazione centralizza tutte le impostazioni.

#### 3.7.1 Config Classes

**`AppConfig`**: Configurazione principale
```python
@dataclass
class AppConfig:
    livekit: LiveKitConfig
    ollama: OllamaConfig
    whisper: WhisperConfig
    tts: TTSConfig
    vision: VisionConfig
    server: ServerConfig
```

**`LiveKitConfig`**: Configurazione LiveKit
- `url`: URL WebSocket LiveKit
- `api_key`: API key
- `api_secret`: API secret

**`OllamaConfig`**: Configurazione Ollama
- `host`: URL server Ollama
- `model`: Nome modello

**`WhisperConfig`**: Configurazione Whisper
- `model`: Dimensione modello (tiny, base, small, medium, large)
- `language`: Lingua (it, en, ecc.)
- `device`: Device (cpu, cuda)

**`TTSConfig`**: Configurazione TTS
- `default_engine`: Engine di default
- Parametri specifici per ogni engine (piper, coqui, edge, kokoro, vibevoice, chatterbox)

**`VisionConfig`**: Configurazione Vision
- `enabled`: Abilita/disabilita analisi video
- `openrouter_vision_model`: Modello vision per OpenRouter
- `ollama_vision_model`: Modello vision per Ollama
- `max_frame_rate`: Rate limiting estrazione frame

**`ServerConfig`**: Configurazione Server
- `web_port`: Porta web server
- `log_level`: Livello logging

#### 3.7.2 Environment Variables

Tutte le configurazioni sono leggibili da variabili d'ambiente (con `.env` file support):

```bash
# LiveKit
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gpt-oss

# Whisper
WHISPER_MODEL=base
WHISPER_LANGUAGE=it
WHISPER_DEVICE=cpu

# TTS
DEFAULT_TTS=edge
PIPER_MODEL=it_IT-riccardo-x_low
EDGE_VOICE=it-IT-DiegoNeural
# ... altri parametri TTS

# Server
WEB_PORT=8080
LOG_LEVEL=INFO
```

**File**: [`env.example`](env.example)

#### 3.7.3 Global Config Instance

```python
# Istanza globale della configurazione
config = AppConfig()

def reload_config() -> AppConfig:
    """Ricarica la configurazione dall'ambiente"""
    global config
    load_dotenv()
    config = AppConfig()
    return config
```

---

## 4. Flussi di Dati

### 4.1 Flusso Completo: Browser → LiveKit → Agent → STT → LLM → TTS → LiveKit → Browser

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant WebServer
    participant LiveKit
    participant Agent
    participant STT
    participant LLM
    participant TTS
    participant DB
    
    Note over User,Browser: 1. Inizializzazione
    User->>Browser: Apri pagina web
    Browser->>WebServer: GET /
    WebServer-->>Browser: HTML/JS
    
    Browser->>WebServer: POST /api/token
    WebServer->>LiveKit: Create Room
    WebServer->>LiveKit: Dispatch Agent
    WebServer-->>Browser: Token + URL
    
    Note over User,Browser: 2. Connessione WebRTC
    Browser->>LiveKit: WebRTC Connect (token)
    LiveKit->>Agent: Agent Connected
    Agent->>DB: Load Settings
    DB-->>Agent: Settings (LLM, TTS, ecc.)
    
    Note over User,Browser: 3. Input Vocale
    User->>Browser: Parla (microfono)
    Browser->>LiveKit: Audio Stream (WebRTC)
    LiveKit->>Agent: Audio Frame
    
    Note over User,Browser: 4. Speech-to-Text
    Agent->>STT: Transcribe Audio
    STT->>STT: Whisper Processing
    STT-->>Agent: Text + Language
    
    Note over User,Browser: 5. Trigger Detection
    Agent->>Agent: Check Trigger (@sophyai)
    alt Single User o Force Mode
        Agent->>Agent: Auto-respond
    else Multi User senza Trigger
        Agent->>Agent: Ignore
    end
    
    Note over User,Browser: 6. LLM Generation
    Agent->>DB: Get Conversation History
    DB-->>Agent: History
    Agent->>LLM: Generate Response (con context)
    LLM->>LLM: Processing
    LLM-->>Agent: Response Text (streaming)
    
    Note over User,Browser: 7. Text-to-Speech
    Agent->>TTS: Synthesize Text
    TTS->>TTS: Audio Generation
    TTS-->>Agent: Audio Data (PCM)
    
    Note over User,Browser: 8. Output Vocale
    Agent->>LiveKit: Audio Stream
    LiveKit->>Browser: Audio Playback (WebRTC)
    Browser->>User: Audio (altoparlanti)
    
    Note over User,Browser: 9. Persistenza
    Agent->>DB: Save Message (user)
    Agent->>DB: Save Message (assistant)
```

### 4.2 Flusso Multi-user

```mermaid
sequenceDiagram
    participant User1
    participant User2
    participant Browser1
    participant Browser2
    participant LiveKit
    participant Agent
    
    User1->>Browser1: Connessione
    Browser1->>LiveKit: Connect (token1)
    LiveKit->>Agent: User1 Connected
    
    User2->>Browser2: Connessione
    Browser2->>LiveKit: Connect (token2)
    LiveKit->>Agent: User2 Connected
    
    Note over User1,Agent: User1 parla
    User1->>Browser1: Parla
    Browser1->>LiveKit: Audio Stream
    LiveKit->>Agent: Audio (User1)
    Agent->>Agent: STT: "Ciao a tutti"
    Agent->>Agent: Check: No @sophyai trigger
    Agent->>Agent: Ignore (multi-user mode)
    
    Note over User2,Agent: User2 menziona agent
    User2->>Browser2: Parla "@sophyai come stai?"
    Browser2->>LiveKit: Audio Stream
    LiveKit->>Agent: Audio (User2)
    Agent->>Agent: STT: "@sophyai come stai?"
    Agent->>Agent: Check: @sophyai found
    Agent->>Agent: Process & Respond
    Agent->>LiveKit: Audio Response
    LiveKit->>Browser1: Audio (User1 ascolta)
    LiveKit->>Browser2: Audio (User2 ascolta)
```

### 4.3 Flusso Configurazione Dinamica

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant WebServer
    participant DB
    participant Agent
    
    User->>Browser: Cambia TTS engine
    Browser->>WebServer: POST /api/tts/current
    WebServer->>DB: Save TTS Config
    DB-->>WebServer: OK
    WebServer->>WebServer: Save to /app/config/tts_config.json
    WebServer-->>Browser: OK
    
    Note over Agent: Agent rileva cambio config
    Agent->>Agent: Check /app/config/tts_config.json
    Agent->>Agent: Reload TTS Engine
    Agent->>Agent: New TTS Ready
    
    User->>Browser: Prossima richiesta
    Browser->>Agent: Audio Input
    Agent->>Agent: Use New TTS Engine
```

### 4.4 Flusso Persistenza Chat

```mermaid
sequenceDiagram
    participant Agent
    participant DB
    participant WebServer
    participant Browser
    
    Note over Agent,DB: Durante Conversazione
    Agent->>DB: add_message(chat_id, "user", text)
    DB->>DB: INSERT message
    DB->>DB: UPDATE chat.updated_at
    DB-->>Agent: message_id
    
    Agent->>DB: add_message(chat_id, "assistant", response)
    DB->>DB: INSERT message
    DB-->>Agent: message_id
    
    Note over Browser,DB: Visualizzazione Chat
    Browser->>WebServer: GET /api/chats
    WebServer->>DB: get_chats()
    DB-->>WebServer: Lista chat
    WebServer-->>Browser: JSON
    
    Browser->>WebServer: GET /api/chats/{id}
    WebServer->>DB: get_chat(id)
    WebServer->>DB: get_messages(id)
    DB-->>WebServer: Chat + Messages
    WebServer-->>Browser: JSON
```

---

## 5. Deployment e Infrastruttura

### 5.1 Docker Compose Setup

Il sistema è completamente containerizzato tramite Docker Compose.

**File**: [`docker-compose.yml`](docker-compose.yml)

#### 5.1.1 Servizi Containerizzati

```mermaid
graph TB
    subgraph Docker["Docker Network: voiceagent"]
        Postgres[PostgreSQL<br/>voice-agent-db<br/>Port: 5432]
        Redis[Redis<br/>livekit-redis<br/>Port: 6379]
        Web[Web Server<br/>voice-agent-web<br/>Ports: 8080, 8443]
        Agent[Voice Agent<br/>voice-agent-worker]
        SIP[SIP Bridge<br/>livekit-sip<br/>Ports: 5060, 10000-10100]
    end
    
    subgraph Host["Host Machine"]
        Ollama[Ollama<br/>localhost:11434]
        TTS_Server[TTS Server<br/>localhost:8092]
        LiveKit[LiveKit<br/>localhost:7880]
    end
    
    Web --> Postgres
    Agent --> Postgres
    Agent --> LiveKit
    Web --> LiveKit
    Agent --> Ollama
    Agent --> TTS_Server
    LiveKit --> Redis
```

#### 5.1.2 PostgreSQL Service

```yaml
postgres:
  image: postgres:16-alpine
  container_name: voice-agent-db
  environment:
    - POSTGRES_DB=voiceagent
    - POSTGRES_USER=voiceagent
    - POSTGRES_PASSWORD=voiceagent_pwd
  volumes:
    - postgres_data:/var/lib/postgresql/data
    - ./db/schema.sql:/docker-entrypoint-initdb.d/schema.sql:ro
  ports:
    - "5432:5432"
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U voiceagent -d voiceagent"]
```

**Volumi**:
- `postgres_data`: Dati persistenti

**Inizializzazione**: Lo schema SQL viene eseguito automaticamente al primo avvio.

#### 5.1.3 Redis Service

```yaml
redis:
  image: redis:7-alpine
  container_name: livekit-redis
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
```

**Uso**: Cache per LiveKit (session state, room metadata).

#### 5.1.4 Web Server Service

```yaml
web:
  build:
    context: .
    dockerfile: Dockerfile
  container_name: voice-agent-web
  ports:
    - "8080:8080"     # HTTP
    - "8443:8443"     # HTTPS
  volumes:
    - ./models:/app/models
    - ./certs:/app/certs:ro
    - ./config:/app/config
    - ./web:/app/web:ro
    - ./server.py:/app/server.py:ro
    - ./db:/app/db:ro
  depends_on:
    postgres:
      condition: service_healthy
```

**Volumi Montati**:
- `./models`: Modelli ML (Whisper, TTS)
- `./certs`: Certificati SSL
- `./config`: Configurazioni (es. `tts_config.json`)
- `./web`: Frontend files
- `./server.py`: Server code (hot-reload)
- `./db`: Database module

#### 5.1.5 Agent Service

```yaml
agent:
  build:
    context: .
    dockerfile: Dockerfile.agent
  container_name: voice-agent-worker
  environment:
    - LIVEKIT_URL=ws://host.docker.internal:7880
    - TTS_SERVER_URL=http://host.docker.internal:8092
    - WHISPER_SERVER_URL=http://host.docker.internal:8091
  volumes:
    - ./models:/app/models
    - ./config:/app/config
    - ./agent:/app/agent:ro
  depends_on:
    postgres:
      condition: service_healthy
```

**Note**:
- Si connette a LiveKit su host (`host.docker.internal`)
- Usa TTS Server esterno per GPU access
- Condivide `./config` con web server per configurazione TTS

#### 5.1.6 SIP Bridge Service (Opzionale)

```yaml
sip:
  image: livekit/sip:latest
  container_name: livekit-sip
  ports:
    - "5060:5060/udp"   # SIP UDP
    - "5060:5060/tcp"   # SIP TCP
    - "5061:5061/tcp"   # SIP TLS
    - "10000-10100:10000-10100/udp"  # RTP media
```

**Uso**: Permette chiamate telefoniche tradizionali (SIP) al sistema.

### 5.2 Network Configuration

**Network**: `voiceagent` (bridge)

**Extra Hosts**: Tutti i container hanno accesso a `host.docker.internal` per:
- Ollama (localhost:11434)
- TTS Server (localhost:8092)
- Whisper Server (localhost:8091)
- LiveKit (localhost:7880)

### 5.3 Volume Mounts

| Volume | Container | Path | Descrizione |
|--------|-----------|------|-------------|
| `postgres_data` | postgres | `/var/lib/postgresql/data` | Dati PostgreSQL |
| `redis_data` | redis | `/data` | Dati Redis |
| `./models` | web, agent | `/app/models` | Modelli ML condivisi |
| `./config` | web, agent | `/app/config` | Configurazioni condivise |
| `./certs` | web | `/app/certs` | Certificati SSL |
| `./web` | web | `/app/web` | Frontend files |

### 5.4 Environment Variables

Le variabili d'ambiente sono configurabili in `.env` o direttamente in `docker-compose.yml`.

**Variabili Principali**:

```bash
# LiveKit
LIVEKIT_URL=ws://0.0.0.0:7880
LIVEKIT_INTERNAL_URL=ws://host.docker.internal:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret_dev_key_change_in_production

# Ollama
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=devstral-small-2

# Whisper
WHISPER_MODEL=small
WHISPER_LANGUAGE=it
WHISPER_DEVICE=cpu

# TTS
DEFAULT_TTS=edge
TTS_SERVER_URL=http://host.docker.internal:8092

# Database
DATABASE_URL=postgresql://voiceagent:voiceagent_pwd@postgres:5432/voiceagent

# Server
WEB_PORT=8443
LOG_LEVEL=INFO
```

### 5.5 Avvio Sistema

#### Opzione 1: Docker Compose (Consigliato)

```bash
# Avvia tutti i servizi
docker-compose up -d

# Verifica status
docker-compose ps

# Logs
docker-compose logs -f agent
docker-compose logs -f web
```

#### Opzione 2: Script Python

```bash
python run.py
```

**File**: [`run.py`](run.py)

#### Opzione 3: Script Shell

```bash
./start_all.sh
```

**File**: [`start_all.sh`](start_all.sh)

### 5.6 LiveKit Server

LiveKit deve essere avviato separatamente sul host (non in Docker):

```bash
# Avvia LiveKit
livekit-server --config livekit-local.yaml
```

**Configurazione**: [`livekit-local.yaml`](livekit-local.yaml)

**Porte**:
- `7880`: WebSocket (HTTP)
- `7443`: WebSocket (HTTPS/TLS)

---

## 6. API Reference

### 6.1 REST API Endpoints

Base URL: `https://localhost:8443` (HTTPS) o `http://localhost:8080` (HTTP)

#### 6.1.1 Autenticazione

##### `POST /api/token`

Genera token LiveKit per partecipante.

**Request**:
```json
{
  "room_name": "default",
  "participant_name": "user123"
}
```

**Response**:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "url": "ws://localhost:7880"
}
```

**Errori**:
- `409`: Nome utente già in uso nella room
- `500`: Errore generazione token

#### 6.1.2 Configurazione

##### `GET /api/config`

Configurazione pubblica.

**Response**:
```json
{
  "livekit_url": "ws://localhost:7880",
  "livekit_url_configured": "ws://0.0.0.0:7880",
  "server_ip": "localhost",
  "default_tts": "edge",
  "whisper_model": "small",
  "ollama_model": "devstral-small-2"
}
```

##### `GET /api/settings`

Tutte le impostazioni.

**Response**:
```json
{
  "llm_provider": "ollama",
  "ollama_model": "devstral-small-2",
  "openrouter_model": "",
  "openrouter_api_key": "",
  "whisper_model": "small",
  "whisper_language": "it",
  "tts_engine": "edge",
  "system_prompt": "...",
  "context_injection": ""
}
```

##### `POST /api/settings`

Aggiorna multiple impostazioni.

**Request**:
```json
{
  "settings": {
    "tts_engine": "vibevoice",
    "whisper_model": "medium"
  }
}
```

**Response**:
```json
{
  "status": "ok",
  "updated": ["tts_engine", "whisper_model"]
}
```

#### 6.1.3 LLM Management

##### `GET /api/ollama/models`

Lista modelli Ollama disponibili.

**Response**:
```json
{
  "models": [
    {
      "id": "gpt-oss",
      "name": "gpt-oss",
      "size": 1234567890,
      "modified_at": "2024-01-01T00:00:00Z"
    }
  ],
  "host": "http://localhost:11434"
}
```

##### `POST /api/ollama/select`

Seleziona modello Ollama.

**Request**:
```json
{
  "model": "gpt-oss"
}
```

**Response**:
```json
{
  "status": "ok",
  "model": "gpt-oss",
  "provider": "ollama"
}
```

##### `GET /api/openrouter/models`

Lista modelli OpenRouter (con filtri opzionali).

**Query Parameters**:
- `search`: Filtro ricerca
- `sort_by`: Ordinamento (name, cost, cost_desc, context, vision, tools)

**Response**:
```json
{
  "models": [
    {
      "id": "openai/gpt-4-turbo",
      "name": "GPT-4 Turbo",
      "description": "...",
      "context_length": 128000,
      "prompt_cost": 10.0,
      "completion_cost": 30.0,
      "total_cost": 40.0,
      "supports_vision": false,
      "supports_tools": true,
      "supports_json_mode": true
    }
  ],
  "count": 150
}
```

##### `POST /api/openrouter/select`

Seleziona modello OpenRouter.

**Request**:
```json
{
  "model": "openai/gpt-4-turbo",
  "name": "GPT-4 Turbo",
  "context_length": 128000,
  "supports_vision": false,
  "supports_tools": true
}
```

#### 6.1.4 TTS Management

##### `GET /api/tts/engines`

Lista TTS engines disponibili.

**Response**:
```json
{
  "engines": [
    {
      "id": "edge",
      "name": "Edge TTS (Microsoft)",
      "self_hosted": false,
      "description": "Qualità eccellente, richiede internet"
    },
    {
      "id": "vibevoice",
      "name": "VibeVoice (Microsoft)",
      "self_hosted": true,
      "description": "Espressivo, multi-speaker, real-time streaming"
    }
  ],
  "default": "edge"
}
```

##### `GET /api/tts/{engine}/voices`

Voci disponibili per engine.

**Response**:
```json
{
  "engine": "edge",
  "voices": [
    {
      "id": "it-IT-DiegoNeural",
      "name": "Diego (Italiano)",
      "gender": "male",
      "locale": "it-IT"
    }
  ]
}
```

##### `POST /api/tts/current`

Imposta TTS engine corrente.

**Request**:
```json
{
  "engine": "vibevoice",
  "language": "it",
  "voice": "speaker_1",
  "speaker": "speaker_1",
  "speed": 1.0
}
```

**Response**:
```json
{
  "status": "ok",
  "engine": "vibevoice",
  "language": "it",
  "saved_to_file": true,
  "message": "TTS vibevoice configurato. Riavvia l'agent per applicare."
}
```

##### `POST /api/tts/test`

Test sintesi TTS.

**Request**:
```json
{
  "engine": "edge",
  "text": "Ciao, questo è un test",
  "language": "it",
  "voice": "it-IT-DiegoNeural"
}
```

**Response**: Audio WAV (binary)

**Headers**:
- `X-Sample-Rate`: Sample rate (es. 24000)
- `X-Duration`: Durata audio in secondi
- `X-Engine`: Engine usato

#### 6.1.5 Chat Management

##### `GET /api/chats`

Lista tutte le chat.

**Response**:
```json
{
  "chats": [
    {
      "id": 1,
      "title": "Conversazione 1",
      "created_at": "2024-01-01T00:00:00Z",
      "updated_at": "2024-01-01T01:00:00Z",
      "message_count": 10
    }
  ]
}
```

##### `POST /api/chats`

Crea nuova chat.

**Request**:
```json
{
  "title": "Nuova Chat"
}
```

**Response**:
```json
{
  "status": "ok",
  "id": 1,
  "title": "Nuova Chat"
}
```

##### `GET /api/chats/{id}`

Dettagli chat con messaggi.

**Response**:
```json
{
  "id": 1,
  "title": "Conversazione 1",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T01:00:00Z",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "Ciao",
      "created_at": "2024-01-01T00:00:00Z"
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "Ciao! Come posso aiutarti?",
      "created_at": "2024-01-01T00:00:01Z"
    }
  ]
}
```

#### 6.1.6 System Prompt

##### `GET /api/prompt`

Prompt di sistema corrente.

**Response**:
```json
{
  "prompt": "Sei Sophy, assistente vocale..."
}
```

##### `POST /api/prompt`

Aggiorna prompt di sistema.

**Request**:
```json
{
  "prompt": "Nuovo prompt di sistema..."
}
```

#### 6.1.7 Status e Health

##### `GET /api/health`

Health check semplice.

**Response**:
```json
{
  "status": "ok",
  "service": "voice-agent",
  "https": true
}
```

##### `GET /api/status`

Status dettagliato di tutti i servizi.

**Response**:
```json
{
  "livekit": {
    "available": true,
    "message": "Connesso (2 room attive)"
  },
  "ollama": {
    "available": true,
    "message": "Modello devstral-small-2 disponibile"
  },
  "agent": {
    "available": true,
    "message": "Worker attivo e in ascolto"
  },
  "whisper": {
    "available": true,
    "message": "Modello 'small' scaricato e pronto"
  },
  "sip": {
    "available": false,
    "message": "SIP Bridge non avviato (opzionale)"
  },
  "all_ready": true
}
```

##### `GET /api/timing`

Statistiche timing.

**Response**:
```json
{
  "stt": {
    "time_ms": 250,
    "count": 100
  },
  "llm": {
    "time_ms": 1200,
    "ttft_ms": 300,
    "count": 100
  },
  "tts": {
    "time_ms": 500,
    "audio_sec": 2.5,
    "count": 100
  },
  "latency": {
    "e2e_ms": 1950,
    "to_first_audio_ms": 800,
    "count": 100
  }
}
```

### 6.2 Request/Response Formats

#### 6.2.1 Content-Type

- **Request**: `application/json` per POST/PUT
- **Response**: `application/json` (tranne `/api/tts/test` che ritorna `audio/wav`)

#### 6.2.2 Error Format

```json
{
  "detail": "Messaggio di errore"
}
```

**Status Codes**:
- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `409`: Conflict (es. nome utente duplicato)
- `500`: Internal Server Error
- `503`: Service Unavailable

### 6.3 Authentication

Attualmente non è richiesta autenticazione per le API (sviluppo). In produzione:
- Implementare JWT tokens
- Rate limiting
- CORS più restrittivo

---

## 7. Configurazione Avanzata

### 7.1 TTS Engine Selection

#### 7.1.1 Configurazione via API

```bash
# Cambia TTS engine
curl -X POST https://localhost:8443/api/tts/current \
  -H "Content-Type: application/json" \
  -d '{
    "engine": "vibevoice",
    "language": "it",
    "speaker": "speaker_1",
    "speed": 1.0
  }'
```

**Nota**: Dopo il cambio, riavviare l'agent per applicare:
```bash
docker-compose restart agent
```

#### 7.1.2 Configurazione via Environment

```bash
# .env
DEFAULT_TTS=vibevoice
VIBEVOICE_MODEL=realtime
VIBEVOICE_LANGUAGE=it
VIBEVOICE_SPEAKER=speaker_1
VIBEVOICE_SPEED=1.0
VIBEVOICE_GPU=true
```

#### 7.1.3 File di Configurazione

Il TTS viene salvato anche in `/app/config/tts_config.json` (condiviso tra web server e agent):

```json
{
  "engine": "vibevoice",
  "language": "it",
  "voice": "speaker_1",
  "speaker": "speaker_1",
  "speed": 1.0,
  "updated_at": "2024-01-01T00:00:00Z"
}
```

### 7.2 LLM Provider Configuration

#### 7.2.1 Ollama Setup

1. **Installa Ollama**: https://ollama.ai
2. **Scarica Modello**:
   ```bash
   ollama pull gpt-oss
   # o
   ollama pull devstral-small-2
   ```
3. **Configura**:
   ```bash
   # .env
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_MODEL=gpt-oss
   ```
4. **Verifica**:
   ```bash
   curl http://localhost:11434/api/tags
   ```

#### 7.2.2 OpenRouter Setup

1. **Registrati**: https://openrouter.ai
2. **Crea API Key**
3. **Configura**:
   ```bash
   # .env
   OPENROUTER_API_KEY=sk-or-xxx...
   OPENROUTER_MODEL=openai/gpt-4-turbo
   ```
4. **Salva via API**:
   ```bash
   curl -X POST https://localhost:8443/api/openrouter/key \
     -H "Content-Type: application/json" \
     -d '{"api_key": "sk-or-xxx..."}'
   ```

#### 7.2.3 Switch Provider

```bash
# Via API
curl -X POST https://localhost:8443/api/settings \
  -H "Content-Type: application/json" \
  -d '{
    "settings": {
      "llm_provider": "openrouter",
      "openrouter_model": "openai/gpt-4-turbo"
    }
  }'
```

**Nota**: Riavviare l'agent dopo il cambio.

### 7.3 Whisper Model Selection

#### 7.3.1 Modelli Disponibili

| Modello | Use Case | RAM | GPU |
|---------|----------|-----|-----|
| `tiny` | Sviluppo/test | 1GB | No |
| `base` | Uso generale | 1GB | Opzionale |
| `small` | Qualità buona | 2GB | Consigliato |
| `medium` | Alta qualità | 5GB | Richiesto |
| `large` | Massima qualità | 10GB | Richiesto |

#### 7.3.2 Configurazione

```bash
# .env
WHISPER_MODEL=small
WHISPER_LANGUAGE=it
WHISPER_DEVICE=cuda  # o cpu
```

#### 7.3.3 Download Modelli

I modelli vengono scaricati automaticamente al primo utilizzo in `~/.cache/huggingface/hub/` o `HF_HOME`.

**Manuale**:
```python
from faster_whisper import WhisperModel
model = WhisperModel("small", device="cpu")
# Scarica automaticamente
```

### 7.4 Database Setup

#### 7.4.1 Inizializzazione Automatica

Il database viene inizializzato automaticamente al primo avvio del container PostgreSQL tramite `db/schema.sql`.

#### 7.4.2 Backup

```bash
# Backup
docker exec voice-agent-db pg_dump -U voiceagent voiceagent > backup.sql

# Restore
docker exec -i voice-agent-db psql -U voiceagent voiceagent < backup.sql
```

#### 7.4.3 Migrazioni

Per modifiche schema, modificare `db/schema.sql` e ricreare il container:

```bash
docker-compose down
docker volume rm sophyai-live-server_postgres_data
docker-compose up -d
```

### 7.5 SSL/HTTPS Setup

#### 7.5.1 Generazione Certificati Self-Signed

```bash
# Crea directory
mkdir -p certs

# Genera certificato
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes

# Durante la generazione, inserisci:
# Common Name: localhost
```

#### 7.5.2 Certificati Let's Encrypt (Produzione)

```bash
# Usa certbot
certbot certonly --standalone -d yourdomain.com

# Copia certificati
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem certs/key.pem
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem certs/cert.pem
```

#### 7.5.3 Configurazione

I certificati vengono automaticamente rilevati se presenti in `certs/`:
- `certs/key.pem`: Chiave privata
- `certs/cert.pem`: Certificato

**Nota**: I browser mostreranno un warning per certificati self-signed. Accetta l'eccezione per sviluppo.

### 7.6 Performance Tuning

#### 7.6.1 Whisper Ottimizzazioni

```bash
# CPU
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# GPU NVIDIA
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

#### 7.6.2 TTS Ottimizzazioni

**VibeVoice/Chatterbox**: Richiedono GPU per performance ottimali.

**Edge TTS**: Nessuna ottimizzazione necessaria (cloud).

**Piper**: CPU è sufficiente.

#### 7.6.3 LLM Ottimizzazioni

**Ollama**: Usa quantizzazione automatica. Per modelli grandi:
```bash
# Usa quantizzazione
ollama pull llama2:7b-q4_0
```

**OpenRouter**: Nessuna ottimizzazione lato client.

### 7.7 Multi-User Configuration

#### 7.7.1 Mention-based Activation

In multi-user mode (2+ utenti), l'agent risponde solo se menzionato:

**Trigger**:
- `@sophyai`
- `@sophy`
- `sophy ai`
- `sophy` (parola intera)

#### 7.7.2 Force Response Mode

Per forzare risposta sempre (anche in multi-user):

```javascript
// Frontend
fetch('/api/settings', {
  method: 'POST',
  body: JSON.stringify({
    settings: { force_agent_response: 'true' }
  })
});
```

### 7.8 Video Analysis Configuration

#### 7.8.1 Abilitazione

```bash
# .env
VIDEO_ANALYSIS_ENABLED=true
OPENROUTER_VISION_MODEL=openai/gpt-4-vision-preview
OLLAMA_VISION_MODEL=llava
VIDEO_MAX_FRAME_RATE=1.0
```

#### 7.8.2 Modelli Vision

**OpenRouter**:
- `openai/gpt-4-vision-preview`: Alta qualità
- `google/gemini-pro-vision`: Alternativa

**Ollama**:
- `llava`: Modello vision locale

#### 7.8.3 Rate Limiting

`VIDEO_MAX_FRAME_RATE` limita l'estrazione frame (default 1.0 fps) per ridurre costi API.

---

## Conclusioni

Questa documentazione copre l'architettura completa del sistema SophyAI Live Server, dai componenti principali ai dettagli di configurazione avanzata. Il sistema è progettato per essere modulare, estendibile e completamente self-hosted (tranne per Edge TTS e OpenRouter che sono opzionali).

Per supporto o contributi, consultare il repository del progetto.

---

**Versione Documentazione**: 1.0  
**Ultimo Aggiornamento**: 2024-01-01  
**Autore**: Sistema di Documentazione Automatica
