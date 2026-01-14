# Presentazione Tecnica - SophyAI Live Server
## Sistema di Assistente Vocale Self-Hosted

---

## Slide 1: Titolo

# SophyAI Live Server
## Sistema di Assistente Vocale Self-Hosted con WebRTC

**Incontro Tecnico**  
*Architettura, Componenti e Funzionamento*

---

## Slide 2: Agenda

### Contenuti della Presentazione

1. **Panoramica del Sistema**
2. **Architettura e Componenti**
3. **Stack Tecnologico**
4. **Moduli Principali**
5. **Flussi di Dati**
6. **Deployment e Infrastruttura**
7. **Demo e Caratteristiche Avanzate**
8. **Q&A**

---

## Slide 3: Cos'è SophyAI Live Server?

### Sistema Completo di Assistente Vocale

- ✅ **Completamente Self-Hosted** (tranne servizi opzionali cloud)
- ✅ **WebRTC Real-Time** per comunicazioni vocali bidirezionali
- ✅ **Multi-Engine** per STT, LLM e TTS
- ✅ **Multi-User** support con gestione intelligente
- ✅ **Video Analysis** opzionale per analisi immagini/video
- ✅ **Persistenza** completa di conversazioni e configurazioni

### Use Cases
- Assistente vocale personale
- Call center automatizzato
- Supporto clienti vocale
- Interfaccia vocale per applicazioni

---

## Slide 4: Architettura ad Alto Livello

Il sistema SophyAI Live Server è basato su un'architettura modulare che separa chiaramente le responsabilità tra client, infrastruttura, applicazione e servizi AI. Il browser client comunica con LiveKit tramite WebRTC per lo streaming audio in tempo reale. LiveKit gestisce la comunicazione bidirezionale e dispatcha automaticamente il Voice Agent quando necessario. L'agent orchestra i tre componenti AI principali: Whisper per la trascrizione vocale, il provider LLM (Ollama locale o OpenRouter cloud) per la generazione di risposte, e i motori TTS per la sintesi vocale. Tutti i dati vengono persistiti in PostgreSQL per garantire continuità tra sessioni.

---

## Slide 5: Componenti Principali

### 5 Componenti Chiave

1. **Web Server (FastAPI)**
   - Frontend HTML/JS
   - API REST per configurazione
   - Token generation LiveKit
   - Database management

2. **Voice Agent (LiveKit Worker)**
   - Orchestrazione STT → LLM → TTS
   - Gestione sessioni
   - Multi-user support

3. **LiveKit Server**
   - WebRTC infrastructure
   - Media streaming real-time
   - Room management

4. **PostgreSQL Database**
   - Settings persistence
   - Chat history
   - Message storage

5. **Redis Cache**
   - LiveKit session state
   - Room metadata

---

## Slide 6: Stack Tecnologico

Lo stack tecnologico è stato scelto per bilanciare performance, facilità di sviluppo e scalabilità. Sul backend utilizziamo FastAPI per le API REST asincrone, LiveKit per la gestione WebRTC real-time, PostgreSQL per la persistenza robusta e Redis per la cache delle sessioni. Tutto è containerizzato con Docker per facilità di deployment. Per l'AI, abbiamo scelto faster-whisper per STT (ottimizzato rispetto a Whisper standard), Ollama per LLM locale con privacy totale, OpenRouter per accesso a modelli cloud avanzati, e sei diversi motori TTS per coprire tutti i casi d'uso. Il supporto vision opzionale permette analisi di immagini e video.

---

## Slide 7: Speech-to-Text (STT)

### Whisper STT Implementation

**Caratteristiche**:
- ✅ Modelli: tiny, base, small, medium, large
- ✅ VAD (Voice Activity Detection) integrato
- ✅ Language detection automatica
- ✅ Streaming transcription
- ✅ Supporto GPU (CUDA) e CPU

**Performance**:
- **tiny**: ~1GB RAM, molto veloce
- **small**: ~2GB RAM, qualità buona (consigliato)
- **medium**: ~5GB RAM, alta qualità
- **large**: ~10GB RAM, qualità eccellente

**Configurazione**:
```python
stt = WhisperSTT(
    model_size="small",
    language="it",
    device="cuda"  # o "cpu"
)
```

---

## Slide 8: Large Language Model (LLM)

### Due Provider Disponibili

#### 1. Ollama (Locale)
- ✅ **Privacy totale** - tutto locale
- ✅ Modelli: gpt-oss, llama2, mistral, ecc.
- ✅ Streaming responses
- ✅ Zero costi API

#### 2. OpenRouter (Cloud)
- ✅ **100+ modelli** disponibili
- ✅ GPT-4, Claude, Gemini, ecc.
- ✅ Vision models support
- ✅ Function calling

**Configurazione Dinamica**:
- Switch provider via API
- Modelli configurabili in runtime
- System prompt personalizzabile

---

## Slide 9: Text-to-Speech (TTS)

Il sistema supporta sei diversi motori TTS, ognuno con caratteristiche uniche. Edge TTS di Microsoft offre qualità eccellente via cloud senza bisogno di risorse locali. Piper è ideale per deployment leggeri con modelli ONNX efficienti. Coqui e Kokoro forniscono alta qualità neurale, con Kokoro particolarmente adatto per scenari multilingua. VibeVoice di Microsoft offre streaming real-time con latenza molto bassa, mentre Chatterbox permette voice cloning e controllo dell'emozione. Il cambio di engine può essere fatto in runtime senza riavviare il sistema, permettendo di adattare la voce alle esigenze specifiche.

---

## Slide 10: Flusso Completo End-to-End

Il flusso completo di una conversazione inizia quando l'utente parla nel microfono. L'audio viene catturato dal browser e inviato a LiveKit tramite WebRTC in tempo reale. LiveKit inoltra i frame audio al Voice Agent, che accumula l'audio fino a rilevare un silenzio. A questo punto, Whisper STT trascrive l'audio in testo con una latenza tipica di 250ms. Il testo viene poi inviato al LLM (Ollama o OpenRouter) che genera una risposta, con un tempo totale di circa 1200ms (di cui 300ms per il primo token). La risposta testuale viene sintetizzata in audio dal motore TTS configurato in circa 500ms. L'audio finale viene inviato a LiveKit e poi al browser per la riproduzione. La latenza end-to-end totale è tipicamente sotto i 2 secondi, garantendo una conversazione fluida e naturale.

---

## Slide 11: Multi-User Support

Il sistema gestisce intelligentemente le conversazioni multi-utente. Quando c'è un solo utente nella room, l'agent risponde automaticamente a ogni input vocale senza bisogno di trigger, creando un'esperienza fluida e naturale. Quando invece ci sono due o più utenti, l'agent entra in modalità "mention-based", rispondendo solo quando viene esplicitamente menzionato con trigger come @sophyai, @sophy o "sophy ai". Questo evita che l'agent interrompa conversazioni tra utenti umani. È disponibile anche una modalità "force response" che può essere attivata dal frontend per forzare l'agent a rispondere sempre, utile in scenari specifici come call center o assistenza clienti.

---

## Slide 12: Video Analysis (Opzionale)

### Analisi Immagini e Video

**Capacità**:
- ✅ Estrazione frame da webcam/screen sharing
- ✅ Analisi con modelli vision (GPT-4 Vision, LLaVA)
- ✅ Function calling per analisi specifiche

**Funzioni Disponibili**:
- `analyze_video()`: Analisi generica
- `extract_document_data()`: Estrazione dati documenti
- `estimate_age()`: Stima età
- `describe_environment()`: Descrizione ambiente

**Configurazione**:
```bash
VIDEO_ANALYSIS_ENABLED=true
OPENROUTER_VISION_MODEL=openai/gpt-4-vision-preview
VIDEO_MAX_FRAME_RATE=1.0  # Rate limiting
```

---

## Slide 13: Database Schema

Il database PostgreSQL utilizza uno schema semplice ma efficace composto da tre tabelle principali. La tabella `settings` funziona come un key-value store per tutte le configurazioni del sistema, permettendo aggiornamenti dinamici senza riavviare servizi. La tabella `chats` rappresenta le sessioni conversazionali, con timestamp automatici per tracciare creazione e ultimo aggiornamento. La tabella `messages` contiene tutti i messaggi delle conversazioni, con una foreign key verso chats che implementa cascade delete per garantire integrità referenziale. Gli indici su chat_id e created_at ottimizzano le query per recupero rapido della cronologia. Questo schema bilancia semplicità, performance e facilità di manutenzione.

---

## Slide 14: Deployment - Docker Compose

Il deployment utilizza Docker Compose per orchestrare tutti i servizi in modo isolato ma comunicante. All'interno della Docker network "voiceagent" troviamo PostgreSQL per la persistenza dati, Redis per la cache di LiveKit, il Web Server FastAPI che espone le API REST e il frontend, e il Voice Agent Worker che gestisce le conversazioni. Il SIP Bridge opzionale permette integrazione con telefonia tradizionale. Sul host machine girano invece Ollama per LLM locale, il TTS Server dedicato per accesso a GPU/MPS, e LiveKit Server per la gestione WebRTC. Questa separazione permette di sfruttare le risorse del host (GPU, MPS) mentre mantiene i servizi core containerizzati per facilità di deployment e scalabilità.

---

## Slide 15: API REST - Endpoints Principali

### Categorie di Endpoint

#### Autenticazione
- `POST /api/token` - Genera token LiveKit

#### Configurazione
- `GET/POST /api/settings` - Gestione impostazioni
- `GET /api/config` - Configurazione pubblica

#### LLM Management
- `GET /api/ollama/models` - Lista modelli Ollama
- `GET /api/openrouter/models` - Lista modelli OpenRouter (100+)
- `POST /api/ollama/select` - Seleziona modello
- `POST /api/openrouter/select` - Seleziona modello

#### TTS Management
- `GET /api/tts/engines` - Lista engines
- `GET /api/tts/{engine}/voices` - Voci disponibili
- `POST /api/tts/current` - Imposta TTS
- `POST /api/tts/test` - Test sintesi (ritorna audio)

#### Chat Management
- `GET/POST /api/chats` - Gestione chat
- `GET /api/chats/{id}` - Dettagli con messaggi

#### Status
- `GET /api/health` - Health check
- `GET /api/status` - Status tutti servizi
- `GET /api/timing` - Metriche performance

---

## Slide 16: Configurazione Avanzata

### Personalizzazione Sistema

#### TTS Engine Selection
```bash
# Via API
POST /api/tts/current
{
  "engine": "vibevoice",
  "language": "it",
  "speaker": "speaker_1",
  "speed": 1.0
}
```

#### LLM Provider Switch
```bash
# Cambia da Ollama a OpenRouter
POST /api/settings
{
  "settings": {
    "llm_provider": "openrouter",
    "openrouter_model": "openai/gpt-4-turbo"
  }
}
```

#### Whisper Model Tuning
```bash
# .env
WHISPER_MODEL=small      # tiny, base, small, medium, large
WHISPER_DEVICE=cuda      # cpu o cuda
WHISPER_LANGUAGE=it      # auto-detect se None
```

#### System Prompt
```bash
POST /api/prompt
{
  "prompt": "Sei un assistente specializzato in..."
}
```

---

## Slide 17: Performance e Ottimizzazioni

### Metriche e Tuning

**Metriche Tracciate**:
- STT Time: Tempo trascrizione
- LLM Time: Tempo generazione (incluso TTFT)
- TTS Time: Tempo sintesi
- E2E Latency: Latenza totale
- To First Audio: Tempo primo chunk audio

**Ottimizzazioni**:

**Whisper**:
- GPU: `device=cuda`, `compute_type=float16`
- CPU: `device=cpu`, `compute_type=int8`

**TTS**:
- VibeVoice/Chatterbox: Richiedono GPU
- Edge TTS: Nessuna ottimizzazione (cloud)
- Piper: CPU sufficiente

**LLM**:
- Ollama: Quantizzazione automatica
- OpenRouter: Nessuna ottimizzazione lato client

---

## Slide 18: Caratteristiche Avanzate

### Funzionalità Aggiuntive

#### 1. Video Analysis
- Estrazione frame da webcam/screen
- Analisi con GPT-4 Vision o LLaVA
- Function calling per analisi specifiche

#### 2. Voice Cloning (Chatterbox)
- Clona voce da file audio
- Emotion control
- Exaggeration control

#### 3. Multi-Language Support
- Auto-detection lingua (Whisper)
- TTS multilingua (Edge, Kokoro, Chatterbox)
- LLM risponde nella lingua dell'utente

#### 4. Conversation History
- Persistenza completa in PostgreSQL
- Context injection configurabile
- Chat management via API

#### 5. Real-Time Metrics
- Timing stats via API
- Service status monitoring
- Performance tracking

---

## Slide 19: Requisiti di Sistema

### Minimi vs Consigliati

#### Requisiti Minimi
- **CPU**: 4 core
- **RAM**: 8GB
- **Storage**: 20GB
- **Network**: Internet (per Edge TTS/OpenRouter)

#### Requisiti Consigliati
- **CPU**: 8+ core
- **RAM**: 16GB+ (32GB per TTS avanzati)
- **Storage**: 50GB+ (per modelli)
- **GPU**: NVIDIA con CUDA (opzionale, consigliato)
- **Network**: Connessione stabile

#### Software Richiesto
- Docker & Docker Compose
- Python 3.10+
- Ollama (per LLM locale)
- LiveKit Server

---

## Slide 20: Avvio e Deployment

### Quick Start

#### 1. Setup Iniziale
```bash
# Clona repository
git clone <repo>
cd sophyai-live-server

# Configura
cp env.example .env
nano .env  # Modifica configurazioni
```

#### 2. Avvio Servizi
```bash
# Docker Compose
docker-compose up -d

# Verifica
docker-compose ps
docker-compose logs -f agent
```

#### 3. Avvio LiveKit (Host)
```bash
livekit-server --config livekit-local.yaml
```

#### 4. Accesso
```
https://localhost:8443  # HTTPS
http://localhost:8080   # HTTP
```

#### 5. Verifica Status
```bash
curl https://localhost:8443/api/status
```

---

## Slide 21: Sicurezza e Best Practices

### Considerazioni di Sicurezza

#### Sviluppo (Attuale)
- ⚠️ Certificati SSL self-signed
- ⚠️ Nessuna autenticazione API
- ⚠️ CORS aperto a tutte le origini

#### Produzione (Raccomandato)
- ✅ Certificati Let's Encrypt o commerciali
- ✅ JWT authentication per API
- ✅ CORS restrittivo
- ✅ Rate limiting
- ✅ Firewall rules
- ✅ Secrets management (non in .env)
- ✅ Database encryption at rest

#### Best Practices
- Backup regolari database
- Monitoring e logging
- Update regolari dipendenze
- Security scanning

---

## Slide 22: Estendibilità

### Come Estendere il Sistema

#### 1. Nuovo TTS Engine
```python
# agent/tts/my_tts.py
class MyTTS(BaseTTS):
    def synthesize(self, text: str) -> TTSResult:
        # Implementazione
        pass
```

#### 2. Nuovo LLM Provider
```python
# agent/llm/my_llm.py
class MyLLM(llm.LLM):
    # Implementa interfaccia LiveKit
    pass
```

#### 3. Nuove API Endpoints
```python
# server.py
@app.post("/api/my-endpoint")
async def my_endpoint():
    # Implementazione
    pass
```

#### 4. Custom Function Tools
```python
# agent/main.py
@function_tool(description="...")
async def my_tool(context: RunContext) -> str:
    # Implementazione
    pass
```

---

## Slide 23: Roadmap e Sviluppi Futuri

### Possibili Evoluzioni

#### Breve Termine
- [ ] Autenticazione utenti
- [ ] Rate limiting
- [ ] Monitoring dashboard
- [ ] Export conversazioni (PDF, TXT)

#### Medio Termine
- [ ] Supporto più lingue native
- [ ] Custom voice training
- [ ] Integration con calendari/email
- [ ] Multi-room support

#### Lungo Termine
- [ ] Federated learning
- [ ] Custom model training
- [ ] Mobile app native
- [ ] Enterprise features

---

## Slide 24: Metriche e Performance

Le performance del sistema variano significativamente in base all'hardware disponibile. Con modelli piccoli su CPU, otteniamo latenze totali end-to-end di 1.6-2.4 secondi, con STT che contribuisce per 200-300ms, LLM per 1-1.5 secondi (di cui 200-400ms per il primo token), e TTS per 400-600ms. Utilizzando GPU e modelli avanzati, possiamo ridurre la latenza totale a 1.1-1.8 secondi, con miglioramenti particolarmente evidenti in STT (100-200ms con CUDA) e TTS (200-400ms con VibeVoice streaming). Il throughput è principalmente limitato dal LLM: Ollama locale gestisce circa 1-2 conversazioni simultanee per GPU, mentre OpenRouter è limitato dai rate limits dell'API. Le risorse tipiche richieste sono 8-16GB di RAM, 4-8GB di VRAM GPU per modelli avanzati, e 20-50GB di storage per tutti i modelli.

---

## Slide 25: Troubleshooting Comune

### Problemi e Soluzioni

#### LiveKit non si avvia
```bash
# Verifica log
docker-compose logs livekit

# Riavvia
docker-compose restart
```

#### Ollama non raggiungibile
```bash
# Verifica Ollama
curl http://localhost:11434/api/tags

# Avvia Ollama
ollama serve
```

#### TTS non funziona
- **Edge**: Richiede internet
- **Piper**: Verifica modelli scaricati
- **VibeVoice**: Richiede GPU e TTS server esterno
- **Chatterbox**: Verifica installazione e dipendenze

#### Whisper lento
- Usa modello più piccolo (`tiny` o `base`)
- Abilita GPU: `WHISPER_DEVICE=cuda`
- Riduci `beam_size`

#### Database connection error
```bash
# Verifica PostgreSQL
docker-compose ps postgres
docker-compose logs postgres

# Riavvia
docker-compose restart postgres
```

---

## Slide 26: Demo - Flusso Completo

### Esempio di Conversazione

```
1. User apre browser → https://localhost:8443
2. Clicca "Connetti" → Token generato
3. WebRTC connesso → Agent dispatchato
4. User parla: "Ciao, come stai?"
   
   → STT: "Ciao, come stai?"
   → LLM: "Ciao! Sto bene, grazie. Come posso aiutarti?"
   → TTS: Audio generato
   → Playback: User sente risposta

5. User: "@sophyai che tempo fa?"
   
   → STT: "@sophyai che tempo fa?"
   → LLM: "Non ho accesso alle previsioni meteo..."
   → TTS: Audio
   → Playback

6. Conversazione salvata in database
7. User può vedere history via API
```

---

## Slide 27: Integrazioni Possibili

### Come Integrare con Altri Sistemi

#### 1. Webhook Integration
```python
# In agent/main.py
async def send_webhook(data):
    async with aiohttp.ClientSession() as session:
        await session.post("https://your-api.com/webhook", json=data)
```

#### 2. External API Calls
```python
# Function tool per chiamate API
@function_tool(description="Chiama API esterna")
async def call_external_api(context: RunContext, url: str) -> str:
    # Implementazione
    pass
```

#### 3. Database External
- Modifica `DATABASE_URL` in `.env`
- Supporta qualsiasi PostgreSQL remoto

#### 4. Custom Frontend
- API REST completamente documentate
- WebSocket LiveKit per audio/video
- Frontend può essere qualsiasi framework

---

## Slide 28: Conclusioni

### Punti Chiave

✅ **Sistema Completo**: STT + LLM + TTS integrato  
✅ **Self-Hosted**: Privacy e controllo totale  
✅ **Modulare**: Facile estendere e personalizzare  
✅ **Multi-Engine**: Scelta tra diverse tecnologie  
✅ **Production-Ready**: Docker, database, monitoring  
✅ **Open Source**: Base per personalizzazioni  

### Prossimi Passi

1. **Setup Ambiente**: Docker Compose + LiveKit
2. **Configurazione**: TTS, LLM, Whisper
3. **Testing**: Verifica tutti i componenti
4. **Personalizzazione**: System prompt, voci, modelli
5. **Deployment**: Produzione con sicurezza

---

## Slide 29: Q&A

### Domande?

**Contatti**:
- Repository: [GitHub]
- Documentazione: `DOCUMENTAZIONE.md`
- Issues: [GitHub Issues]

**Risorse**:
- LiveKit Docs: https://docs.livekit.io
- Ollama: https://ollama.ai
- OpenRouter: https://openrouter.ai
- Whisper: https://github.com/openai/whisper

---

## Slide 30: Grazie

# Grazie per l'Attenzione!

### SophyAI Live Server
*Sistema di Assistente Vocale Self-Hosted*

**Domande e Discussione**

---

## Note per il Presentatore

### Timing Consigliato

- **Slide 1-5**: Introduzione (5 min)
- **Slide 6-12**: Architettura e Moduli (15 min)
- **Slide 13-18**: Deployment e API (10 min)
- **Slide 19-24**: Avanzato e Performance (10 min)
- **Slide 25-30**: Demo e Q&A (10 min)

**Totale**: ~50 minuti

### Slide da Evidenziare

- **Slide 4**: Architettura (mostra diagramma)
- **Slide 10**: Flusso end-to-end (chiave del sistema)
- **Slide 14**: Deployment (infrastruttura)
- **Slide 26**: Demo (mostra funzionamento)

### Preparazione Demo

1. Sistema avviato e funzionante
2. Browser aperto su localhost:8443
3. Microfono testato
4. Esempi di conversazione pronti
5. API testate (curl o Postman)
