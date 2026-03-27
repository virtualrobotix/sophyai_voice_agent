# Architettura del Sistema SophyAI Voice Agent

## Panoramica

SophyAI Voice Agent e' una piattaforma self-hosted per assistenza vocale in tempo reale basata su WebRTC.
Il sistema integra riconoscimento vocale (STT), modelli linguistici (LLM), sintesi vocale (TTS) e telefonia SIP in un'architettura a microservizi orchestrata con Docker Compose.

## Diagramma Architetturale

```
                                  Internet / LAN
                                       |
                     +-----------------+-----------------+
                     |                                   |
              HTTPS :8443                          WSS :7443
                     |                                   |
              +------+------+                  +---------+---------+
              |  FastAPI    |                  |  Nginx TLS Proxy  |
              |  server.py  |                  |  (livekit-tls-    |
              |  Web + API  |                  |   proxy)          |
              +------+------+                  +---------+---------+
                     |                                   |
                     |  Docker Network                   |
                     |  (voiceagent 172.18.x.x)         |
                     |                                   |
              +------+------+                  +---------+---------+
              |  PostgreSQL |                  |  LiveKit Server   |
              |  :5432      |                  |  :7880 (host)     |
              +-------------+                  +---------+---------+
                                                    |    |
                                               +----+    +----+
                                               |              |
                                     +---------+---+   +------+------+
                                     | Voice Agent |   | SIP Bridge  |
                                     | (worker)    |   | (livekit/   |
                                     |             |   |  sip)       |
                                     +------+------+   +-------------+
                                            |
                    +-----------+-----------+-----------+
                    |           |           |           |
              +-----+---+ +----+----+ +----+----+ +---+-------+
              | Whisper  | | Ollama  | | TTS     | | Redis     |
              | STT      | | LLM    | | Engine  | | :6379     |
              | :8091    | | :11434 | | :8092   | +-----------+
              +----------+ +---------+ +---------+
```

## Componenti

### 1. Web Server (`voice-agent-web`)

**File**: `server.py`
**Framework**: FastAPI + Uvicorn
**Porte**: HTTP 8080, HTTPS 8443

Responsabilita':
- Serve le pagine frontend (index.html, admin.html, login.html)
- API REST per configurazione, chat, utenti, chiamate
- Generazione token LiveKit per connessioni WebRTC
- Middleware di autenticazione JWT con cookie httponly
- Gestione webhook LiveKit
- Proxy per restart agent tramite Docker socket

### 2. Voice Agent Worker (`voice-agent-worker`)

**File**: `agent/main.py`
**Framework**: LiveKit Agents SDK

Responsabilita':
- Orchestrazione pipeline vocale: STT -> LLM -> TTS
- Gestione sessioni LiveKit (join/leave room)
- Voice Activity Detection (VAD) con Silero
- Wake word detection e interruzione TTS (barge-in)
- Calibrazione audio real-time (canali SIP e Web separati)
- Logging chiamate SIP nel database
- Vision/video analysis (opzionale)
- Remote LLM con RAG via server esterno

### 3. Database (`voice-agent-db`)

**Tecnologia**: PostgreSQL 16
**Porta**: 5432

Tabelle:
| Tabella | Descrizione |
|---------|-------------|
| `settings` | Configurazione key-value (LLM, TTS, Whisper, SMTP, soglie VAD, JWT secret) |
| `chats` | Sessioni di conversazione web |
| `messages` | Messaggi individuali per chat |
| `users` | Utenti della piattaforma (username, password bcrypt, ruolo, email) |
| `password_reset_tokens` | Token per reset password via email |
| `call_logs` | Registro chiamate SIP (durata, stato, metadati) |
| `call_messages` | Trascrizioni delle conversazioni telefoniche |

### 4. LiveKit Server (host)

**Porta**: 7880 (WS), 7881-7882 (RTC/TCP)
**Configurazione**: `livekit-server.yaml`

Non containerizzato, gira sull'host. Gestisce:
- Stanze WebRTC (rooms)
- Segnalazione e trasporto media
- Routing SIP via LiveKit SIP Bridge

### 5. LiveKit TLS Proxy (`livekit-tls-proxy`)

**Tecnologia**: Nginx
**Porta**: 7443

Proxy WSS -> WS per connessioni LiveKit sicure dal browser.
Necessario perche' i browser richiedono WSS quando la pagina e' servita via HTTPS.

### 6. SIP Bridge (`livekit-sip`)

**Immagine**: `livekit/sip:latest`
**Network**: host mode (per NAT SIP/RTP)

Gestisce:
- Ricezione chiamate SIP (trunk inbound)
- Dispatch verso room LiveKit
- Supporto codec G.711, Opus

### 7. Redis (`livekit-redis`)

**Porta**: 127.0.0.1:6379
Usato internamente da LiveKit per la gestione delle stanze e del signaling.

### 8. Whisper STT Server (profilo NVIDIA)

**Porta**: 8091
**Modello**: faster-whisper (large-v3 default)
Server HTTP per trascrizione vocale con GPU CUDA.

### 9. TTS Server (profilo NVIDIA)

**Porta**: 8092
Serve motori TTS multipli (Piper, Kokoro, Edge, VibeVoice, Chatterbox, ElevenLabs, Coqui).

### 10. Ollama (host)

**Porta**: 11434
Runtime LLM locale. I container accedono via `host.docker.internal:11434`.

## Flusso di una Conversazione Vocale

```
1. Browser -> HTTPS :8443 -> FastAPI -> Autenticazione JWT
2. Browser <- Token LiveKit + URL WSS
3. Browser -> WSS :7443 -> Nginx -> LiveKit :7880 -> Room join
4. LiveKit -> Agent worker -> Room join
5. Utente parla -> Browser cattura audio -> LiveKit -> Agent
6. Agent -> Whisper STT -> Testo trascritto
7. Agent -> Ollama/OpenRouter LLM -> Risposta generata
8. Agent -> TTS Engine -> Audio sintetizzato
9. Audio -> LiveKit -> Browser -> Altoparlante
```

## Flusso Chiamata SIP

```
1. Telefono -> SIP Trunk -> livekit-sip -> LiveKit room
2. LiveKit -> Agent worker -> Room join
3. Audio SIP -> Agent -> STT -> LLM -> TTS -> Audio SIP
4. Fine chiamata -> call_logs registrato in PostgreSQL
```

## Rete Docker

- **Network**: `voiceagent` (bridge, subnet 172.18.0.0/16)
- **host.docker.internal**: Risolve all'IP dell'host per raggiungere LiveKit e Ollama
- **Richieste interne** (172.x.x.x): Bypassano l'autenticazione per comunicazione container-to-container
- **SIP Bridge**: Usa `network_mode: host` per evitare problemi NAT

## Sicurezza

### Autenticazione
- **JWT** con cookie httponly (`sophyai_session`)
- **Bcrypt** per hashing password
- Due ruoli: `admin` (accesso completo) e `user` (solo chat)
- Cambio password obbligatorio al primo accesso
- Reset password via SMTP (Gmail, Brevo, Resend, Mailgun, custom)

### Protezione Route
| Path | Accesso |
|------|---------|
| `/login.html`, `/api/auth/*` | Pubblico |
| `/` (chat) | Utente autenticato |
| `/admin` | Solo admin |
| `/api/settings`, `/api/admin/*`, `/api/sip/*` | Solo admin |
| `/api/*` (altri) | Utente autenticato |
| Richieste Docker interne (172.x.x.x) | Bypass auth |

### HTTPS
- Certificati SSL in `certs/` (cert.pem, key.pem)
- Porta 8443 per HTTPS, 8080 per HTTP
- LiveKit WSS su porta 7443 via Nginx proxy

## Persistenza dei Dati

| Volume Docker | Contenuto |
|---------------|-----------|
| `postgres_data` | Database PostgreSQL |
| `redis_data` | Dati Redis LiveKit |
| `whisper_models` | Modelli Whisper scaricati |
| `tts_models` | Modelli TTS scaricati |

## File di Configurazione

| File | Descrizione |
|------|-------------|
| `.env` | Variabili d'ambiente (da `env.example`) |
| `docker-compose.yml` | Definizione servizi Docker |
| `livekit-server.yaml` | Configurazione LiveKit Server |
| `sip-config.yaml` | Configurazione SIP trunk/dispatch |
| `nginx-livekit-proxy.conf` | Nginx TLS proxy per LiveKit |
| `certs/` | Certificati SSL (cert.pem, key.pem) |
| `agent/config.py` | Configurazione runtime agent (da env) |
| DB `settings` | Parametri operativi persistiti |
