# SophyAI Voice Agent

Piattaforma self-hosted per assistenza vocale in tempo reale basata su WebRTC.
Integra riconoscimento vocale (STT), modelli linguistici (LLM), sintesi vocale (TTS),
telefonia SIP e chat testuale con autenticazione utenti e pannello di amministrazione.

## Architettura

```
                     HTTPS :8443              WSS :7443
                         |                        |
                  +------+------+       +---------+---------+
                  |  FastAPI    |       |  Nginx TLS Proxy  |
                  |  Web + API  |       |  -> LiveKit       |
                  +------+------+       +---------+---------+
                         |                        |
              +----------+----------+    +--------+--------+
              |  PostgreSQL :5432   |    | LiveKit :7880   |
              +---------------------+    +--------+--------+
                                              |    |
                                    +---------+    +----------+
                                    |                         |
                              +-----+-------+          +-----+-----+
                              | Voice Agent |          | SIP Bridge|
                              |  (worker)   |          |           |
                              +------+------+          +-----------+
                                     |
                    +--------+-------+--------+
                    |        |        |        |
                 Whisper  Ollama    TTS     Redis
                  STT      LLM    Engine
```

## Funzionalita'

- **Assistente vocale WebRTC** con latenza ultra-bassa
- **Multi-LLM**: Ollama (locale), OpenRouter (100+ modelli cloud), Remote LLM con RAG
- **Multi-TTS**: Edge, Piper, Kokoro, ElevenLabs, VibeVoice, Chatterbox, Coqui
- **STT**: Whisper (faster-whisper) con supporto GPU
- **Telefonia SIP**: Ricezione chiamate con logging e contesti per numero
- **Chat testuale**: Conversazioni persistenti con storico
- **Calibrazione audio**: Taratura automatica e manuale per canali Web e SIP
- **Autenticazione**: Login con JWT, ruoli admin/utente, cambio password obbligatorio
- **Reset password**: Via SMTP con supporto Gmail, Brevo, Resend, Mailgun
- **Pannello admin**: Configurazione completa da interfaccia web
- **Video/Vision**: Analisi frame video (opzionale)
- **Branding personalizzabile**: Nome app, assistente, trigger

## Requisiti

- **Docker** e **Docker Compose**
- **Ollama** installato e in esecuzione (per LLM locale)
- **LiveKit Server** installato sull'host
- 16 GB+ RAM (32 GB consigliati)
- GPU NVIDIA (opzionale, consigliato per Whisper large e TTS avanzati)

## Installazione Rapida

```bash
# Clona il repository
git clone https://github.com/virtualrobotix/sophyai_voice_agent.git
cd sophyai_voice_agent

# Configura l'ambiente
cp env.example .env
nano .env

# Genera certificati SSL (self-signed per sviluppo)
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem \
  -days 365 -nodes -subj "/CN=sophyai"

# Avvia il sistema
docker compose up -d

# Accedi
# https://<ip-server>:8443
# Login: admin / admin123 (cambio password obbligatorio)
```

## Documentazione

| Documento | Descrizione |
|-----------|-------------|
| [Architettura](docs/ARCHITECTURE.md) | Diagrammi, componenti, flussi, rete, sicurezza |
| [Manuale Utente](docs/USER_MANUAL.md) | Guida completa all'uso della piattaforma |
| [Manuale Installazione](docs/INSTALLATION.md) | Setup, configurazione, GPU, SIP, backup, troubleshooting |
| [Configurazione SIP](docs/SIP_CONFIGURATION.md) | Dettagli configurazione telefonia SIP |
| [Parametri VAD](docs/VAD_PARAMETERS.md) | Parametri Voice Activity Detection |

## Struttura Progetto

```
sophyai_voice_agent/
├── docker-compose.yml          # Servizi Docker
├── Dockerfile                  # Build container web
├── Dockerfile.agent            # Build container agent
├── Dockerfile.whisper          # Build container Whisper (GPU)
├── Dockerfile.tts              # Build container TTS (GPU)
├── server.py                   # FastAPI: API, auth, webhook, frontend
├── requirements.txt            # Dipendenze Python
├── env.example                 # Template variabili d'ambiente
├── agent/
│   ├── main.py                 # Agent principale (STT->LLM->TTS pipeline)
│   ├── config.py               # Configurazione runtime
│   ├── stt/whisper_stt.py      # Whisper STT
│   ├── llm/
│   │   ├── ollama_llm.py       # LLM Ollama
│   │   ├── openrouter_llm.py   # LLM OpenRouter
│   │   └── remote_llm.py       # LLM remoto con RAG
│   └── tts/                    # Motori TTS (edge, piper, kokoro, ecc.)
├── db/
│   ├── schema.sql              # Schema PostgreSQL
│   └── database.py             # Servizio database async
├── web/
│   ├── index.html              # Pagina chat principale
│   ├── admin.html              # Pannello amministrazione
│   └── login.html              # Pagina di login
├── certs/                      # Certificati SSL
├── config/                     # Configurazioni aggiuntive
├── docs/                       # Documentazione
├── livekit-server.yaml         # Config LiveKit Server
├── sip-config.yaml             # Config SIP
└── nginx-livekit-proxy.conf    # Proxy TLS per LiveKit
```

## Servizi Docker

| Container | Servizio | Porte |
|-----------|----------|-------|
| `voice-agent-web` | FastAPI web server | 8080, 8443 |
| `voice-agent-worker` | LiveKit Agent worker | - |
| `voice-agent-db` | PostgreSQL 16 | 5432 |
| `livekit-redis` | Redis | 127.0.0.1:6379 |
| `livekit-tls-proxy` | Nginx WSS proxy | 7443 |
| `livekit-sip` | SIP Bridge | host network |
| `voice-agent-whisper` | Whisper STT (GPU) | 8091 |
| `voice-agent-tts` | TTS Server (GPU) | 8092 |

## Database

| Tabella | Descrizione |
|---------|-------------|
| `settings` | Configurazione key-value |
| `users` | Utenti (username, password bcrypt, ruolo, email) |
| `password_reset_tokens` | Token reset password |
| `chats` | Sessioni di conversazione |
| `messages` | Messaggi delle chat |
| `call_logs` | Registro chiamate SIP |
| `call_messages` | Trascrizioni chiamate |

## API Principali

| Gruppo | Endpoints |
|--------|-----------|
| **Auth** | `POST /api/auth/login\|logout`, `GET /api/auth/me`, cambio/reset password |
| **Admin Utenti** | `GET/POST/PUT/DELETE /api/admin/users` |
| **Settings** | `GET/POST /api/settings` |
| **Chat** | `GET/POST /api/chats`, messaggi, prompt, context |
| **LiveKit** | `POST /api/token`, `GET /api/rooms`, webhook |
| **TTS** | Engines, voices, test, selezione |
| **LLM** | Modelli Ollama/OpenRouter, selezione, test |
| **SIP** | Status, config, trunks, contesti |
| **Chiamate** | Log, dettaglio, statistiche |
| **SMTP** | Test connessione SMTP |

## Opzioni TTS

| Motore | Self-Hosted | Qualita' | Latenza |
|--------|-------------|----------|---------|
| Edge (Microsoft) | No | Ottima | Bassa |
| Piper | Si | Buona | Molto bassa |
| Kokoro | Si | Alta | Media |
| ElevenLabs | No | Eccellente | Media |
| VibeVoice | Si (GPU) | Eccellente | Bassa |
| Chatterbox | Si | Eccellente | Media |
| Coqui | Si | Alta | Media |

## Opzioni LLM

| Provider | Locale | Modelli |
|----------|--------|---------|
| Ollama | Si | Qualsiasi modello Ollama |
| OpenRouter | No | 100+ (GPT-4, Claude, Gemini, Mistral...) |
| Remote LLM | Configurabile | Server esterno con RAG |

## Sicurezza

- Autenticazione JWT con cookie httponly
- Password hashate con bcrypt
- Ruoli: admin (accesso completo) e user (solo chat)
- Cambio password obbligatorio al primo accesso
- HTTPS con certificati SSL
- Richieste Docker interne esenti da auth
- SMTP configurabile per reset password

## Porte per Accesso Esterno

Per accesso da Internet, configurare il NAT/firewall con queste porte:

| Porta | Proto | Servizio |
|-------|-------|----------|
| **8443** | TCP | Web HTTPS |
| **7443** | TCP | LiveKit WSS signaling |
| **7881-7882** | TCP | LiveKit RTC |
| **50000-60000** | UDP | WebRTC media |
| 5060 | UDP+TCP | SIP (opzionale) |
| 10000-10100 | UDP | SIP RTP (opzionale) |

Per dettagli completi vedi [docs/INSTALLATION.md](docs/INSTALLATION.md#firewall-e-nat-per-accesso-esterno).

## Licenza

MIT License

## Crediti

- [LiveKit](https://livekit.io/) - Infrastruttura WebRTC
- [Whisper](https://github.com/openai/whisper) - Riconoscimento vocale
- [Ollama](https://ollama.ai/) - Runtime LLM locale
- [OpenRouter](https://openrouter.ai/) - API LLM multi-modello
- [Piper](https://github.com/rhasspy/piper) - TTS veloce
- [Edge TTS](https://github.com/rany2/edge-tts) - Microsoft Edge TTS
- [VibeVoice](https://github.com/microsoft/VibeVoice) - Microsoft Real-time TTS
- [Chatterbox](https://github.com/resemble-ai/chatterbox) - Resemble AI TTS
- [Kokoro](https://github.com/hexgrad/kokoro) - TTS multilingua
- [ElevenLabs](https://elevenlabs.io/) - TTS premium
