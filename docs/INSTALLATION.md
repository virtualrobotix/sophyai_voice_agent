# Manuale di Installazione - SophyAI Voice Agent

## Indice

1. [Requisiti di Sistema](#requisiti-di-sistema)
2. [Prerequisiti Software](#prerequisiti-software)
3. [Installazione Rapida (Docker)](#installazione-rapida-docker)
4. [Configurazione](#configurazione)
5. [Certificati SSL](#certificati-ssl)
6. [Avvio del Sistema](#avvio-del-sistema)
7. [Installazione LiveKit Server](#installazione-livekit-server)
8. [Configurazione SIP](#configurazione-sip)
9. [Installazione con GPU NVIDIA](#installazione-con-gpu-nvidia)
10. [Aggiornamento](#aggiornamento)
11. [Backup e Ripristino](#backup-e-ripristino)
12. [Risoluzione Problemi](#risoluzione-problemi)

---

## Requisiti di Sistema

### Hardware Minimo

| Componente | Minimo | Consigliato |
|------------|--------|-------------|
| CPU | 4 core | 8+ core |
| RAM | 16 GB | 32+ GB |
| Disco | 50 GB SSD | 100+ GB SSD |
| GPU | - | NVIDIA con 8+ GB VRAM |

### Hardware per GPU (Whisper + TTS + LLM)

| GPU | VRAM | Capacita' |
|-----|------|-----------|
| RTX 3060 | 12 GB | LLM 7B + Whisper medium |
| RTX 3090 / 4090 | 24 GB | LLM 13B + Whisper large + TTS |
| A100 / H100 | 40-80 GB | LLM 30B+ + tutti i servizi |

### Porte di Rete

| Porta | Protocollo | Servizio | Obbligatoria |
|-------|-----------|----------|--------------|
| 8443 | TCP/HTTPS | Web server (frontend + API) | Si |
| 8080 | TCP/HTTP | Web server (fallback) | No |
| 7443 | TCP/WSS | LiveKit TLS proxy | Si |
| 7880 | TCP/WS | LiveKit server | Si (solo locale) |
| 7881-7882 | TCP | LiveKit RTC | Si |
| 50000-60000 | UDP | LiveKit WebRTC media | Si |
| 5432 | TCP | PostgreSQL | Solo locale |
| 5060 | UDP/TCP | SIP signaling | Solo se SIP |
| 10000-10100 | UDP | SIP RTP media | Solo se SIP |

### Firewall e NAT per Accesso Esterno

Per consentire l'accesso alla piattaforma da Internet (fuori dalla rete locale), e' necessario configurare il port forwarding sul router/firewall.

**Porte obbligatorie:**

| Porta | Protocollo | Servizio | Note |
|-------|-----------|----------|------|
| **8443** | TCP | Web HTTPS (frontend + API + login) | Pagina web e tutte le API |
| **7443** | TCP | LiveKit WSS signaling | Segnalazione WebRTC. **Senza questa porta la connessione alla room fallisce** con "could not establish signal connection: Failed to fetch" |
| **7881-7882** | TCP | LiveKit RTC/TCP | Trasporto media WebRTC via TCP (fallback) |
| **50000-60000** | UDP | WebRTC media (audio/video) | Trasporto media RTP. Range configurabile in `livekit-server.yaml` |

**Porte opzionali (solo se si usa telefonia SIP):**

| Porta | Protocollo | Servizio | Note |
|-------|-----------|----------|------|
| 5060 | UDP + TCP | SIP signaling | Ricezione chiamate VoIP |
| 10000-10100 | UDP | SIP RTP media | Audio chiamate telefoniche |

**Esempio configurazione router** (tutte le porte puntano all'IP locale del server):

```
8443      TCP  ->  192.168.1.100:8443   # Web HTTPS
7443      TCP  ->  192.168.1.100:7443   # LiveKit WSS
7881-7882 TCP  ->  192.168.1.100:7881-7882  # LiveKit RTC
50000-60000 UDP -> 192.168.1.100:50000-60000  # WebRTC media
```

**Esempio con `ufw` (firewall Linux):**

```bash
sudo ufw allow 8443/tcp    # Web HTTPS
sudo ufw allow 7443/tcp    # LiveKit WSS signaling
sudo ufw allow 7881:7882/tcp  # LiveKit RTC
sudo ufw allow 50000:60000/udp  # WebRTC media
# Solo se SIP:
sudo ufw allow 5060        # SIP signaling
sudo ufw allow 10000:10100/udp  # SIP RTP
```

**Esempio con `iptables`:**

```bash
# Web HTTPS
iptables -A INPUT -p tcp --dport 8443 -j ACCEPT
# LiveKit WSS
iptables -A INPUT -p tcp --dport 7443 -j ACCEPT
# LiveKit RTC
iptables -A INPUT -p tcp --dport 7881:7882 -j ACCEPT
# WebRTC media
iptables -A INPUT -p udp --dport 50000:60000 -j ACCEPT
```

**Ridurre il range UDP:** Se il router non supporta range ampi, modifica `livekit-server.yaml`:

```yaml
rtc:
  port_range_start: 50000
  port_range_end: 50100   # Range ridotto (100 porte)
  use_external_ip: true
```

**Verifica connettivita':** Dopo aver configurato il NAT, verifica dall'esterno:

```bash
# Test porta web
curl -sk https://chatbotdev.sophyai.io:8443/api/health

# Test porta LiveKit WSS (deve rispondere con upgrade WebSocket)
curl -sk -I https://chatbotdev.sophyai.io:7443
```

---

## Prerequisiti Software

### Docker e Docker Compose

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install docker.io docker-compose-plugin
sudo usermod -aG docker $USER
# Riloggati per applicare il gruppo

# Verifica
docker --version
docker compose version
```

### Ollama (LLM locale)

```bash
# Installa Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Scarica un modello (esempio)
ollama pull llama3.2:latest

# Verifica
ollama list
```

### LiveKit Server

Vedi la sezione dedicata [Installazione LiveKit Server](#installazione-livekit-server).

---

## Installazione Rapida (Docker)

### 1. Clona il Repository

```bash
git clone https://github.com/virtualrobotix/sophyai_voice_agent.git
cd sophyai_voice_agent
```

### 2. Configura l'Ambiente

```bash
cp env.example .env
nano .env
```

Parametri essenziali da configurare:

```env
# Nome dell'applicazione
APP_NAME=SophyAI

# IP del server (rilevato automaticamente se vuoto)
SERVER_IP=

# LiveKit
LIVEKIT_URL=ws://localhost:7880

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest

# Whisper
WHISPER_MODEL=medium
```

### 3. Genera Certificati SSL

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem \
  -days 365 -nodes -subj "/CN=sophyai"
```

Per certificati validi (Let's Encrypt), vedi la sezione [Certificati SSL](#certificati-ssl).

### 4. Avvia il Sistema

```bash
# Avvia tutti i servizi
docker compose up -d

# Verifica lo stato
docker compose ps

# Controlla i log
docker compose logs -f web
```

### 5. Primo Accesso

1. Apri `https://<ip-server>:8443` nel browser
2. Accetta il certificato self-signed (se applicabile)
3. Login con `admin` / `admin123`
4. Cambia la password al primo accesso

---

## Configurazione

### File `.env`

Il file `.env` contiene tutte le variabili d'ambiente. Copia da `env.example`:

| Variabile | Default | Descrizione |
|-----------|---------|-------------|
| `APP_NAME` | Receptionist AI | Nome mostrato nell'interfaccia |
| `ASSISTANT_NAME` | Receptionist | Nome dell'assistente |
| `LIVEKIT_URL` | ws://localhost:7880 | URL del server LiveKit |
| `LIVEKIT_API_KEY` | devkey | Chiave API LiveKit |
| `LIVEKIT_API_SECRET` | secret_dev_key... | Secret API LiveKit |
| `SERVER_IP` | (auto) | IP del server per connessioni remote |
| `OLLAMA_HOST` | http://localhost:11434 | URL Ollama |
| `OLLAMA_MODEL` | gpt-oss:20b | Modello LLM da utilizzare |
| `WHISPER_MODEL` | base | Modello Whisper (tiny/base/small/medium/large-v3) |
| `DEFAULT_TTS` | edge | Motore TTS di default |
| `WEB_PORT` | 8080 | Porta HTTP (HTTPS e' 8443) |

### Configurazione Database

Il database PostgreSQL viene inizializzato automaticamente con lo schema in `db/schema.sql`.
Credenziali default (modificare in produzione):

```
Host: postgres (interno Docker) / localhost:5432 (esterno)
Database: voiceagent
Username: voiceagent
Password: voiceagent_pwd
```

### Configurazione da Admin Panel

La maggior parte dei parametri operativi sono configurabili dall'interfaccia admin senza dover modificare file:
- Provider e modello LLM
- Motore e voce TTS
- System prompt e context injection
- Soglie VAD e calibrazione audio
- Configurazione SMTP
- Gestione utenti

---

## Certificati SSL

### Self-Signed (Sviluppo)

```bash
mkdir -p certs
openssl req -x509 -newkey rsa:4096 \
  -keyout certs/key.pem -out certs/cert.pem \
  -days 365 -nodes \
  -subj "/CN=sophyai" \
  -addext "subjectAltName=IP:<tuo-ip>,DNS:localhost"
```

### Let's Encrypt (Produzione)

```bash
# Installa certbot
sudo apt install certbot

# Genera certificato (richiede dominio pubblico)
sudo certbot certonly --standalone -d tuodominio.com

# Copia i certificati
cp /etc/letsencrypt/live/tuodominio.com/fullchain.pem certs/cert.pem
cp /etc/letsencrypt/live/tuodominio.com/privkey.pem certs/key.pem
```

I certificati devono essere in `certs/cert.pem` e `certs/key.pem`.

---

## Avvio del Sistema

### Avvio Completo

```bash
# Avvia LiveKit server (se non e' un servizio systemd)
livekit-server --config livekit-server.yaml &

# Avvia tutti i container Docker
docker compose up -d

# Per ambienti con GPU NVIDIA
docker compose --profile nvidia up -d
```

### Verifica Stato

```bash
# Stato container
docker compose ps

# Log del web server
docker compose logs -f web

# Log dell'agent
docker compose logs -f agent

# Health check
curl -sk https://localhost:8443/api/health
```

### Stop e Restart

```bash
# Stop tutto
docker compose down

# Restart singolo servizio
docker compose restart web
docker compose restart agent

# Rebuild dopo modifiche al Dockerfile
docker compose up -d --build web agent
```

---

## Installazione LiveKit Server

### Metodo 1: Binary (Consigliato)

```bash
# Scarica l'ultima versione
curl -sSL https://get.livekit.io | bash

# Oppure scarica manualmente da:
# https://github.com/livekit/livekit/releases
```

### Metodo 2: Systemd Service

Crea `/etc/systemd/system/livekit.service`:

```ini
[Unit]
Description=LiveKit Server
After=network.target

[Service]
Type=simple
User=livekit
WorkingDirectory=/opt/livekit
ExecStart=/usr/local/bin/livekit-server --config /opt/livekit/config.yaml
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable livekit
sudo systemctl start livekit
```

### Configurazione LiveKit (`livekit-server.yaml`)

```yaml
port: 7880
rtc:
  tcp_port: 7881
  port_range_start: 50000
  port_range_end: 60000
  use_external_ip: true
redis:
  address: 127.0.0.1:6379
keys:
  devkey: secret_dev_key_change_in_production
logging:
  level: info
```

**IMPORTANTE**: Cambia `devkey` e il secret in produzione.

---

## Configurazione SIP

### Prerequisiti

- Trunk SIP con un provider VoIP (es. Twilio, VoIP.ms, provider locale)
- Porte 5060 (SIP) e 10000-10100 (RTP) aperte nel firewall

### File `sip-config.yaml`

```yaml
sip:
  # IP pubblico del server per NAT traversal
  nat_1_to_1_ip: "TUO_IP_PUBBLICO"
```

### Configurazione Trunk da Admin

Dal tab SIP nel pannello admin:
1. Configura il trunk SIP (numero, credenziali provider)
2. Imposta le regole di dispatch (quale room per quale numero)
3. Configura contesti per numero (risposte personalizzate)
4. Testa la configurazione

---

## Installazione con GPU NVIDIA

### Prerequisiti

```bash
# Driver NVIDIA
nvidia-smi  # Verifica che i driver siano installati

# NVIDIA Container Toolkit
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Avvio con Profilo NVIDIA

```bash
# Avvia tutti i servizi inclusi Whisper e TTS con GPU
docker compose --profile nvidia up -d

# Verifica che la GPU sia visibile nei container
docker exec voice-agent-whisper nvidia-smi
```

### Variabili GPU

```env
WHISPER_DEVICE=cuda
WHISPER_MODEL=large-v3
TTS_DEVICE=cuda
```

---

## Aggiornamento

### Aggiornamento del Codice

```bash
# Pull delle ultime modifiche
git pull origin main

# Rebuild dei container
docker compose up -d --build web agent

# Se ci sono modifiche al database schema,
# riavvia il container postgres (le migrazioni sono automatiche)
docker compose restart web
```

### Aggiornamento delle Dipendenze

```bash
# Rebuild completo
docker compose build --no-cache web agent
docker compose up -d
```

### Nota sulle Dipendenze Runtime

Se hai installato dipendenze manualmente nel container (es. `docker exec ... pip install`),
queste vengono perse al rebuild. Aggiungi sempre le dipendenze a `requirements.txt`.

---

## Backup e Ripristino

### Backup Database

```bash
# Backup completo
docker exec voice-agent-db pg_dump -U voiceagent voiceagent > backup_$(date +%Y%m%d).sql

# Backup solo struttura
docker exec voice-agent-db pg_dump -U voiceagent --schema-only voiceagent > schema_backup.sql
```

### Ripristino Database

```bash
# Ripristino completo
cat backup.sql | docker exec -i voice-agent-db psql -U voiceagent voiceagent
```

### Backup Configurazione

```bash
# Salva tutti i file di configurazione
tar -czf config_backup_$(date +%Y%m%d).tar.gz \
  .env \
  certs/ \
  sip-config.yaml \
  livekit-server.yaml \
  config/
```

---

## Risoluzione Problemi

### Il server non si avvia

```bash
# Controlla i log
docker compose logs web --tail 50

# Errori comuni:
# - "Database non disponibile": PostgreSQL non pronto, attendi e riprova
# - "passlib/bcrypt": Versione bcrypt incompatibile, usa bcrypt==4.2.1
# - "Certificate error": Verifica che certs/cert.pem e key.pem esistano
```

### Non riesco a connettermi da remoto

1. Verifica che le porte siano aperte nel firewall:
   ```bash
   sudo ufw allow 8443/tcp   # Web HTTPS
   sudo ufw allow 7443/tcp   # LiveKit WSS
   sudo ufw allow 7881:7882/tcp  # LiveKit RTC
   sudo ufw allow 50000:60000/udp  # WebRTC media
   ```
2. Verifica che `SERVER_IP` in `.env` sia corretto (o lascialo vuoto per auto-detect)
3. Se usi certificati self-signed, accetta l'eccezione nel browser

### "Could not establish signal connection: Failed to fetch"

Questo errore significa che il browser non riesce a raggiungere il server LiveKit sulla porta 7443.

1. **Causa piu' comune**: La porta **7443 TCP** non e' nattata/forwardata sul router
2. Configura il port forwarding per la porta 7443 TCP verso l'IP locale del server
3. Vedi la sezione [Firewall e NAT per Accesso Esterno](#firewall-e-nat-per-accesso-esterno) per la lista completa delle porte
4. Verifica dall'esterno: `curl -sk https://tuodominio.com:7443` - deve dare una risposta (anche errore 400 va bene, significa che la porta e' raggiungibile)

### LiveKit non si connette

1. Verifica che LiveKit server sia in esecuzione:
   ```bash
   curl http://localhost:7880
   ```
2. Verifica che il proxy TLS funzioni:
   ```bash
   curl -sk https://localhost:7443
   ```
3. Controlla i log del proxy:
   ```bash
   docker compose logs livekit-tls-proxy
   ```

### L'agent non risponde

```bash
# Verifica che l'agent sia connesso
docker compose logs agent --tail 20

# Verifica Ollama
curl http://localhost:11434/api/tags

# Restart dell'agent
docker compose restart agent
```

### Errore 401/403 sulle API

- **401**: Non autenticato. Effettua il login o verifica che il cookie JWT sia valido.
- **403**: Accesso negato. L'endpoint richiede ruolo admin.
- Le richieste interne Docker (172.x.x.x) bypassano l'auth.

### Audio non funziona

1. Verifica i permessi del microfono nel browser
2. Prova la calibrazione automatica dal pannello audio
3. Controlla le soglie VAD nel tab "Voice" dell'admin

### Email non si inviano

1. Verifica la configurazione SMTP nel tab "Email / SMTP"
2. Usa "Invia Email di Test" per verificare
3. Per Gmail: assicurati di usare una App Password, non la password dell'account
4. Controlla i log: `docker compose logs web | grep SMTP`
