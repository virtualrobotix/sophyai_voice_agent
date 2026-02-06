# Configurazione SIP per SophyAI Voice Agent

Guida completa per configurare il bridge SIP-WebRTC con LiveKit per ricevere chiamate telefoniche tramite Twilio.

## Indice

1. [Architettura](#architettura)
2. [Prerequisiti](#prerequisiti)
3. [Configurazione Server SIP](#configurazione-server-sip)
4. [Configurazione Trunk SIP](#configurazione-trunk-sip)
5. [Configurazione Dispatch Rules](#configurazione-dispatch-rules)
6. [Configurazione Twilio](#configurazione-twilio)
7. [Persistenza e Avvio Automatico](#persistenza-e-avvio-automatico)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Architettura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              TWILIO                                     │
│  Numero: +39 011 19517814                                              │
│  SIP Trunk: aims-dev-trunk.pstn.twilio.com                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ SIP INVITE (porta 5060)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SERVER (188.166.134.148)                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LiveKit SIP Service                          │   │
│  │  - Riceve chiamate SIP                                          │   │
│  │  - Converte SIP ↔ WebRTC                                        │   │
│  │  - Gestisce NAT traversal                                       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    │ WebRTC                             │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    LiveKit Server                               │   │
│  │  - Crea Room per ogni chiamata                                  │   │
│  │  - Gestisce partecipanti                                        │   │
│  │  - Routing audio/video                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                    │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Voice Agent                                  │   │
│  │  - STT (Whisper)                                                │   │
│  │  - LLM (Ollama)                                                 │   │
│  │  - TTS (Edge/Kokoro/etc)                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisiti

### Servizi Docker Richiesti

| Servizio | Porta | Descrizione |
|----------|-------|-------------|
| `redis` | 6379 | Stato sessioni SIP |
| `livekit` | 7880-7882 | LiveKit Server |
| `sip` | 5060 (UDP/TCP) | LiveKit SIP Bridge |
| `agent` | - | Voice Agent |

### Porte Firewall

Aprire le seguenti porte nel firewall:

```bash
# SIP Signaling
ufw allow 5060/udp
ufw allow 5060/tcp

# RTP Media (range)
ufw allow 10000:20000/udp

# LiveKit WebRTC
ufw allow 7880/tcp
ufw allow 7881/tcp
ufw allow 7882/udp
```

---

## Configurazione Server SIP

### File: `sip-config.yaml`

```yaml
# LiveKit SIP Configuration

# Redis per stato sessioni
redis:
  address: 127.0.0.1:6379

# Credenziali LiveKit API
api_key: devkey
api_secret: secret_dev_key_change_in_production
ws_url: ws://127.0.0.1:7880

# Porta SIP signaling
sip_port: 5060

# Range porte RTP per media
rtp_port:
  start: 10000
  end: 20000

# NAT Configuration
# IMPORTANTE: Se il server è dietro NAT/firewall, specificare l'IP pubblico
nat_1_to_1_ip: "188.166.134.148"      # IP pubblico per SIP headers
media_nat_1_to_1_ip: "188.166.134.148" # IP pubblico per RTP media

# Logging
logging:
  level: info
```

### Parametri NAT Importanti

| Parametro | Descrizione |
|-----------|-------------|
| `nat_1_to_1_ip` | IP pubblico da usare negli header SIP Contact e Via |
| `media_nat_1_to_1_ip` | IP pubblico da usare nell'SDP per il media RTP |
| `use_external_ip` | Auto-detect IP (NON usare insieme a nat_1_to_1_ip) |

**Nota**: Se il server ha un IP privato ma è raggiungibile da un IP pubblico diverso (NAT 1:1), usare `nat_1_to_1_ip`.

### Docker Compose - Servizio SIP

```yaml
sip:
  image: livekit/sip:latest
  container_name: livekit-sip
  restart: unless-stopped
  network_mode: host  # Necessario per SIP/RTP
  environment:
    - LIVEKIT_URL=ws://127.0.0.1:7880
    - LIVEKIT_API_KEY=devkey
    - LIVEKIT_API_SECRET=secret_dev_key_change_in_production
    - SIP_CONFIG_FILE=/sip/config.yaml
  volumes:
    - ./sip-config.yaml:/sip/config.yaml:ro
```

---

## Configurazione Trunk SIP

### Cos'è un Trunk SIP?

Il trunk SIP definisce da dove arrivano le chiamate e quali numeri sono associati. Per chiamate inbound (in entrata), serve un **Inbound Trunk**.

### Creare un Inbound Trunk

#### Metodo 1: API HTTP (Twirp)

```bash
# Genera JWT token
TOKEN=$(python3 -c "
import jwt, time
token = jwt.encode({
    'iss': 'devkey',
    'sub': 'admin',
    'iat': int(time.time()),
    'exp': int(time.time()) + 3600,
    'video': {'roomAdmin': True, 'room': '*'},
    'sip': {'admin': True, 'call': True}
}, 'secret_dev_key_change_in_production', algorithm='HS256')
print(token)
")

# Crea trunk
curl -X POST "http://localhost:7880/twirp/livekit.SIP/CreateSIPInboundTrunk" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "trunk": {
      "name": "Twilio AIMS Trunk",
      "numbers": ["+3901119517814"],
      "allowed_addresses": [
        "aims-dev-trunk.pstn.twilio.com",
        "54.172.60.0/23",
        "54.244.51.0/24",
        "35.156.191.128/25",
        "54.171.127.192/26"
      ],
      "metadata": "{\"provider\": \"twilio\"}"
    }
  }'
```

#### Metodo 2: LiveKit CLI

```bash
# Installa CLI
pip install livekit-cli

# Crea file JSON
cat > trunk.json << 'EOF'
{
  "name": "Twilio AIMS Trunk",
  "numbers": ["+3901119517814"],
  "allowed_addresses": [
    "aims-dev-trunk.pstn.twilio.com",
    "54.172.60.0/23",
    "54.244.51.0/24"
  ]
}
EOF

# Crea trunk
livekit-cli sip inbound create --request trunk.json
```

### Parametri Trunk Inbound

| Campo | Tipo | Descrizione |
|-------|------|-------------|
| `name` | string | Nome identificativo del trunk |
| `numbers` | array | Numeri di telefono associati (E.164 format) |
| `allowed_addresses` | array | IP/CIDR da cui accettare chiamate |
| `allowed_numbers` | array | Numeri chiamanti permessi (regex) |
| `auth_username` | string | Username per digest auth (opzionale) |
| `auth_password` | string | Password per digest auth (opzionale) |
| `metadata` | string | Metadata JSON custom |

### Listare Trunk Esistenti

```bash
curl -X POST "http://localhost:7880/twirp/livekit.SIP/ListSIPInboundTrunk" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{}'
```

### Eliminare un Trunk

```bash
curl -X POST "http://localhost:7880/twirp/livekit.SIP/DeleteSIPTrunk" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"sipTrunkId": "ST_xxxxxxxxxxxx"}'
```

---

## Configurazione Dispatch Rules

### Cos'è una Dispatch Rule?

Le dispatch rules determinano come instradare le chiamate in arrivo. Definiscono in quale room LiveKit posizionare il chiamante.

### Tipi di Dispatch Rules

| Tipo | Descrizione |
|------|-------------|
| `dispatchRuleDirect` | Manda tutte le chiamate a una room specifica |
| `dispatchRuleIndividual` | Crea una room unica per ogni chiamata |
| `dispatchRuleCallee` | Usa il numero chiamato come nome room |

### Creare una Dispatch Rule

#### Rule Individual (Consigliata per Voice Agent)

Crea una room separata per ogni chiamata:

```bash
curl -X POST "http://localhost:7880/twirp/livekit.SIP/CreateSIPDispatchRule" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "rule": {
      "name": "Default Inbound Handler",
      "dispatchRuleIndividual": {
        "roomPrefix": "sip-call-"
      }
    }
  }'
```

Questo crea room con nomi come: `sip-call-abc123`, `sip-call-def456`, etc.

#### Rule Direct (Room Fissa)

Manda tutte le chiamate alla stessa room:

```bash
curl -X POST "http://localhost:7880/twirp/livekit.SIP/CreateSIPDispatchRule" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{
    "rule": {
      "name": "Conference Room",
      "dispatchRuleDirect": {
        "roomName": "conference-room"
      }
    }
  }'
```

### Parametri Dispatch Rule

| Campo | Tipo | Descrizione |
|-------|------|-------------|
| `name` | string | Nome identificativo della rule |
| `trunk_ids` | array | Limita a specifici trunk (vuoto = tutti) |
| `hide_phone_number` | bool | Nasconde numero negli attributi partecipante |
| `inbound_numbers` | array | Filtra per numeri chiamati (regex) |
| `metadata` | string | Metadata passato alla room |

### Listare Dispatch Rules

```bash
curl -X POST "http://localhost:7880/twirp/livekit.SIP/ListSIPDispatchRule" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{}'
```

---

## Configurazione Twilio

### 1. Creare SIP Trunk su Twilio

1. Accedi a [Twilio Console](https://console.twilio.com)
2. Vai a **Elastic SIP Trunking** → **Trunks**
3. Crea nuovo trunk

### 2. Configurare Origination (Chiamate Uscenti)

Non necessario per solo inbound.

### 3. Configurare Termination (Chiamate Entranti)

1. Nel trunk, vai a **Termination**
2. Aggiungi **Termination SIP URI**:
   ```
   sip:188.166.134.148:5060
   ```
3. Configura **Credential List** se usi autenticazione

### 4. Associare Numero di Telefono

1. Vai a **Phone Numbers** → **Manage** → **Active Numbers**
2. Seleziona il numero (+39 011 19517814)
3. In **Voice Configuration**:
   - **Configure With**: SIP Trunk
   - **SIP Trunk**: Seleziona il trunk creato

### 5. IP Addresses Twilio (per allowed_addresses)

Twilio invia chiamate da questi IP ranges (aggiornare periodicamente):

```
# AIMS (Interconnect)
54.172.60.0/23
54.244.51.0/24
35.156.191.128/25
54.171.127.192/26
35.156.191.0/25

# Standard SIP
54.172.60.0/23
54.244.51.0/24
54.171.127.192/26
```

Verifica sempre su: [Twilio IP Addresses](https://www.twilio.com/docs/sip-trunking/ip-addresses)

---

## Persistenza e Avvio Automatico

### Problema

LiveKit **non persiste** trunk e dispatch rules. Al riavvio del servizio, vanno ricreate.

### Soluzione: Init Script

Lo script `init-sip.sh` viene eseguito all'avvio per ricreare la configurazione:

```yaml
# docker-compose.yml
sip-init:
  image: python:3.11-slim
  container_name: voice-agent-sip-init
  restart: "no"
  network_mode: host
  environment:
    - LIVEKIT_URL=http://127.0.0.1:7880
    - LIVEKIT_API_KEY=devkey
    - LIVEKIT_API_SECRET=secret_dev_key_change_in_production
  volumes:
    - ./init-sip.sh:/init-sip.sh:ro
  entrypoint: ["/bin/bash", "-c"]
  command:
    - |
      pip install -q pyjwt && sleep 5 && /init-sip.sh
```

### Esecuzione Manuale

```bash
# Ricreare configurazione SIP
docker-compose up sip-init

# Verificare stato
docker logs voice-agent-sip-init
```

---

## Troubleshooting

### Verificare Stato SIP

```bash
# Logs servizio SIP
docker logs livekit-sip --tail 50

# Deve mostrare:
# "local": "188.166.134.148", "external": "188.166.134.148"
```

### Verificare Trunk e Rules

```bash
# Con script
./init-sip.sh

# Output atteso:
# [INFO] Configurazione attuale: 1 trunk, 1 dispatch rules
```

### Problemi Comuni

| Problema | Causa | Soluzione |
|----------|-------|-----------|
| "flood" reject | Trunk/Rules mancanti | Esegui `init-sip.sh` |
| "local" IP sbagliato | Manca `nat_1_to_1_ip` | Configura in `sip-config.yaml` |
| Timeout chiamate | Firewall blocca RTP | Apri porte 10000-20000/udp |
| Audio unidirezionale | NAT non configurato | Verifica `media_nat_1_to_1_ip` |
| 401 Unauthorized | Auth mancante/errata | Verifica `allowed_addresses` |

### Test Connettività

```bash
# Test SIP port
nc -vzu 188.166.134.148 5060

# Test da esterno (richiede sipsak)
sipsak -s sip:test@188.166.134.148:5060
```

---

## API Reference

### Endpoint Base

```
http://localhost:7880/twirp/livekit.SIP/
```

### Metodi Disponibili

| Metodo | Descrizione |
|--------|-------------|
| `CreateSIPInboundTrunk` | Crea trunk per chiamate in entrata |
| `CreateSIPOutboundTrunk` | Crea trunk per chiamate in uscita |
| `ListSIPInboundTrunk` | Lista trunk inbound |
| `ListSIPOutboundTrunk` | Lista trunk outbound |
| `DeleteSIPTrunk` | Elimina un trunk |
| `CreateSIPDispatchRule` | Crea dispatch rule |
| `ListSIPDispatchRule` | Lista dispatch rules |
| `DeleteSIPDispatchRule` | Elimina dispatch rule |
| `CreateSIPParticipant` | Avvia chiamata outbound |

### Autenticazione

Tutte le API richiedono JWT token con claims:

```json
{
  "iss": "API_KEY",
  "sub": "admin",
  "iat": 1234567890,
  "exp": 1234571490,
  "video": {"roomAdmin": true, "room": "*"},
  "sip": {"admin": true, "call": true}
}
```

---

## Riferimenti

- [LiveKit SIP Documentation](https://docs.livekit.io/sip/)
- [LiveKit SIP GitHub](https://github.com/livekit/sip)
- [Twilio SIP Trunking](https://www.twilio.com/docs/sip-trunking)
- [Twilio IP Addresses](https://www.twilio.com/docs/sip-trunking/ip-addresses)

---

*Documento aggiornato: 6 Febbraio 2026*
