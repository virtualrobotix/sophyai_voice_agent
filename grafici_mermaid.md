# Diagrammi Mermaid per la Presentazione

Questi diagrammi possono essere:
1. Visualizzati direttamente in Markdown viewers (GitHub, VS Code, ecc.)
2. Convertiti in PNG usando [Mermaid CLI](https://github.com/mermaid-js/mermaid-cli) o tool online
3. Copiati in PowerPoint come immagini

## 1. Architettura ad Alto Livello

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

## 2. Flusso End-to-End

```mermaid
sequenceDiagram
    participant User
    participant Browser
    participant LiveKit
    participant Agent
    participant STT
    participant LLM
    participant TTS
    
    User->>Browser: Parla (microfono)
    Browser->>LiveKit: Audio Stream (WebRTC)
    LiveKit->>Agent: Audio Frame
    
    Note over Agent,STT: Speech-to-Text
    Agent->>STT: Transcribe Audio
    STT-->>Agent: Text (~250ms)
    
    Note over Agent,LLM: Language Model
    Agent->>LLM: Generate Response
    LLM-->>Agent: Response Text (~1200ms)
    
    Note over Agent,TTS: Text-to-Speech
    Agent->>TTS: Synthesize Text
    TTS-->>Agent: Audio Data (~500ms)
    
    Agent->>LiveKit: Audio Stream
    LiveKit->>Browser: Audio Playback (WebRTC)
    Browser->>User: Audio (altoparlanti)
    
    Note over User,TTS: Totale E2E: ~1950ms
```

## 3. Deployment Docker

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
    Agent -->|host.docker.internal| Ollama
    Agent -->|host.docker.internal| TTS_Server
    LiveKit --> Redis
```

## 4. Database Schema

```mermaid
erDiagram
    SETTINGS {
        varchar key PK
        text value
        timestamp updated_at
    }
    
    CHATS {
        serial id PK
        varchar title
        timestamp created_at
        timestamp updated_at
    }
    
    MESSAGES {
        serial id PK
        integer chat_id FK
        varchar role
        text content
        timestamp created_at
    }
    
    CHATS ||--o{ MESSAGES : "has many"
```

## 5. Sequenza Multi-User

```mermaid
sequenceDiagram
    participant User1
    participant Browser1
    participant LiveKit
    participant Agent
    participant Browser2
    participant User2
    
    User1->>Browser1: "Ciao a tutti"
    Browser1->>LiveKit: Audio Stream
    LiveKit->>Agent: Audio (User1)
    Agent->>Agent: STT: "Ciao a tutti"
    Agent->>Agent: Check: No @sophyai
    Agent->>Agent: ❌ Ignore (multi-user mode)
    
    User2->>Browser2: "@sophyai come stai?"
    Browser2->>LiveKit: Audio Stream
    LiveKit->>Agent: Audio (User2)
    Agent->>Agent: STT: "@sophyai come stai?"
    Agent->>Agent: Check: @sophyai found
    Agent->>Agent: ✅ Process & Respond
    Agent->>LiveKit: Audio Response
    LiveKit->>Browser1: Audio (User1 ascolta)
    LiveKit->>Browser2: Audio (User2 ascolta)
```

## 6. Stack Tecnologico

```mermaid
graph TD
    subgraph Frontend["Frontend Layer"]
        HTML[HTML5]
        JS[JavaScript ES6+]
        WebRTC[WebRTC API]
        Audio[Web Audio API]
    end
    
    subgraph Backend["Backend Layer"]
        FastAPI[FastAPI]
        Python[Python 3.10+]
        REST[REST API]
    end
    
    subgraph WebRTC_Layer["WebRTC Layer"]
        LiveKit_SDK[LiveKit]
        Agents[Agents SDK]
    end
    
    subgraph AI_ML["AI/ML Layer"]
        Whisper[Whisper STT]
        Ollama[Ollama LLM]
        OpenRouter[OpenRouter]
        TTS_Engines[TTS Engines]
    end
    
    subgraph Infra["Infrastructure"]
        PostgreSQL[PostgreSQL 16]
        Redis[Redis 7]
        Docker[Docker]
    end
    
    Frontend --> Backend
    Frontend --> WebRTC_Layer
    Backend --> WebRTC_Layer
    Backend --> Infra
    WebRTC_Layer --> AI_ML
    AI_ML --> Infra
```

## 7. Confronto TTS Engines

```mermaid
graph LR
    subgraph Cloud["Cloud-based"]
        Edge[Edge TTS<br/>Microsoft<br/>Qualità: ⭐⭐⭐⭐<br/>Velocità: ⭐⭐⭐⭐⭐]
    end
    
    subgraph SelfHosted["Self-Hosted"]
        Piper[Piper TTS<br/>Qualità: ⭐⭐⭐<br/>Velocità: ⭐⭐⭐⭐⭐]
        Coqui[Coqui TTS<br/>Qualità: ⭐⭐⭐⭐<br/>Velocità: ⭐⭐⭐]
        Kokoro[Kokoro TTS<br/>Qualità: ⭐⭐⭐⭐<br/>Velocità: ⭐⭐⭐]
        VibeVoice[VibeVoice<br/>Microsoft<br/>Qualità: ⭐⭐⭐⭐⭐<br/>Velocità: ⭐⭐⭐⭐]
        Chatterbox[Chatterbox<br/>Resemble AI<br/>Qualità: ⭐⭐⭐⭐⭐<br/>Velocità: ⭐⭐⭐]
    end
```

## 8. Metriche Performance

```mermaid
gantt
    title Latenze End-to-End (CPU vs GPU)
    dateFormat X
    axisFormat %L ms
    
    section CPU (Small Model)
    STT           :0, 250
    LLM           :250, 1200
    TTS           :1450, 500
    Totale E2E    :0, 1950
    
    section GPU (Advanced)
    STT           :0, 150
    LLM           :150, 800
    TTS           :950, 300
    Totale E2E    :0, 1250
```

## Come Convertire in Immagini

### Opzione 1: Mermaid CLI

```bash
# Installa Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Converti diagramma
mmdc -i grafici_mermaid.md -o grafici_presentazione/ -b transparent
```

### Opzione 2: Tool Online

1. Vai su https://mermaid.live
2. Incolla il codice Mermaid
3. Esporta come PNG/SVG

### Opzione 3: VS Code Extension

1. Installa extension "Markdown Preview Mermaid Support"
2. Apri questo file in VS Code
3. Usa "Export Diagram" dal menu contestuale

### Opzione 4: Python Script (matplotlib)

```bash
# Installa dipendenze
./installa_dipendenze_grafici.sh

# Genera grafici
python3 genera_grafici.py
```
