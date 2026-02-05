# Documentazione VAD e Flusso Audio - SophyAI Voice Agent

Questo documento descrive in dettaglio il funzionamento del Voice Activity Detection (VAD) e il processing audio del sistema SophyAI.

---

## Indice

1. [Panoramica del Sistema](#1-panoramica-del-sistema)
2. [Flusso Audio Completo](#2-flusso-audio-completo)
3. [Parametri Configurabili](#3-parametri-configurabili)
4. [Parametri Hardcoded](#4-parametri-hardcoded)
5. [Parametri Whisper VAD](#5-parametri-whisper-vad)
6. [Timeline di Esempio](#6-timeline-di-esempio)
7. [Configurazioni per Scenari](#7-configurazioni-per-scenari)
8. [API di Configurazione](#8-api-di-configurazione)

---

## 1. Panoramica del Sistema

Il sistema audio di SophyAI gestisce:

- **Rilevamento della voce** (VAD) per capire quando l'utente parla
- **Accumulazione audio** in un buffer prima della trascrizione
- **Barge-in** per interrompere l'agent quando l'utente parla
- **Trascrizione** tramite Whisper (locale o server)
- **Cooldown** per evitare che l'agent "senta" la propria voce

### Caratteristiche Audio

| Proprietà | Valore |
|-----------|--------|
| Sample Rate | 16000 Hz |
| Bit Depth | 16 bit |
| Canali | Mono |
| Frame Size | ~3200 bytes (~100ms) |
| Bytes per ms | 32 bytes |

---

## 2. Flusso Audio Completo

### 2.1 Fase 1: Rilevamento Inizio Parlato

Ogni ~50ms arriva un frame audio. Il sistema calcola l'energia media del frame.

```
┌──────────────────────────────────────────────────────────────────┐
│  AUDIO IN ARRIVO (ogni ~50ms arriva un frame da 3200 bytes)     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Calcola energia │
                    │   del frame     │
                    └─────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    energia >= 25?                    energia < 25?
    (SPEECH_ENERGY_THRESHOLD)         
              │                               │
              ▼                               ▼
    ✅ Accumula nel buffer           ❌ Ignora frame
       speech_frames++                  (rumore di fondo)
```

**Tempo**: Istantaneo, ogni frame viene valutato in tempo reale.

---

### 2.2 Fase 2: Accumulazione e Rilevamento Fine Parlato

Mentre l'utente parla, il buffer si riempie. Quando l'utente fa una pausa, il sistema conta i frame di silenzio.

```
┌──────────────────────────────────────────────────────────────────┐
│  UTENTE STA PARLANDO - Buffer si riempie                        │
│                                                                  │
│  speech_frames: 1, 2, 3... 30... 50...                          │
│  audio_bytes: 3200, 6400, 9600... 32000... 160000...            │
└──────────────────────────────────────────────────────────────────┘
                              │
                    Utente fa pausa
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  SILENZIO RILEVATO (energia < speech_energy_threshold)          │
│                                                                  │
│  silence_frames: 1, 2, 3... 10... 30... 60                      │
│                                                                  │
│  Ogni frame = ~50ms                                              │
│  60 frames = ~3 SECONDI di silenzio                             │
└──────────────────────────────────────────────────────────────────┘
                              │
            silence_frames >= 60? (SILENCE_THRESHOLD)
                              │
                              ▼
                    ✅ FINE UTTERANCE RILEVATA
```

**Tempo di attesa silenzio attuale**: ~3 secondi (60 frames × 50ms)

---

### 2.3 Fase 3: Validazione Prima di Inviare a Whisper

Prima di inviare l'audio a Whisper, il sistema verifica che ci sia abbastanza contenuto.

```
┌──────────────────────────────────────────────────────────────────┐
│  VALIDAZIONE AUDIO ACCUMULATO                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
         speech_frames >= 30?    len(buffer) >= 32000 bytes?
         (MIN_SPEECH_FRAMES)     (MIN_AUDIO_BYTES)
         (~1.5 secondi voce)     (~1 secondo audio)
                    │                   │
                    └─────────┬─────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
        ENTRAMBI SI                     ALMENO UNO NO
              │                               │
              ▼                               ▼
    ✅ INVIA A WHISPER              ❌ SCARTA (troppo corto)
                                       Log: "Audio troppo corto"
```

**Requisiti minimi attuali**:
- Almeno **30 frame con voce** = ~1.5 secondi di parlato effettivo
- Almeno **32000 bytes** = ~1 secondo di audio totale

---

### 2.4 Fase 4: Whisper Processa l'Audio

Whisper riceve il buffer completo e applica il suo VAD interno.

```
┌──────────────────────────────────────────────────────────────────┐
│  WHISPER RICEVE L'AUDIO                                         │
│  (tutto il buffer accumulato)                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  VAD INTERNO DI WHISPER (Silero VAD)                            │
│                                                                  │
│  • Cerca segmenti di voce nell'audio                            │
│  • threshold=0.3 → sensibile, cattura anche voci deboli         │
│  • Aggiunge 400ms di padding prima/dopo ogni segmento           │
│  • Segmenta se trova silenzio > 800ms                           │
│  • Combina tutti i segmenti trascritti                          │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  📝 TESTO TRASCRITTO                                            │
│  Ritorna al sistema per elaborazione LLM                        │
└──────────────────────────────────────────────────────────────────┘
```

---

### 2.5 Barge-In (Interruzione TTS)

Quando l'agent sta parlando (TTS attivo), un thread VAD separato monitora l'audio per rilevare se l'utente vuole interrompere.

```
┌──────────────────────────────────────────────────────────────────┐
│  TTS IN CORSO - Agent sta parlando                              │
└──────────────────────────────────────────────────────────────────┘
                              │
              Audio utente in arrivo
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  VAD MONITOR (Thread separato)                                  │
│                                                                  │
│  • Calcola energia audio frame                                  │
│  • Se energia > 70 (VAD_ENERGY_THRESHOLD)                       │
│    → consecutive_speech_frames++                                │
│  • Se consecutive_speech_frames >= 3                            │
│    → 🛑 BARGE-IN! Ferma TTS                                     │
└──────────────────────────────────────────────────────────────────┘
```

**Tempo per barge-in**: ~150ms (3 frame × 50ms)

---

### 2.6 TTS Cooldown

Dopo che l'agent finisce di parlare, c'è un periodo di "sordità" per evitare eco.

```
┌──────────────────────────────────────────────────────────────────┐
│  TTS TERMINATO                                                  │
│                                                                  │
│  ⏱️ Inizia cooldown di 1.5 secondi                              │
│                                                                  │
│  Durante questo tempo:                                          │
│  • L'audio in arrivo viene IGNORATO                             │
│  • Previene che l'agent "senta" la propria voce (eco)           │
└──────────────────────────────────────────────────────────────────┘
                              │
            Dopo 1.5 secondi
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  ✅ Sistema pronto ad ascoltare nuovamente                      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Parametri Configurabili

Questi parametri possono essere modificati tramite l'interfaccia web o API `/api/settings`.

### 3.1 `vad_energy_threshold`

| Proprietà | Valore |
|-----------|--------|
| **Default** | `70` |
| **Range consigliato** | 20-150 |
| **Unità** | Energia audio (media assoluta campioni) |

**Cosa fa**: Soglia di energia per rilevare la voce dell'utente **durante il TTS** (barge-in). Quando l'energia audio supera questa soglia mentre l'agent sta parlando, il TTS viene interrotto.

**Effetti**:
| Valore | Comportamento |
|--------|---------------|
| 20-40 | Molto sensibile, si interrompe facilmente (anche con rumori) |
| 50-70 | Bilanciato |
| 80-150 | Poco sensibile, l'utente deve parlare forte per interrompere |

---

### 3.2 `speech_energy_threshold`

| Proprietà | Valore |
|-----------|--------|
| **Default** | `25` |
| **Range consigliato** | 15-100 |
| **Unità** | Energia audio |

**Cosa fa**: Soglia per rilevare l'**inizio del parlato** dell'utente. Quando l'energia supera questa soglia, il sistema inizia ad accumulare audio nel buffer.

**Effetti**:
| Valore | Comportamento |
|--------|---------------|
| 15-25 | Sensibile, rileva anche voci deboli o lontane |
| 30-50 | Moderato |
| 60-100 | Poco sensibile, richiede voce forte, filtra rumori |

---

### 3.3 `silence_threshold`

| Proprietà | Valore |
|-----------|--------|
| **Default** | `60` |
| **Range consigliato** | 20-80 |
| **Unità** | Frame audio (~50ms ciascuno) |

**Cosa fa**: Numero di frame di silenzio consecutivi prima di considerare **terminata un'utterance** e inviarla a Whisper.

**Calcolo durata**:
```
Tempo silenzio = silence_threshold × 50ms

20 frames = ~1.0 secondi
40 frames = ~2.0 secondi  
60 frames = ~3.0 secondi (attuale)
80 frames = ~4.0 secondi
```

**Effetti**:
| Valore | Tempo | Comportamento |
|--------|-------|---------------|
| 20-30 | 1-1.5s | Risposta veloce ma rischia di tagliare frasi con pause |
| 40-50 | 2-2.5s | Buon compromesso |
| 60-70 | 3-3.5s | Cattura frasi complete ma risposta più lenta |
| 80+ | 4s+ | Per discorsi molto lunghi con pause |

---

### 3.4 `tts_cooldown_seconds`

| Proprietà | Valore |
|-----------|--------|
| **Default** | `1.5` |
| **Range consigliato** | 0.5-5.0 |
| **Unità** | Secondi |

**Cosa fa**: Tempo di **"sordità"** dopo la fine del TTS. Durante questo periodo l'audio viene ignorato per evitare che l'agent trascriva la propria voce (eco dagli speaker).

**Effetti**:
| Valore | Comportamento |
|--------|---------------|
| 0.5-1.0 | Reattivo ma rischio di eco |
| 1.5-2.0 | Bilanciato |
| 3.0-5.0 | Nessun eco ma pausa lunga dopo ogni risposta |

---

### 3.5 `wake_timeout_seconds`

| Proprietà | Valore |
|-----------|--------|
| **Default** | `30` |
| **Range consigliato** | 10-60 |
| **Unità** | Secondi |

**Cosa fa**: Timeout di inattività per la sessione wake word. Dopo questo tempo senza interazione, la sessione si disattiva.

**Nota**: Per le chiamate SIP questo parametro **non è rilevante** perché l'agent risponde sempre automaticamente senza wake word.

---

## 4. Parametri Hardcoded

Questi parametri sono definiti nel codice sorgente (`agent/main.py`) e richiedono modifica del file per essere cambiati.

### 4.1 `MIN_SPEECH_FRAMES`

| Proprietà | Valore |
|-----------|--------|
| **Valore** | `30` |
| **Equivalente** | ~1.5 secondi |
| **File** | `agent/main.py` |

**Cosa fa**: Numero **minimo di frame con voce** prima di considerare valida un'utterance. Se l'audio contiene meno di 30 frame "parlati", viene scartato.

**Perché**: Previene trascrizioni di rumori brevi, colpi, o suoni accidentali.

---

### 4.2 `MIN_AUDIO_BYTES`

| Proprietà | Valore |
|-----------|--------|
| **Valore** | `32000` |
| **Equivalente** | ~1 secondo |
| **File** | `agent/main.py` |

**Cosa fa**: Dimensione **minima del buffer audio** prima di inviarlo a Whisper.

**Calcolo**:
```
32000 bytes ÷ 32 bytes/ms = 1000ms = 1 secondo
(Formula: 16kHz × 16bit × mono = 32 bytes/ms)
```

**Perché**: Whisper funziona meglio con almeno 1 secondo di contesto audio.

---

### 4.3 Parametri VAD Monitor

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| `_min_speech_frames` | `3` | Frame consecutivi per triggerare barge-in (~150ms) |
| `_interrupt_cooldown` | `0.5s` | Minimo tempo tra interrupt consecutivi |
| `_audio_queue.maxsize` | `100` | Dimensione coda audio del VAD monitor |

---

## 5. Parametri Whisper VAD

Whisper ha un suo VAD interno (Silero VAD) che processa l'audio ricevuto.

### Configurazione in `agent/stt/whisper_stt.py`

```python
vad_parameters=dict(
    min_silence_duration_ms=800,  # Silenzio per segmentare
    speech_pad_ms=400,            # Padding attorno al parlato
    threshold=0.3                 # Soglia probabilità voce
)
```

### 5.1 `min_silence_duration_ms`

| Valore | `800` |
|--------|-------|

**Cosa fa**: Millisecondi di silenzio per considerare finito un **segmento interno**. Whisper può segmentare l'audio in più parti se trova pause lunghe.

---

### 5.2 `speech_pad_ms`

| Valore | `400` |
|--------|-------|

**Cosa fa**: Millisecondi di padding aggiunti **prima e dopo** ogni segmento di parlato rilevato. Evita di tagliare l'inizio o la fine delle parole.

---

### 5.3 `threshold`

| Valore | `0.3` |
|--------|-------|

**Cosa fa**: Soglia di probabilità (0.0-1.0) per classificare un frame come "voce".

| Valore | Comportamento |
|--------|---------------|
| 0.1-0.3 | Sensibile, cattura anche voci deboli |
| 0.5 | Default, bilanciato |
| 0.7-0.9 | Selettivo, solo voce chiara e forte |

---

## 6. Timeline di Esempio

### Scenario: Utente dice "Ciao, come stai?"

```
TEMPO     EVENTO                                  STATO
──────────────────────────────────────────────────────────────────
0.00s     Utente inizia: "Ciao..."               speech_frames=1
0.05s     Frame 2 con voce                       speech_frames=2
0.10s     Frame 3 con voce                       speech_frames=3
...
1.50s     Frame 30 con voce                      speech_frames=30 ✅
...
3.00s     Utente finisce: "...come stai?"        speech_frames=60
3.05s     Silenzio - frame 1                     silence_frames=1
3.10s     Silenzio - frame 2                     silence_frames=2
...
4.50s     Silenzio - frame 30                    silence_frames=30
...
6.00s     Silenzio - frame 60                    silence_frames=60 ✅
──────────────────────────────────────────────────────────────────
6.00s     🚀 VALIDAZIONE OK → Invia a Whisper
6.50s     📝 Whisper ritorna: "Ciao, come stai?"
6.60s     🤖 LLM genera risposta
7.50s     🔊 TTS inizia a parlare
12.00s    🔊 TTS termina
12.00s    ⏱️ Cooldown inizia (1.5s)
13.50s    ✅ Sistema pronto ad ascoltare
```

### Tempi Totali

| Fase | Durata |
|------|--------|
| Parlato utente | ~3 secondi |
| Attesa silenzio | ~3 secondi |
| Trascrizione Whisper | ~0.5 secondi |
| Generazione LLM | ~1 secondo |
| TTS risposta | ~4-5 secondi |
| Cooldown | ~1.5 secondi |
| **Totale ciclo** | **~13-14 secondi** |

---

## 7. Configurazioni per Scenari

### 7.1 Chiamate SIP (Telefono)

Audio telefonico spesso compresso, serve più tolleranza.

```json
{
  "vad_energy_threshold": "70",
  "speech_energy_threshold": "25",
  "silence_threshold": "60",
  "tts_cooldown_seconds": "1.5"
}
```

---

### 7.2 WebRTC (Browser con microfono)

Audio di qualità migliore, può essere più reattivo.

```json
{
  "vad_energy_threshold": "50",
  "speech_energy_threshold": "30",
  "silence_threshold": "40",
  "tts_cooldown_seconds": "1.0"
}
```

---

### 7.3 Ambiente Rumoroso

Soglie alte per filtrare rumori di fondo.

```json
{
  "vad_energy_threshold": "100",
  "speech_energy_threshold": "60",
  "silence_threshold": "50",
  "tts_cooldown_seconds": "2.0"
}
```

---

### 7.4 Risposta Veloce (rischio tagli)

Per casi dove la velocità è prioritaria.

```json
{
  "vad_energy_threshold": "50",
  "speech_energy_threshold": "20",
  "silence_threshold": "30",
  "tts_cooldown_seconds": "0.8"
}
```

---

## 8. API di Configurazione

### Leggere configurazione attuale

```bash
curl -s https://SERVER:8443/api/settings | jq '{
  vad_energy_threshold,
  speech_energy_threshold,
  silence_threshold,
  tts_cooldown_seconds,
  wake_timeout_seconds
}'
```

### Modificare parametri

```bash
curl -X POST https://SERVER:8443/api/settings \
  -H "Content-Type: application/json" \
  -d '{
    "settings": {
      "vad_energy_threshold": "70",
      "silence_threshold": "60"
    }
  }'
```

### Riavviare agent per applicare

```bash
docker-compose restart agent
```

---

## Riepilogo Parametri

| Parametro | Valore Attuale | Effetto |
|-----------|----------------|---------|
| `vad_energy_threshold` | 70 | Soglia barge-in |
| `speech_energy_threshold` | 25 | Soglia inizio parlato |
| `silence_threshold` | 60 (~3s) | Attesa fine frase |
| `tts_cooldown_seconds` | 1.5 | Pausa post-TTS |
| `MIN_SPEECH_FRAMES` | 30 (~1.5s) | Min parlato valido |
| `MIN_AUDIO_BYTES` | 32000 (~1s) | Min audio per Whisper |
| Whisper `threshold` | 0.3 | Sensibilità VAD interno |

---

*Documento aggiornato il 2026-02-05*
