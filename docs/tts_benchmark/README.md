# TTS Benchmark - SophyAI

File audio di benchmark per confrontare i TTS engine funzionanti con GPU.

## Frase di Test

> "Ciao sono Sophy il tuo agente turistico personale come posso aiutarti?"

## File Audio

| # | Engine | File | Size | Status |
|---|--------|------|------|--------|
| 1 | **Edge** | `01_edge.wav` | 245 KB | ✅ Microsoft Cloud |
| 2 | **Qwen** | `02_qwen.wav` | 253 KB | ✅ Self-hosted CUDA |
| 3 | **Piper** | `03_piper.wav` | 201 KB | ✅ Self-hosted |
| 4 | **Kokoro** | `04_kokoro.wav` | 228 KB | ✅ Self-hosted CUDA |

## Caratteristiche

| Engine | Self-Hosted | GPU | Velocità | Qualità | Note |
|--------|-------------|-----|----------|---------|------|
| **Edge** | ❌ Cloud | N/A | ⚡⚡⚡ (~0.5s) | ⭐⭐⭐⭐⭐ | Richiede internet |
| **Qwen** | ✅ | ✅ CUDA | ⚡⚡ (~10s) | ⭐⭐⭐⭐ | Alibaba, 10 lingue, speaker: ryan |
| **Piper** | ✅ | ❌ CPU | ⚡⚡⚡ (~0.6s) | ⭐⭐⭐ | Leggero, veloce, modello riccardo |
| **Kokoro** | ✅ | ✅ CUDA | ⚡⚡ (~2s) | ⭐⭐⭐⭐⭐ | Alta qualità |

## TTS Non Disponibili

| Engine | Motivo |
|--------|--------|
| Chatterbox | Problemi dipendenze Python 3.12 |
| Coqui | Problemi dipendenze Python 3.12 |
| VibeVoice | Non installato nel container |

## Come Ascoltare

I file sono in formato WAV standard (16-bit PCM, 24kHz).

### Copia su macchina locale
```bash
scp root@SERVER:/root/sophyai-live-server/docs/tts_benchmark/*.wav ./
```

---

*Generato il 2026-02-05 - Tutti i TTS verificati con hash diversi*
