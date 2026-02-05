# TTS Benchmark - SophyAI

Audio di benchmark per tutti i TTS engine funzionanti con GPU.

## Frase di Test

> "Ciao sono Sophy il tuo agente turistico personale come posso aiutarti?"

## File Audio

| # | File | Engine | Sample Rate | Dimensione |
|---|------|--------|-------------|------------|
| 1 | 01_edge.wav | Edge TTS (Isabella) | 24000 Hz | 227 KB |
| 2 | 02_qwen.wav | Qwen TTS (Ryan) | 24000 Hz | 195 KB |
| 3 | 03_piper.wav | Piper (Riccardo IT) | 22050 Hz | 119 KB |
| 4 | 04_kokoro.wav | Kokoro (Sara IT) | 24000 Hz | 223 KB |
| 5 | 05_chatterbox.wav | Chatterbox | 24000 Hz | 231 KB |
| 6 | 06_coqui.wav | Coqui (Mai Female IT) | 22050 Hz | 146 KB |

## Performance (GPU NVIDIA)

| Engine | Caricamento | Generazione | Note |
|--------|-------------|-------------|------|
| Edge TTS | - | ~0.4s | Cloud-based (Microsoft), richiede internet |
| Qwen TTS | ~4.2s | ~4.5s | Alibaba, self-hosted CUDA, 10+ lingue |
| Piper | - | ~0.1s | Locale, molto veloce, CPU |
| Kokoro | ~1.2s | ~0.7s | Locale, buona qualità |
| Chatterbox | ~6.8s | ~2.0s | Locale, alta qualità, voice cloning |
| Coqui | ~0.3s | ~0.6s | Locale, italiano nativo |

## Note

- **VibeVoice**: Non funzionante (bug interno incompatibilità transformers)
- Generati il 5 Febbraio 2026
