# Report benchmark complessivo TTS + STT

Data esecuzione: 2026-05-10 07:56:52
TTS endpoint: `http://127.0.0.1:8092`
STT endpoint: `http://127.0.0.1:8091`
Run per frase: `2`

## Contesto test
- Le metriche TTS usano endpoint `/synthesize` e misurano latenza end-to-end.
- Le metriche STT usano audio WAV/PCM generato in fase benchmark e endpoint Whisper.
- Accuratezza STT misurata con WER (Word Error Rate): più basso = migliore.

## Risultati TTS
| Engine | Successo | Latenza media | P95 | Audio medio | RTF medio |
|---|---:|---:|---:|---:|---:|
| chatterbox | 100% | 430 ms | 504 ms | 7.25 s | 0.059 |
| coqui | 100% | 662 ms | 710 ms | 7.16 s | 0.093 |
| edge | 100% | 557 ms | 810 ms | 7.25 s | 0.077 |
| kokoro | 100% | 3125 ms | 3335 ms | 8.85 s | 0.355 |
| piper | 100% | 1146 ms | 1191 ms | 6.18 s | 0.186 |
| qwen | 100% | 11992 ms | 14251 ms | 7.40 s | 1.622 |
| vibevoice | 100% | 8309 ms | 9295 ms | 7.33 s | 1.134 |

## Risultati STT
| Modalità | Successo | Latenza media | P95 | WER medio |
|---|---:|---:|---:|---:|
| transcribe_raw | 100% | 510 ms | 590 ms | 0.026 |
| transcribe_wav | 100% | 510 ms | 659 ms | 0.026 |

## Note operative
- Il benchmark non altera la configurazione runtime del sistema.
- In caso di failure, verificare log e disponibilità modelli nei rispettivi container/server.
- Per confronti nel tempo usare sempre stesso set frasi e stesso numero di run.

## Frasi usate
1. Ciao, questa è una prova di sintesi vocale per misurare prestazioni e qualità dei motori disponibili.
2. SophyAI deve rispondere rapidamente mantenendo una voce naturale e comprensibile durante una conversazione.
3. Questo benchmark confronta latenza, stabilità e accuratezza della pipeline speech to text e text to speech.
