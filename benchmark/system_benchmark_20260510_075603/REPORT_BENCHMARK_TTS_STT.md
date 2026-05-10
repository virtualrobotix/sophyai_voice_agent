# Report benchmark complessivo TTS + STT

Data esecuzione: 2026-05-10 07:55:53
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
| edge | 100% | 641 ms | 788 ms | 7.25 s | 0.088 |

## Risultati STT
| Modalità | Successo | Latenza media | P95 | WER medio |
|---|---:|---:|---:|---:|
| transcribe_raw | 100% | 482 ms | 494 ms | 0.026 |
| transcribe_wav | 100% | 498 ms | 563 ms | 0.026 |

## Note operative
- Il benchmark non altera la configurazione runtime del sistema.
- In caso di failure, verificare log e disponibilità modelli nei rispettivi container/server.
- Per confronti nel tempo usare sempre stesso set frasi e stesso numero di run.

## Frasi usate
1. Ciao, questa è una prova di sintesi vocale per misurare prestazioni e qualità dei motori disponibili.
2. SophyAI deve rispondere rapidamente mantenendo una voce naturale e comprensibile durante una conversazione.
3. Questo benchmark confronta latenza, stabilità e accuratezza della pipeline speech to text e text to speech.
