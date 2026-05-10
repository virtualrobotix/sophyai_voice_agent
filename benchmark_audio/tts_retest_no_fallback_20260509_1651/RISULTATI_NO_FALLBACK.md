# Test TTS senza fallback

Data: 2026-05-09 14:52:08
Server: http://127.0.0.1:8092

Regola applicata: se `X-Engine` diverso da engine richiesto => test SCARTATO e nessun file audio salvato.

## Standard (/synthesize)
- `edge`: OK (615 ms), file `standard_edge.wav`
- `coqui`: SCARTATO/ERRORE (500), {"detail":"500: Engine TTS sconosciuto: coqui"}
- `piper`: OK (1163 ms), file `standard_piper.wav`
- `kokoro`: OK (2967 ms), file `standard_kokoro.wav`
- `qwen`: OK (9082 ms), file `standard_qwen.wav`
- `chatterbox`: SCARTATO/ERRORE (500), {"detail":"503: Engine requested 'chatterbox' unavailable: 500: 'NoneType' object is not callable"}
- `vibevoice`: SCARTATO/ERRORE (500), {"detail":"503: Engine requested 'vibevoice' unavailable: Nessuna voce disponibile"}

## Stream (/synthesize lettura chunk)
- `edge`: OK (TTFB 428 ms, totale 428 ms), file `stream_edge.wav`
- `coqui`: SCARTATO/ERRORE (500), {"detail":"500: Engine TTS sconosciuto: coqui"}
- `piper`: OK (TTFB 1187 ms, totale 1187 ms), file `stream_piper.wav`
- `kokoro`: OK (TTFB 2994 ms, totale 2995 ms), file `stream_kokoro.wav`
- `qwen`: OK (TTFB 10397 ms, totale 10398 ms), file `stream_qwen.wav`
- `chatterbox`: SCARTATO/ERRORE (500), {"detail":"503: Engine requested 'chatterbox' unavailable: 500: 'NoneType' object is not callable"}
- `vibevoice`: SCARTATO/ERRORE (500), {"detail":"503: Engine requested 'vibevoice' unavailable: Nessuna voce disponibile"}

## File
- JSON: `benchmark_audio/tts_retest_no_fallback_20260509_1651/results.json`
- Cartella output: `benchmark_audio/tts_retest_no_fallback_20260509_1651`
