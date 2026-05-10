# Risultati test TTS (standard + stream)

Data test: 2026-05-09 14:32:48

Testo usato: "Ciao, questa è una prova completa di sintesi vocale per verificare prestazioni e qualità dei motori TTS in modalità standard e stream."

## Stato endpoint
- API /api/tts/engines: HTTP 200
- API /api/tts/current: HTTP 200

## Test standard (/api/tts/test)
- `edge`: OK, 1284 ms, X-Engine=edge, file `standard_edge.wav`
- `coqui`: OK, 361 ms, X-Engine=coqui, file `standard_coqui.wav`
- `piper`: OK, 1149 ms, X-Engine=piper, file `standard_piper.wav`
- `kokoro`: OK, 1605 ms, X-Engine=kokoro, file `standard_kokoro.wav`
- `qwen`: OK, 14171 ms, X-Engine=qwen, file `standard_qwen.wav`
- `chatterbox`: ERRORE HTTP 500, 5834 ms
- `vibevoice`: OK, 398 ms, X-Engine=vibevoice, file `standard_vibevoice.wav`

## Test stream (/synthesize, lettura a chunk)
- `edge`: OK, TTFB 433 ms, totale 433 ms, chunk 114, X-Engine=edge, file `stream_edge.wav`
- `coqui`: OK, TTFB 442 ms, totale 443 ms, chunk 114, X-Engine=coqui, file `stream_coqui.wav`
- `piper`: OK, TTFB 1185 ms, totale 1186 ms, chunk 88, X-Engine=piper, file `stream_piper.wav`
- `kokoro`: OK, TTFB 1967 ms, totale 1967 ms, chunk 114, X-Engine=edge, file `stream_kokoro.wav`
- `qwen`: OK, TTFB 9549 ms, totale 9550 ms, chunk 85, X-Engine=qwen, file `stream_qwen.wav`
- `chatterbox`: ERRORE HTTP 500, 5552 ms
- `vibevoice`: OK, TTFB 460 ms, totale 460 ms, chunk 114, X-Engine=edge, file `stream_vibevoice.wav`

## Audio identici (possibile fallback)
- Standard: [['edge', 'coqui'], ['kokoro', 'vibevoice']]
- Stream: [['edge', 'coqui', 'kokoro', 'vibevoice']]

## File output
- Report JSON: `benchmark_audio/tts_retest_20260509_1632/results.json`
- Cartella audio: `benchmark_audio/tts_retest_20260509_1632`
