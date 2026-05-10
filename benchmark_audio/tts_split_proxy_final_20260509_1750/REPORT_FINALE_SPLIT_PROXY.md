# Report finale TTS split + proxy

Data: 2026-05-09 15:47:48
Proxy: http://127.0.0.1:8092

Regola: fallback non ammesso (X-Engine deve coincidere con engine richiesto).

## Standard
- `edge`: OK (1524 ms)
- `piper`: OK (1771 ms)
- `kokoro`: OK (11500 ms)
- `qwen`: OK (61540 ms)
- `coqui`: OK (504 ms)
- `chatterbox`: OK (563 ms)
- `vibevoice`: OK (5909 ms)

## Stream
- `edge`: OK (TTFB 411 ms, totale 411 ms)
- `piper`: OK (TTFB 1131 ms, totale 1131 ms)
- `kokoro`: OK (TTFB 3040 ms, totale 3040 ms)
- `qwen`: OK (TTFB 11195 ms, totale 11195 ms)
- `coqui`: OK (TTFB 672 ms, totale 672 ms)
- `chatterbox`: OK (TTFB 481 ms, totale 481 ms)
- `vibevoice`: OK (TTFB 6201 ms, totale 6201 ms)

## File
- JSON: `benchmark_audio/tts_split_proxy_final_20260509_1750/results.json`
- Cartella audio: `benchmark_audio/tts_split_proxy_final_20260509_1750`
