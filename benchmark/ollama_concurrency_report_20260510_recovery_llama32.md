# Benchmark Ollama: Concorrenza Massima per Modello

- Data UTC: `2026-05-10T10:46:48.273593+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Livello massimo testato: `12`
- Round per livello: `2`
- Soglia successo minima: `100.0%`
- Fattore degrado latenza consentito: `3.0x`
- Soglia realtime avg/p95: `800 ms / 1500 ms`
- Early-stop su degrado realtime: `True`

## Ranking concorrenza stabile

1. **llama3.2:latest** - max_stable=0, recommended=1, best_throughput=0.00 req/s, baseline_avg=1 ms

## Dettaglio per modello

### llama3.2:latest
- Max stable concurrency: **0**
- Recommended concurrency: **1**
- Best stable throughput: **0.00 req/s**
- Baseline latency: **avg 1 ms, p95 1 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg n/d > 800 ms o p95 n/d > 1500 ms)**
- Note stabilita:
  - livello=1: success_rate 0.0% sotto soglia 100.0%
  - livello=1: fuori soglia realtime (avg n/d > 800 ms o p95 n/d > 1500 ms)

#### Livelli testati
- level=1, req=2, success=0.0%, avg=n/d, p95=n/d, rps=0.00, stable=False, realtime=False

