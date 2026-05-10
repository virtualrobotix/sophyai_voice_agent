# Benchmark Ollama: Concorrenza Massima per Modello

- Data UTC: `2026-05-10T09:38:21.829238+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Livello massimo testato: `4`
- Round per livello: `1`
- Soglia successo minima: `100.0%`
- Fattore degrado latenza consentito: `3.0x`
- Soglia realtime avg/p95: `100 ms / 120 ms`
- Early-stop su degrado realtime: `True`

## Ranking concorrenza stabile

1. **llama3.2:latest** - max_stable=1, recommended=1, best_throughput=6.66 req/s, baseline_avg=149 ms

## Dettaglio per modello

### llama3.2:latest
- Max stable concurrency: **1**
- Recommended concurrency: **1**
- Best stable throughput: **6.66 req/s**
- Baseline latency: **avg 149 ms, p95 149 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg 149 ms > 100 ms o p95 149 ms > 120 ms)**
- Note stabilita:
  - livello=1: fuori soglia realtime (avg 149 ms > 100 ms o p95 149 ms > 120 ms)

#### Livelli testati
- level=1, req=1, success=100.0%, avg=149 ms, p95=149 ms, rps=6.66, stable=True, realtime=False

