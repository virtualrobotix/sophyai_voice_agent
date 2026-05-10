# Benchmark Ollama: Concorrenza Massima per Modello

- Data UTC: `2026-05-10T13:58:09.986912+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Livello massimo testato: `20`
- Round per livello: `1`
- Soglia successo minima: `100.0%`
- Fattore degrado latenza consentito: `3.0x`
- Soglia realtime avg/p95: `5000 ms / 10000 ms`
- Early-stop su degrado realtime: `True`

## Ranking concorrenza stabile

1. **qwen3.6:35b** - max_stable=1, recommended=1, best_throughput=0.09 req/s, baseline_avg=11148 ms
2. **gemma4:31b** - max_stable=0, recommended=1, best_throughput=0.00 req/s, baseline_avg=1 ms
3. **gpt-oss:20b** - max_stable=0, recommended=1, best_throughput=0.00 req/s, baseline_avg=1 ms
4. **qwen2.5:7b** - max_stable=0, recommended=1, best_throughput=0.00 req/s, baseline_avg=1 ms
5. **llama3.1:8b** - max_stable=0, recommended=1, best_throughput=0.00 req/s, baseline_avg=1 ms
6. **gpt-oss:120b** - max_stable=0, recommended=1, best_throughput=0.00 req/s, baseline_avg=1 ms
7. **mistral-small:latest** - max_stable=0, recommended=1, best_throughput=0.00 req/s, baseline_avg=1 ms
8. **mistral-medium-3.5:latest** - max_stable=0, recommended=1, best_throughput=0.00 req/s, baseline_avg=1 ms
9. **llama3.2:latest** - max_stable=0, recommended=1, best_throughput=0.00 req/s, baseline_avg=1 ms

## Dettaglio per modello

### gemma4:31b
- Max stable concurrency: **0**
- Recommended concurrency: **1**
- Best stable throughput: **0.00 req/s**
- Baseline latency: **avg 1 ms, p95 1 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)**
- Note stabilita:
  - livello=1: success_rate 0.0% sotto soglia 100.0%
  - livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)

#### Livelli testati
- level=1, req=1, success=0.0%, avg=n/d, p95=n/d, rps=0.00, stable=False, realtime=False

### gpt-oss:20b
- Max stable concurrency: **0**
- Recommended concurrency: **1**
- Best stable throughput: **0.00 req/s**
- Baseline latency: **avg 1 ms, p95 1 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)**
- Note stabilita:
  - livello=1: success_rate 0.0% sotto soglia 100.0%
  - livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)

#### Livelli testati
- level=1, req=1, success=0.0%, avg=n/d, p95=n/d, rps=0.00, stable=False, realtime=False

### qwen2.5:7b
- Max stable concurrency: **0**
- Recommended concurrency: **1**
- Best stable throughput: **0.00 req/s**
- Baseline latency: **avg 1 ms, p95 1 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)**
- Note stabilita:
  - livello=1: success_rate 0.0% sotto soglia 100.0%
  - livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)

#### Livelli testati
- level=1, req=1, success=0.0%, avg=n/d, p95=n/d, rps=0.00, stable=False, realtime=False

### llama3.1:8b
- Max stable concurrency: **0**
- Recommended concurrency: **1**
- Best stable throughput: **0.00 req/s**
- Baseline latency: **avg 1 ms, p95 1 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)**
- Note stabilita:
  - livello=1: success_rate 0.0% sotto soglia 100.0%
  - livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)

#### Livelli testati
- level=1, req=1, success=0.0%, avg=n/d, p95=n/d, rps=0.00, stable=False, realtime=False

### gpt-oss:120b
- Max stable concurrency: **0**
- Recommended concurrency: **1**
- Best stable throughput: **0.00 req/s**
- Baseline latency: **avg 1 ms, p95 1 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)**
- Note stabilita:
  - livello=1: success_rate 0.0% sotto soglia 100.0%
  - livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)

#### Livelli testati
- level=1, req=1, success=0.0%, avg=n/d, p95=n/d, rps=0.00, stable=False, realtime=False

### mistral-small:latest
- Max stable concurrency: **0**
- Recommended concurrency: **1**
- Best stable throughput: **0.00 req/s**
- Baseline latency: **avg 1 ms, p95 1 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)**
- Note stabilita:
  - livello=1: success_rate 0.0% sotto soglia 100.0%
  - livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)

#### Livelli testati
- level=1, req=1, success=0.0%, avg=n/d, p95=n/d, rps=0.00, stable=False, realtime=False

### qwen3.6:35b
- Max stable concurrency: **1**
- Recommended concurrency: **1**
- Best stable throughput: **0.09 req/s**
- Baseline latency: **avg 11148 ms, p95 11148 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg 11148 ms > 5000 ms o p95 11148 ms > 10000 ms)**
- Note stabilita:
  - livello=1: fuori soglia realtime (avg 11148 ms > 5000 ms o p95 11148 ms > 10000 ms)

#### Livelli testati
- level=1, req=1, success=100.0%, avg=11148 ms, p95=11148 ms, rps=0.09, stable=True, realtime=False

### mistral-medium-3.5:latest
- Max stable concurrency: **0**
- Recommended concurrency: **1**
- Best stable throughput: **0.00 req/s**
- Baseline latency: **avg 1 ms, p95 1 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)**
- Note stabilita:
  - livello=1: success_rate 0.0% sotto soglia 100.0%
  - livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)

#### Livelli testati
- level=1, req=1, success=0.0%, avg=n/d, p95=n/d, rps=0.00, stable=False, realtime=False

### llama3.2:latest
- Max stable concurrency: **0**
- Recommended concurrency: **1**
- Best stable throughput: **0.00 req/s**
- Baseline latency: **avg 1 ms, p95 1 ms**
- Tested levels: **1**
- Early-stop realtime: **True**
- Stop reason: **livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)**
- Note stabilita:
  - livello=1: success_rate 0.0% sotto soglia 100.0%
  - livello=1: fuori soglia realtime (avg n/d > 5000 ms o p95 n/d > 10000 ms)

#### Livelli testati
- level=1, req=1, success=0.0%, avg=n/d, p95=n/d, rps=0.00, stable=False, realtime=False

