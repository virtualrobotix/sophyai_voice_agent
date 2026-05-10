# Benchmark Ollama: Concorrenza Massima per Modello

- Data UTC: `2026-05-10T13:38:23.055021+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Livello massimo testato: `20`
- Round per livello: `1`
- Soglia successo minima: `100.0%`
- Fattore degrado latenza consentito: `3.0x`
- Soglia realtime avg/p95: `800 ms / 1500 ms`
- Early-stop su degrado realtime: `False`

## Ranking concorrenza stabile

1. **qwen3.6:35b** - max_stable=3, recommended=2, best_throughput=0.11 req/s, baseline_avg=11388 ms

## Dettaglio per modello

### qwen3.6:35b
- Max stable concurrency: **3**
- Recommended concurrency: **2**
- Best stable throughput: **0.11 req/s**
- Baseline latency: **avg 11388 ms, p95 11388 ms**
- Tested levels: **20**
- Early-stop realtime: **False**
- Note stabilita:
  - livello=4: latenza oltre limite (avg 22844>34163 o p95 34398>34163)
  - livello=5: latenza oltre limite (avg 20613>34163 o p95 36845>34163)
  - livello=6: latenza oltre limite (avg 30841>34163 o p95 48846>34163)
  - livello=7: latenza oltre limite (avg 25875>34163 o p95 43722>34163)
  - livello=8: latenza oltre limite (avg 42088>34163 o p95 72042>34163)

#### Livelli testati
- level=1, req=1, success=100.0%, avg=11388 ms, p95=11388 ms, rps=0.09, stable=True, realtime=False
- level=2, req=2, success=100.0%, avg=18934 ms, p95=24796 ms, rps=0.08, stable=True, realtime=False
- level=3, req=3, success=100.0%, avg=20055 ms, p95=26224 ms, rps=0.11, stable=True, realtime=False
- level=4, req=4, success=100.0%, avg=22844 ms, p95=34398 ms, rps=0.12, stable=False, realtime=False
- level=5, req=5, success=100.0%, avg=20613 ms, p95=36845 ms, rps=0.14, stable=False, realtime=False
- level=6, req=6, success=100.0%, avg=30841 ms, p95=48846 ms, rps=0.12, stable=False, realtime=False
- level=7, req=7, success=100.0%, avg=25875 ms, p95=43722 ms, rps=0.16, stable=False, realtime=False
- level=8, req=8, success=100.0%, avg=42088 ms, p95=72042 ms, rps=0.11, stable=False, realtime=False
- level=9, req=9, success=100.0%, avg=41484 ms, p95=82119 ms, rps=0.11, stable=False, realtime=False
- level=10, req=10, success=100.0%, avg=44431 ms, p95=88993 ms, rps=0.11, stable=False, realtime=False
- level=11, req=11, success=100.0%, avg=43185 ms, p95=74700 ms, rps=0.15, stable=False, realtime=False
- level=12, req=12, success=100.0%, avg=61844 ms, p95=105500 ms, rps=0.11, stable=False, realtime=False
- level=13, req=13, success=69.2%, avg=69626 ms, p95=113889 ms, rps=0.07, stable=False, realtime=False
- level=14, req=14, success=85.7%, avg=69102 ms, p95=118343 ms, rps=0.10, stable=False, realtime=False
- level=15, req=15, success=46.7%, avg=36271 ms, p95=56602 ms, rps=0.06, stable=False, realtime=False
- level=16, req=16, success=0.0%, avg=n/d, p95=n/d, rps=0.00, stable=False, realtime=False
- level=17, req=17, success=76.5%, avg=58087 ms, p95=115221 ms, rps=0.11, stable=False, realtime=False
- level=18, req=18, success=55.6%, avg=63499 ms, p95=118241 ms, rps=0.08, stable=False, realtime=False
- level=19, req=19, success=63.2%, avg=58179 ms, p95=114057 ms, rps=0.10, stable=False, realtime=False
- level=20, req=20, success=65.0%, avg=58158 ms, p95=111978 ms, rps=0.11, stable=False, realtime=False

