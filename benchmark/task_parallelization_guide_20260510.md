# Guida Parallelizzazione Task (basata su test concorrenza)

## Contesto test

- Modello testato: `qwen3.6:35b`
- Test eseguito con livelli di concorrenza da 1 a 20.
- File risultati:
  - `benchmark/ollama_concurrency_qwen36_5_10_20_20260510.json`
  - `benchmark/ollama_concurrency_qwen36_5_10_20_20260510.md`

Valori chiave richiesti:
- **Level 5**: success `100.0%`, avg `20613 ms`, p95 `36845 ms`, rps `0.14`
- **Level 10**: success `100.0%`, avg `44431 ms`, p95 `88993 ms`, rps `0.11`
- **Level 20**: success `65.0%`, avg `58158 ms`, p95 `111978 ms`, rps `0.11`

## Lettura operativa dei risultati

- La massima concorrenza "stabile" del benchmark e bassa (stabile solo ai livelli iniziali).
- Oltre il livello 10 la latenza esplode e il tasso successo degrada.
- A level 20 il sistema non e affidabile per produzione (success 65%).

## Come rendere i task paralleli in modo sicuro

### 1) Worker pool con coda

- Implementa un pool fisso di worker (non creare thread illimitati).
- Tutte le richieste entrano in una coda FIFO.
- Ogni worker prende il prossimo task solo quando il precedente termina.

### 2) Concorrenza dinamica (auto-throttle)

- Parti con `concurrency=5`.
- Ogni finestra di controllo (es. 1-2 minuti) misura:
  - success rate
  - avg latency
  - p95 latency
- Regole pratiche:
  - se `success >= 99%` e `p95` sotto soglia: aumenta di `+1`
  - se `success < 97%` o `p95` supera soglia: riduci di `-2`

### 3) Soglie di protezione realtime

- Definisci una soglia hard, per esempio:
  - `avg <= 800 ms`
  - `p95 <= 1500 ms`
- Se superata per N finestre consecutive:
  - blocca la crescita
  - riduci subito la concorrenza
  - opzionalmente fai circuit-breaker temporaneo

### 4) Separazione classi di task

- Non mischiare in un unico pool task realtime e task batch.
- Usa almeno due code:
  - `realtime_queue` (priorita alta, timeout stretti)
  - `batch_queue` (priorita bassa, retry piu permissivi)

### 5) Retry disciplinato

- Retry solo su errori transienti (timeout/rete), non su errori logici.
- Exponential backoff con jitter (es. 1s, 2s, 4s).
- Dead-letter queue per task falliti oltre il limite.

## Raccomandazione pratica immediata

- Per questo modello e questa macchina:
  - **concurrency iniziale consigliata: 5**
  - **range operativo prudente: 5-8**
  - evitare configurazioni >= 10 per carichi critici realtime

- Se serve throughput alto, valuta:
  - modello piu leggero per realtime
  - sharding su piu istanze Ollama
  - bilanciamento richieste per modello/coda.
