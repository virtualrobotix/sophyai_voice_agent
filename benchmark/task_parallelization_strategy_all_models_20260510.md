# Strategia Parallelizzazione Task (all-model run)

## Run usato per la valutazione

- Report: `benchmark/ollama_concurrency_report_20260510_final.md`
- Retry all-model: `benchmark/ollama_concurrency_report_20260510.md`
- All-model rerun (soglie realtime): `benchmark/ollama_concurrency_all_models_20260510.md`
- Deep test su modello stabile: `benchmark/ollama_concurrency_qwen36_5_10_20_20260510.md`

## Esito sintetico

- Nel run all-model con soglie realtime, la maggior parte dei modelli ha avuto timeout gia al livello 1.
- L unico modello che ha prodotto risposte in quel run e `qwen3.6:35b`.
- Nel test approfondito su `qwen3.6:35b` (livelli fino a 20):
  - level 5: success 100%, avg 20613 ms, p95 36845 ms
  - level 10: success 100%, avg 44431 ms, p95 88993 ms
  - level 20: success 65%, avg 58158 ms, p95 111978 ms

Conclusione pratica: la macchina non regge bene la concorrenza multi-modello in questo stato; serve orchestrazione controllata e routing.

## Come rendere i task paralleli in modo affidabile

### 1) Scheduler centrale a code separate

- Usa una coda per task realtime e una per task batch.
- Esegui i task realtime con priorita alta e timeout stretti.
- Limita il batch quando il realtime degrada.

### 2) Worker pool per modello (non un pool globale)

- Mantieni un pool dedicato per ogni modello.
- Imposta concorrenza massima diversa per modello (cap per modello).
- Evita che un modello lento saturi tutti i worker.

### 3) Cap di concorrenza iniziali consigliati

- `qwen3.6:35b`: partire da `1-2` (incrementi molto cauti).
- Modelli che oggi timeoutano al level 1: tenere `1` finche non passa un health-check.
- Non usare `>=5` su modelli lenti senza autoscaling e controllo p95.

### 4) Health-check prima di accettare parallelismo

- Prima di mettere un modello nel pool:
  - 3 richieste secche consecutive senza timeout
  - p95 sotto soglia operativa definita
- Se fallisce, modello in stato `degraded` (solo fallback manuale).

### 5) Auto-throttle in tempo reale

- Ogni 30-60 secondi misura `success_rate`, `avg`, `p95`, `queue_wait`.
- Regole:
  - se success >= 99% e p95 sotto target: `+1` worker
  - se success < 97% o p95 fuori target: `-1/-2` worker
  - se timeout ripetuti: blocca crescita e apri circuit-breaker.

### 6) Circuit breaker per modello

- Apri il breaker dopo N timeout consecutivi (es. 3).
- Durante apertura, instrada su fallback model.
- Retry del modello originale solo dopo cooldown.

### 7) Routing consigliato

- Realtime: modello veloce/stabile (da validare con health-check).
- Task pesanti o non urgenti: modello lento in coda batch dedicata.
- Non mischiare realtime e batch sullo stesso budget di concorrenza.

## Piano operativo consigliato (subito applicabile)

- Step 1: attiva scheduler con pool per modello e cap iniziale `1`.
- Step 2: abilita auto-throttle e circuit breaker.
- Step 3: fai canary incrementale (1 -> 2 -> 3) su ogni modello.
- Step 4: promuovi solo modelli con success stabile e p95 entro SLO.
