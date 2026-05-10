# Report Preliminare Concorrenza (parziale)

- Generato UTC: `2026-05-10T10:29:49.349606+00:00`
- Sorgente log: `/home/user/.cursor/projects/home-user-sophyai-voice-agent/terminals/726964.txt`
- Stato: benchmark ancora in corso, dati parziali

## Ranking provvisorio (modelli completati)

1. **gemma4:31b** - max_stable_level=12, best_rps_stable=0.17, last_level=12
2. **qwen2.5:7b** - max_stable_level=7, best_rps_stable=22.28, last_level=12
3. **llama3.1:8b** - max_stable_level=6, best_rps_stable=12.01, last_level=12
4. **mistral-small:latest** - max_stable_level=5, best_rps_stable=5.05, last_level=12
5. **gpt-oss:120b** - max_stable_level=5, best_rps_stable=0.76, last_level=12
6. **gpt-oss:20b** - max_stable_level=3, best_rps_stable=0.90, last_level=12

## Stato per modello

### gemma4:31b
- Status: **completed**
- Levels osservati: **12** (max level 12)
- Max stable level: **12**
- Best stable rps: **0.17**
- Ultimo livello: level=12, success=100.0%, avg=58570 ms, p95=76375 ms, rps=0.16, stable=True

### gpt-oss:20b
- Status: **completed**
- Levels osservati: **12** (max level 12)
- Max stable level: **3**
- Best stable rps: **0.90**
- Ultimo livello: level=12, success=100.0%, avg=13901 ms, p95=19648 ms, rps=0.63, stable=False

### qwen2.5:7b
- Status: **completed**
- Levels osservati: **12** (max level 12)
- Max stable level: **7**
- Best stable rps: **22.28**
- Ultimo livello: level=12, success=100.0%, avg=406 ms, p95=554 ms, rps=23.02, stable=False

### llama3.1:8b
- Status: **completed**
- Levels osservati: **12** (max level 12)
- Max stable level: **6**
- Best stable rps: **12.01**
- Ultimo livello: level=12, success=100.0%, avg=753 ms, p95=1000 ms, rps=12.51, stable=False

### gpt-oss:120b
- Status: **completed**
- Levels osservati: **12** (max level 12)
- Max stable level: **5**
- Best stable rps: **0.76**
- Ultimo livello: level=12, success=100.0%, avg=20145 ms, p95=28742 ms, rps=0.49, stable=False

### mistral-small:latest
- Status: **completed**
- Levels osservati: **12** (max level 12)
- Max stable level: **5**
- Best stable rps: **5.05**
- Ultimo livello: level=12, success=100.0%, avg=1753 ms, p95=2286 ms, rps=5.40, stable=False

### qwen3.6:35b
- Status: **in_progress**
- Levels osservati: **11** (max level 11)
- Max stable level: **10**
- Best stable rps: **0.14**
- Ultimo livello: level=11, success=90.9%, avg=61337 ms, p95=114782 ms, rps=0.10, stable=False

