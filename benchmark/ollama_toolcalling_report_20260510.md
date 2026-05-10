# Benchmark Ollama: Tool Calling + Latenza

- Data UTC: `2026-05-10T06:29:01.804673+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Repeats per task: `2`
- Warmup per modello: `1`

## Ranking

1. **llama3.2:latest** - score=1.000, tool_policy=100.0%, avg_latency=169 ms, tok/s=419.6
2. **qwen2.5:7b** - score=0.991, tool_policy=100.0%, avg_latency=229 ms, tok/s=249.8
3. **llama3.1:8b** - score=0.984, tool_policy=100.0%, avg_latency=278 ms, tok/s=233.1
4. **mistral-small:latest** - score=0.967, tool_policy=100.0%, avg_latency=398 ms, tok/s=95.0
5. **gpt-oss:20b** - score=0.954, tool_policy=100.0%, avg_latency=491 ms, tok/s=241.2
6. **gpt-oss:120b** - score=0.932, tool_policy=100.0%, avg_latency=642 ms, tok/s=167.2
7. **mistral-medium-3.5:latest** - score=0.811, tool_policy=100.0%, avg_latency=1484 ms, tok/s=19.1
8. **qwen3.6:35b** - score=0.714, tool_policy=100.0%, avg_latency=2157 ms, tok/s=139.5
9. **gemma4:31b** - score=0.650, tool_policy=100.0%, avg_latency=2599 ms, tok/s=59.6

## Dettaglio per modello

### gemma4:31b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **2599 ms**
- P50 latency: **1995 ms**
- P95 latency: **5495 ms**
- Avg tokens/s: **59.6**

### gpt-oss:20b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **491 ms**
- P50 latency: **453 ms**
- P95 latency: **762 ms**
- Avg tokens/s: **241.2**

### qwen2.5:7b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **229 ms**
- P50 latency: **224 ms**
- P95 latency: **270 ms**
- Avg tokens/s: **249.8**

### llama3.1:8b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **278 ms**
- P50 latency: **248 ms**
- P95 latency: **421 ms**
- Avg tokens/s: **233.1**

### gpt-oss:120b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **642 ms**
- P50 latency: **539 ms**
- P95 latency: **1016 ms**
- Avg tokens/s: **167.2**

### mistral-small:latest
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **398 ms**
- P50 latency: **406 ms**
- P95 latency: **554 ms**
- Avg tokens/s: **95.0**

### qwen3.6:35b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **2157 ms**
- P50 latency: **1896 ms**
- P95 latency: **3883 ms**
- Avg tokens/s: **139.5**

### mistral-medium-3.5:latest
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **1484 ms**
- P50 latency: **1405 ms**
- P95 latency: **2274 ms**
- Avg tokens/s: **19.1**

### llama3.2:latest
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **169 ms**
- P50 latency: **172 ms**
- P95 latency: **196 ms**
- Avg tokens/s: **419.6**

