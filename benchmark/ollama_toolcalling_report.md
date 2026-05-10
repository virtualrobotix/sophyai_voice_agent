# Benchmark Ollama: Tool Calling + Latenza

- Data UTC: `2026-05-09T13:16:40.429930+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Repeats per task: `2`
- Warmup per modello: `1`

## Ranking

1. **llama3.2:latest** - score=1.000, tool_policy=100.0%, avg_latency=175 ms, tok/s=412.7
2. **qwen2.5:7b** - score=0.990, tool_policy=100.0%, avg_latency=235 ms, tok/s=249.0
3. **llama3.1:8b** - score=0.982, tool_policy=100.0%, avg_latency=280 ms, tok/s=233.0
4. **mistral-small:latest** - score=0.962, tool_policy=100.0%, avg_latency=402 ms, tok/s=95.1
5. **gpt-oss:120b** - score=0.923, tool_policy=100.0%, avg_latency=639 ms, tok/s=168.0
6. **mistral-medium-3.5:latest** - score=0.781, tool_policy=100.0%, avg_latency=1489 ms, tok/s=19.1
7. **qwen3.6:35b** - score=0.650, tool_policy=100.0%, avg_latency=2272 ms, tok/s=134.6

## Dettaglio per modello

### qwen2.5:7b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **235 ms**
- P50 latency: **234 ms**
- P95 latency: **279 ms**
- Avg tokens/s: **249.0**

### llama3.1:8b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **280 ms**
- P50 latency: **252 ms**
- P95 latency: **423 ms**
- Avg tokens/s: **233.0**

### gpt-oss:120b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **639 ms**
- P50 latency: **544 ms**
- P95 latency: **1017 ms**
- Avg tokens/s: **168.0**

### mistral-small:latest
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **402 ms**
- P50 latency: **405 ms**
- P95 latency: **557 ms**
- Avg tokens/s: **95.1**

### qwen3.6:35b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **2272 ms**
- P50 latency: **1969 ms**
- P95 latency: **4121 ms**
- Avg tokens/s: **134.6**

### mistral-medium-3.5:latest
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **1489 ms**
- P50 latency: **1410 ms**
- P95 latency: **2282 ms**
- Avg tokens/s: **19.1**

### llama3.2:latest
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **175 ms**
- P50 latency: **178 ms**
- P95 latency: **211 ms**
- Avg tokens/s: **412.7**

