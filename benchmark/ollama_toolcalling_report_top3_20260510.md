# Benchmark Ollama: Tool Calling + Latenza

- Data UTC: `2026-05-10T06:29:35.044759+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Repeats per task: `5`
- Warmup per modello: `2`

## Ranking

1. **llama3.2:latest** - score=1.000, tool_policy=100.0%, avg_latency=162 ms, tok/s=420.5
2. **qwen2.5:7b** - score=0.780, tool_policy=100.0%, avg_latency=231 ms, tok/s=250.0
3. **llama3.1:8b** - score=0.650, tool_policy=100.0%, avg_latency=272 ms, tok/s=234.9

## Dettaglio per modello

### llama3.2:latest
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **162 ms**
- P50 latency: **158 ms**
- P95 latency: **198 ms**
- Avg tokens/s: **420.5**

### qwen2.5:7b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **231 ms**
- P50 latency: **224 ms**
- P95 latency: **279 ms**
- Avg tokens/s: **250.0**

### llama3.1:8b
- Tool success (solo task con tool): **100.0%**
- Tool policy success (incluso no-tool): **100.0%**
- Avg latency: **272 ms**
- P50 latency: **245 ms**
- P95 latency: **416 ms**
- Avg tokens/s: **234.9**

