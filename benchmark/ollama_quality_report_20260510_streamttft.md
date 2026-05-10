# Benchmark Ollama: Qualita Risposte Receptionist

- Data UTC: `2026-05-10T08:05:52.253699+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Domande standard: `50`
- Repeats per domanda: `1`
- Judge model: `llama3.2:latest`

## Ranking qualita

1. **gemma4:31b** - quality_avg=34.5, policy=0.0%, hallucination_risk=0.0%, avg_latency=5982 ms, avg_ttft=5584 ms
2. **gpt-oss:20b** - quality_avg=34.5, policy=0.0%, hallucination_risk=0.0%, avg_latency=1086 ms, avg_ttft=962 ms
3. **qwen2.5:7b** - quality_avg=34.5, policy=0.0%, hallucination_risk=0.0%, avg_latency=181 ms, avg_ttft=75 ms
4. **llama3.1:8b** - quality_avg=34.5, policy=0.0%, hallucination_risk=0.0%, avg_latency=276 ms, avg_ttft=88 ms
5. **gpt-oss:120b** - quality_avg=34.5, policy=0.0%, hallucination_risk=0.0%, avg_latency=1310 ms, avg_ttft=1123 ms
6. **mistral-small:latest** - quality_avg=34.5, policy=0.0%, hallucination_risk=0.0%, avg_latency=426 ms, avg_ttft=97 ms
7. **qwen3.6:35b** - quality_avg=34.5, policy=0.0%, hallucination_risk=0.0%, avg_latency=4959 ms, avg_ttft=4756 ms
8. **mistral-medium-3.5:latest** - quality_avg=34.5, policy=0.0%, hallucination_risk=0.0%, avg_latency=14794 ms, avg_ttft=13685 ms
9. **llama3.2:latest** - quality_avg=34.5, policy=0.0%, hallucination_risk=0.0%, avg_latency=207 ms, avg_ttft=96 ms

## Compatibilita realtime (latenza)

- **qwen2.5:7b**: avg=181 ms, p95=253 ms, avg_ttft=75 ms, p95_ttft=86 ms, profilo=realtime_ready, compatibile=True. Motivo: Tempo prima risposta e latenza totale entro target realtime.
- **llama3.2:latest**: avg=207 ms, p95=292 ms, avg_ttft=96 ms, p95_ttft=104 ms, profilo=realtime_ready, compatibile=True. Motivo: Tempo prima risposta e latenza totale entro target realtime.
- **llama3.1:8b**: avg=276 ms, p95=414 ms, avg_ttft=88 ms, p95_ttft=95 ms, profilo=realtime_ready, compatibile=True. Motivo: Tempo prima risposta e latenza totale entro target realtime.
- **mistral-small:latest**: avg=426 ms, p95=608 ms, avg_ttft=97 ms, p95_ttft=102 ms, profilo=realtime_ready, compatibile=True. Motivo: Tempo prima risposta e latenza totale entro target realtime.
- **gpt-oss:20b**: avg=1086 ms, p95=1707 ms, avg_ttft=962 ms, p95_ttft=1526 ms, profilo=not_realtime, compatibile=False. Motivo: Tempo prima risposta o latenza totale troppo alti per realtime naturale.
- **gpt-oss:120b**: avg=1310 ms, p95=2353 ms, avg_ttft=1123 ms, p95_ttft=2113 ms, profilo=not_realtime, compatibile=False. Motivo: Tempo prima risposta o latenza totale troppo alti per realtime naturale.
- **qwen3.6:35b**: avg=4959 ms, p95=9880 ms, avg_ttft=4756 ms, p95_ttft=9667 ms, profilo=not_realtime, compatibile=False. Motivo: Tempo prima risposta o latenza totale troppo alti per realtime naturale.
- **gemma4:31b**: avg=5982 ms, p95=7486 ms, avg_ttft=5584 ms, p95_ttft=7105 ms, profilo=not_realtime, compatibile=False. Motivo: Tempo prima risposta o latenza totale troppo alti per realtime naturale.
- **mistral-medium-3.5:latest**: avg=14794 ms, p95=19855 ms, avg_ttft=13685 ms, p95_ttft=18359 ms, profilo=not_realtime, compatibile=False. Motivo: Tempo prima risposta o latenza totale troppo alti per realtime naturale.

## Ranking combinato qualita e latenza

1. **qwen2.5:7b** - combined=0.541, quality_avg=34.5, avg_latency=181 ms, avg_ttft=75 ms, policy=0.0%
2. **llama3.2:latest** - combined=0.541, quality_avg=34.5, avg_latency=207 ms, avg_ttft=96 ms, policy=0.0%
3. **llama3.1:8b** - combined=0.540, quality_avg=34.5, avg_latency=276 ms, avg_ttft=88 ms, policy=0.0%
4. **mistral-small:latest** - combined=0.538, quality_avg=34.5, avg_latency=426 ms, avg_ttft=97 ms, policy=0.0%
5. **gpt-oss:20b** - combined=0.523, quality_avg=34.5, avg_latency=1086 ms, avg_ttft=962 ms, policy=0.0%
6. **gpt-oss:120b** - combined=0.518, quality_avg=34.5, avg_latency=1310 ms, avg_ttft=1123 ms, policy=0.0%
7. **qwen3.6:35b** - combined=0.442, quality_avg=34.5, avg_latency=4959 ms, avg_ttft=4756 ms, policy=0.0%
8. **gemma4:31b** - combined=0.422, quality_avg=34.5, avg_latency=5982 ms, avg_ttft=5584 ms, policy=0.0%
9. **mistral-medium-3.5:latest** - combined=0.241, quality_avg=34.5, avg_latency=14794 ms, avg_ttft=13685 ms, policy=0.0%

## Dettaglio per modello

### gemma4:31b
- Quality avg: **34.5/100**
- Quality p50: **34.5**
- Quality p95: **34.5**
- Policy compliance: **0.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **5982 ms**
- P50 latency: **5935 ms**
- P95 latency: **7486 ms**
- Avg TTFT streaming: **5584 ms**
- P95 TTFT streaming: **7105 ms**
- Realtime profile: **not_realtime** (compatibile=False)
- Realtime note: **Tempo prima risposta o latenza totale troppo alti per realtime naturale.**
- Avg tokens/s: **59.9**

#### Esempi affrontati
- Q `q01_checkin_time`: A che ora posso fare il check-in oggi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q02_checkout_time`: Qual e l orario massimo per il check-out?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q26_connecting_rooms`: Ci sono camere comunicanti per famiglia?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.

### gpt-oss:20b
- Quality avg: **34.5/100**
- Quality p50: **34.5**
- Quality p95: **34.5**
- Policy compliance: **0.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **1086 ms**
- P50 latency: **1020 ms**
- P95 latency: **1707 ms**
- Avg TTFT streaming: **962 ms**
- P95 TTFT streaming: **1526 ms**
- Realtime profile: **not_realtime** (compatibile=False)
- Realtime note: **Tempo prima risposta o latenza totale troppo alti per realtime naturale.**
- Avg tokens/s: **237.6**

#### Esempi affrontati
- Q `q01_checkin_time`: A che ora posso fare il check-in oggi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q02_checkout_time`: Qual e l orario massimo per il check-out?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q26_connecting_rooms`: Ci sono camere comunicanti per famiglia?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.

### qwen2.5:7b
- Quality avg: **34.5/100**
- Quality p50: **34.5**
- Quality p95: **34.5**
- Policy compliance: **0.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **181 ms**
- P50 latency: **175 ms**
- P95 latency: **253 ms**
- Avg TTFT streaming: **75 ms**
- P95 TTFT streaming: **86 ms**
- Realtime profile: **realtime_ready** (compatibile=True)
- Realtime note: **Tempo prima risposta e latenza totale entro target realtime.**
- Avg tokens/s: **257.5**

#### Esempi affrontati
- Q `q01_checkin_time`: A che ora posso fare il check-in oggi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q02_checkout_time`: Qual e l orario massimo per il check-out?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q26_connecting_rooms`: Ci sono camere comunicanti per famiglia?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.

### llama3.1:8b
- Quality avg: **34.5/100**
- Quality p50: **34.5**
- Quality p95: **34.5**
- Policy compliance: **0.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **276 ms**
- P50 latency: **254 ms**
- P95 latency: **414 ms**
- Avg TTFT streaming: **88 ms**
- P95 TTFT streaming: **95 ms**
- Realtime profile: **realtime_ready** (compatibile=True)
- Realtime note: **Tempo prima risposta e latenza totale entro target realtime.**
- Avg tokens/s: **236.0**

#### Esempi affrontati
- Q `q01_checkin_time`: A che ora posso fare il check-in oggi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q02_checkout_time`: Qual e l orario massimo per il check-out?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q26_connecting_rooms`: Ci sono camere comunicanti per famiglia?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.

### gpt-oss:120b
- Quality avg: **34.5/100**
- Quality p50: **34.5**
- Quality p95: **34.5**
- Policy compliance: **0.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **1310 ms**
- P50 latency: **1289 ms**
- P95 latency: **2353 ms**
- Avg TTFT streaming: **1123 ms**
- P95 TTFT streaming: **2113 ms**
- Realtime profile: **not_realtime** (compatibile=False)
- Realtime note: **Tempo prima risposta o latenza totale troppo alti per realtime naturale.**
- Avg tokens/s: **171.6**

#### Esempi affrontati
- Q `q01_checkin_time`: A che ora posso fare il check-in oggi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q02_checkout_time`: Qual e l orario massimo per il check-out?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q26_connecting_rooms`: Ci sono camere comunicanti per famiglia?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.

### mistral-small:latest
- Quality avg: **34.5/100**
- Quality p50: **34.5**
- Quality p95: **34.5**
- Policy compliance: **0.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **426 ms**
- P50 latency: **418 ms**
- P95 latency: **608 ms**
- Avg TTFT streaming: **97 ms**
- P95 TTFT streaming: **102 ms**
- Realtime profile: **realtime_ready** (compatibile=True)
- Realtime note: **Tempo prima risposta e latenza totale entro target realtime.**
- Avg tokens/s: **95.4**

#### Esempi affrontati
- Q `q01_checkin_time`: A che ora posso fare il check-in oggi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q02_checkout_time`: Qual e l orario massimo per il check-out?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q26_connecting_rooms`: Ci sono camere comunicanti per famiglia?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.

### qwen3.6:35b
- Quality avg: **34.5/100**
- Quality p50: **34.5**
- Quality p95: **34.5**
- Policy compliance: **0.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **4959 ms**
- P50 latency: **3679 ms**
- P95 latency: **9880 ms**
- Avg TTFT streaming: **4756 ms**
- P95 TTFT streaming: **9667 ms**
- Realtime profile: **not_realtime** (compatibile=False)
- Realtime note: **Tempo prima risposta o latenza totale troppo alti per realtime naturale.**
- Avg tokens/s: **136.8**

#### Esempi affrontati
- Q `q01_checkin_time`: A che ora posso fare il check-in oggi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q02_checkout_time`: Qual e l orario massimo per il check-out?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q26_connecting_rooms`: Ci sono camere comunicanti per famiglia?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.

### mistral-medium-3.5:latest
- Quality avg: **34.5/100**
- Quality p50: **34.5**
- Quality p95: **34.5**
- Policy compliance: **0.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **14794 ms**
- P50 latency: **14711 ms**
- P95 latency: **19855 ms**
- Avg TTFT streaming: **13685 ms**
- P95 TTFT streaming: **18359 ms**
- Realtime profile: **not_realtime** (compatibile=False)
- Realtime note: **Tempo prima risposta o latenza totale troppo alti per realtime naturale.**
- Avg tokens/s: **18.9**

#### Esempi affrontati
- Q `q01_checkin_time`: A che ora posso fare il check-in oggi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q02_checkout_time`: Qual e l orario massimo per il check-out?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q26_connecting_rooms`: Ci sono camere comunicanti per famiglia?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.

### llama3.2:latest
- Quality avg: **34.5/100**
- Quality p50: **34.5**
- Quality p95: **34.5**
- Policy compliance: **0.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **207 ms**
- P50 latency: **207 ms**
- P95 latency: **292 ms**
- Avg TTFT streaming: **96 ms**
- P95 TTFT streaming: **104 ms**
- Realtime profile: **realtime_ready** (compatibile=True)
- Realtime note: **Tempo prima risposta e latenza totale entro target realtime.**
- Avg tokens/s: **414.8**

#### Esempi affrontati
- Q `q01_checkin_time`: A che ora posso fare il check-in oggi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Nessuna nota
- Q `q02_checkout_time`: Qual e l orario massimo per il check-out?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta breve e precisa, corretta e sicura.
- Q `q26_connecting_rooms`: Ci sono camere comunicanti per famiglia?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: 
  Score finale=34.5, rule=10.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.

