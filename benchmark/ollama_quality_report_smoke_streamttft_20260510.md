# Benchmark Ollama: Qualita Risposte Receptionist

- Data UTC: `2026-05-10T07:33:35.952404+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Domande standard: `50`
- Repeats per domanda: `1`
- Judge model: `llama3.2:latest`

## Ranking qualita

1. **llama3.2:latest** - quality_avg=34.5, policy=0.0%, hallucination_risk=0.0%, avg_latency=536 ms, avg_ttft=416 ms

## Compatibilita realtime (latenza)

- **llama3.2:latest**: avg=536 ms, p95=375 ms, avg_ttft=416 ms, p95_ttft=209 ms, profilo=borderline, compatibile=False. Motivo: Usabile ma con ritardo percepibile su avvio risposta o completamento.

## Ranking combinato qualita e latenza

1. **llama3.2:latest** - combined=0.541, quality_avg=34.5, avg_latency=536 ms, avg_ttft=416 ms, policy=0.0%

## Dettaglio per modello

### llama3.2:latest
- Quality avg: **34.5/100**
- Quality p50: **34.5**
- Quality p95: **34.5**
- Policy compliance: **0.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **536 ms**
- P50 latency: **220 ms**
- P95 latency: **375 ms**
- Avg TTFT streaming: **416 ms**
- P95 TTFT streaming: **209 ms**
- Realtime profile: **borderline** (compatibile=False)
- Realtime note: **Usabile ma con ritardo percepibile su avvio risposta o completamento.**
- Avg tokens/s: **408.5**

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

