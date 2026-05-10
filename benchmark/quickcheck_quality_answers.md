# Benchmark Ollama: Qualita Risposte Receptionist

- Data UTC: `2026-05-10T08:31:57.733786+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Domande standard: `50`
- Repeats per domanda: `1`
- Judge model: `llama3.2:latest`

## Ranking qualita

1. **llama3.2:latest** - quality_avg=61.4, policy=82.0%, hallucination_risk=0.0%, avg_latency=269 ms, avg_ttft=95 ms

## Compatibilita realtime (latenza)

- **llama3.2:latest**: avg=269 ms, p95=400 ms, avg_ttft=95 ms, p95_ttft=102 ms, profilo=realtime_ready, compatibile=True. Motivo: Tempo prima risposta e latenza totale entro target realtime.

## Ranking combinato qualita e latenza

1. **llama3.2:latest** - combined=0.729, quality_avg=61.4, avg_latency=269 ms, avg_ttft=95 ms, policy=82.0%

## Dettaglio per modello

### llama3.2:latest
- Quality avg: **61.4/100**
- Quality p50: **62.7**
- Quality p95: **93.0**
- Policy compliance: **82.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **269 ms**
- P50 latency: **266 ms**
- P95 latency: **400 ms**
- Avg TTFT streaming: **95 ms**
- P95 TTFT streaming: **102 ms**
- Realtime profile: **realtime_ready** (compatibile=True)
- Realtime note: **Tempo prima risposta e latenza totale entro target realtime.**
- Avg tokens/s: **243.0**

#### Esempi affrontati
- Q `q06_parking_available`: Avete parcheggio privato?
  R: Sì, abbiamo un parcheggio privato all'interno del complesso. È disponibile per i clienti che lo richiedono.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q13_baby_cot`: Avete culla per neonato?
  R: Sì, abbiamo una culla per neonati disponibile. Posso portarla nella stanza del bambino quando lo desideri.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Nessuna nota
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: Posso collegarti direttamente con l'operatore disponibile. Per favore attendi un momento.
  Score finale=62.7, rule=53.3, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata meglio con una risposta più breve come "Posso collegarti direttamente con l'operatore disponibile"
- Q `q47_smoking_area`: Dove posso fumare senza problemi?
  R: Puoi fumare all'aperto in aree designate come parchi, piazze o lungofiumi, ma non è consentito in locali aperti al pubblico come bar, ristoranti o cinema.
  Score finale=41.0, rule=20.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata più concisa
- Q `q04_double_room_price`: Quanto costa una matrimoniale per una notte?
  R: Il prezzo di una matrimonale per una notte varia a seconda del luogo e della durata dell'esperienza, ma generalmente può variare da 500 a 5.000 euro.
  Score finale=40.5, rule=30.0, judge=60.0. Nota: Nessuna nota

