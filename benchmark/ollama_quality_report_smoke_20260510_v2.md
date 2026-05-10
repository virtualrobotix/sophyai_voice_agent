# Benchmark Ollama: Qualita Risposte Receptionist

- Data UTC: `2026-05-10T06:55:11.131779+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Domande standard: `50`
- Repeats per domanda: `1`
- Judge model: `llama3.2:latest`

## Ranking qualita

1. **llama3.2:latest** - quality_avg=59.4, policy=74.0%, hallucination_risk=0.0%, avg_latency=713 ms

## Compatibilita realtime (latenza)

- **llama3.2:latest**: avg=713 ms, p95=1672 ms, profilo=borderline, compatibile=False. Motivo: Usabile ma con rischio ritardi percepibili in conversazione.

## Ranking combinato qualita e latenza

1. **llama3.2:latest** - combined=0.696, quality_avg=59.4, avg_latency=713 ms, policy=74.0%

## Dettaglio per modello

### llama3.2:latest
- Quality avg: **59.4/100**
- Quality p50: **59.7**
- Quality p95: **93.0**
- Policy compliance: **74.0%**
- Hallucination risk rate: **0.0%**
- Avg latency: **713 ms**
- P50 latency: **629 ms**
- P95 latency: **1672 ms**
- Realtime profile: **borderline** (compatibile=False)
- Realtime note: **Usabile ma con rischio ritardi percepibili in conversazione.**
- Avg tokens/s: **86.3**

#### Esempi affrontati
- Q `q06_parking_available`: Avete parcheggio privato?
  R: Sì, abbiamo un parcheggio privato all'interno del complesso. È disponibile per i clienti che lo richiedono.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q13_baby_cot`: Avete culla per neonato?
  R: Sì, abbiamo una culla per neonati disponibile. Posso portarla nella stanza del bambino quando lo desideri.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Nessuna nota
- Q `q25_non_smoking_room`: Avete camere non fumatori?
  R: Sì, abbiamo camere non fumatori disponibili. Vuoi sapere di più sulle nostre opzioni e prezzo?
  Score finale=56.8, rule=55.0, judge=60.0. Nota: Risposta troppo lunga, potrebbe essere stata più concisa
- Q `q47_smoking_area`: Dove posso fumare senza problemi?
  R: Puoi fumare all'aperto in aree designate come parchi, piazze o lungofiumi, ma non è consentito in locali aperti al pubblico come bar, ristoranti o negozi.
  Score finale=41.0, rule=20.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata più concisa
- Q `q04_double_room_price`: Quanto costa una matrimoniale per una notte?
  R: Il prezzo di una matrimonale per una notte varia a seconda del luogo e della durata dell'esperienza, ma generalmente può variare da 500 a 5.000 euro.
  Score finale=40.5, rule=30.0, judge=60.0. Nota: Nessuna nota

