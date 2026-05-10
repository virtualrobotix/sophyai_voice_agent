# Summary Benchmark Qualita con Ranking ed Esempi

- Data generazione: `2026-05-10T09:10:25.208247+00:00`
- Endpoint: `http://127.0.0.1:11434`
- Judge model: `llama3.2:latest`
- Domande: `50`

## Ranking qualita

1. **qwen3.6:35b** - quality_avg=65.2, policy=88.0%, hallucination_risk=0.0%, avg_latency=4924 ms, avg_ttft=4722 ms
2. **llama3.1:8b** - quality_avg=64.7, policy=82.0%, hallucination_risk=0.0%, avg_latency=273 ms, avg_ttft=86 ms
3. **gpt-oss:120b** - quality_avg=63.2, policy=72.0%, hallucination_risk=0.0%, avg_latency=1340 ms, avg_ttft=1149 ms
4. **mistral-small:latest** - quality_avg=61.8, policy=70.0%, hallucination_risk=0.0%, avg_latency=425 ms, avg_ttft=96 ms
5. **llama3.2:latest** - quality_avg=61.4, policy=82.0%, hallucination_risk=0.0%, avg_latency=201 ms, avg_ttft=91 ms
6. **gpt-oss:20b** - quality_avg=60.8, policy=60.0%, hallucination_risk=0.0%, avg_latency=1216 ms, avg_ttft=1088 ms
7. **qwen2.5:7b** - quality_avg=57.9, policy=64.0%, hallucination_risk=0.0%, avg_latency=179 ms, avg_ttft=73 ms
8. **mistral-medium-3.5:latest** - quality_avg=57.8, policy=44.0%, hallucination_risk=0.0%, avg_latency=14416 ms, avg_ttft=13313 ms
9. **gemma4:31b** - quality_avg=57.0, policy=84.0%, hallucination_risk=0.0%, avg_latency=5976 ms, avg_ttft=5576 ms

## Ranking combinato qualita e latenza

1. **llama3.1:8b** - combined=0.751, quality_avg=64.7, avg_latency=273 ms, avg_ttft=86 ms, policy=82.0%
2. **llama3.2:latest** - combined=0.729, quality_avg=61.4, avg_latency=201 ms, avg_ttft=91 ms, policy=82.0%
3. **mistral-small:latest** - combined=0.729, quality_avg=61.8, avg_latency=425 ms, avg_ttft=96 ms, policy=70.0%
4. **gpt-oss:120b** - combined=0.718, quality_avg=63.2, avg_latency=1340 ms, avg_ttft=1149 ms, policy=72.0%
5. **qwen2.5:7b** - combined=0.705, quality_avg=57.9, avg_latency=179 ms, avg_ttft=73 ms, policy=64.0%
6. **gpt-oss:20b** - combined=0.703, quality_avg=60.8, avg_latency=1216 ms, avg_ttft=1088 ms, policy=60.0%
7. **qwen3.6:35b** - combined=0.654, quality_avg=65.2, avg_latency=4924 ms, avg_ttft=4722 ms, policy=88.0%
8. **gemma4:31b** - combined=0.576, quality_avg=57.0, avg_latency=5976 ms, avg_ttft=5576 ms, policy=84.0%
9. **mistral-medium-3.5:latest** - combined=0.405, quality_avg=57.8, avg_latency=14416 ms, avg_ttft=13313 ms, policy=44.0%

## Esempi completi per modello (ordinati per ranking combinato)

### 1. llama3.1:8b
- Combined score: **0.751**
- Quality avg: **64.7/100**
- Avg latency: **273 ms**
- Avg TTFT: **86 ms**

#### Esempi
- Q `q06_parking_available`: Avete parcheggio privato?
  R: Sì, abbiamo un parcheggio privato all'interno del complesso. È gratuito e disponibile per gli ospiti. Per accedervi, segui le indicazioni di parcheggio che troverai vicino all'ingresso principale.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Nessuna nota
- Q `q12_extra_bed`: E possibile aggiungere un letto extra?
  R: Sì, è possibile aggiungere un letto extra. Ci sono disponibili letti supplementari di diverse dimensioni. Qual è il tipo di letto che desidera aggiungere?
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere meglio dire semplicemente: 'Sì, ci sono letti supplementari disponibili'
- Q `q14_accessible_room`: Ci sono camere accessibili per sedia a rotelle?
  R: Sì, abbiamo camere con accessibilità per persone con disabilità motorie. Posso fornirti informazioni dettagliate e prenotare una stanza adatta alle tue esigenze. Vuoi che ti dia maggiori informazioni o procediamo con la prenotazione?
  Score finale=62.7, rule=53.3, judge=80.0. Nota: Nessuna nota
- Q `q47_smoking_area`: Dove posso fumare senza problemi?
  R: Sono spiacente, ma non è possibile fumare all'interno dell'edificio. Ci sono aree esterne designate per fumatori vicino alla porta principale. Posso indicarti la direzione?
  Score finale=47.5, rule=30.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: Prenotazione dal sito è più veloce e sicura, ma se hai bisogno di aiuto puoi chiamarci.
  Score finale=34.0, rule=20.0, judge=60.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata

### 2. llama3.2:latest
- Combined score: **0.729**
- Quality avg: **61.4/100**
- Avg latency: **201 ms**
- Avg TTFT: **91 ms**

#### Esempi
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

### 3. mistral-small:latest
- Combined score: **0.729**
- Quality avg: **61.8/100**
- Avg latency: **425 ms**
- Avg TTFT: **96 ms**

#### Esempi
- Q `q13_baby_cot`: Avete culla per neonato?
  R: Sì, abbiamo una culla disponibile per i neonati. Vuoi prenotarla o hai bisogno di ulteriori informazioni?
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Nessuna nota
- Q `q19_city_tax`: La tassa di soggiorno e inclusa?
  R: No, la tassa di soggiorno non è inclusa nel prezzo della camera. Sarà addebitata separatamente al check-out.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Risposta breve e precisa, ma potrebbe essere più utile specificare se la tassa di soggiorno è variabile o se ci sono sconti per pagamenti anticipati.
- Q `q33_cancel_free_window`: Entro quando posso cancellare senza penali?
  R: Puoi cancellare senza penali entro 24 ore dalla prenotazione. Controlla i dettagli nella tua conferma di prenotazione per ulteriori informazioni.
  Score finale=62.7, rule=53.3, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata più concisa
- Q `q17_early_checkin`: Posso fare check-in alle dieci del mattino?
  R: Il check-in inizia alle 14:00. Può lasciare i bagagli alla reception se arriva prima.
  Score finale=34.0, rule=20.0, judge=60.0. Nota: Nessuna nota
- Q `q23_quiet_room_request`: Potrei avere una camera silenziosa?
  R: Verifico disponibilità camere silenziose. Attenda un momento.
  Score finale=34.0, rule=20.0, judge=60.0. Nota: Risposta troppo lunga, potrebbe essere stata meglio condensata

### 4. gpt-oss:120b
- Combined score: **0.718**
- Quality avg: **63.2/100**
- Avg latency: **1340 ms**
- Avg TTFT: **1149 ms**

#### Esempi
- Q `q05_breakfast_included`: La colazione e inclusa nella tariffa?
  R: Sì, la colazione è inclusa nella tariffa. Per conferma, verifica la tua prenotazione o contatta la reception.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere meglio dire semplicemente: La colazione è inclusa nella tariffa.
- Q `q16_airport_transfer`: Organizzate transfer da e per aeroporto?
  R: Sì, organizziamo transfer da e per l’aeroporto. Indichi data, orario e numero passeggeri per confermare la prenotazione.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Nessuna nota
- Q `q23_quiet_room_request`: Potrei avere una camera silenziosa?
  R: La camera silenziosa è disponibile per le date richieste; confermi il periodo di soggiorno per procedere con la prenotazione?
  Score finale=62.7, rule=53.3, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere meglio dire semplicemente: La camera silenziosa è disponibile per le date richieste.
- Q `q44_restaurant_recommendation`: Mi consigliate un ristorante vicino?
  R: A pochi passi dal centro c’è “Trattoria Da Marco”, cucina tradizionale e ottime recensioni. Vuoi indicazioni o prenotare?
  Score finale=41.0, rule=20.0, judge=80.0. Nota: Nessuna nota
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: Prenotare sul sito è più veloce e garantisce conferma immediata; se preferisce, può farlo anche qui al telefono.
  Score finale=41.0, rule=20.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata

### 5. qwen2.5:7b
- Combined score: **0.705**
- Quality avg: **57.9/100**
- Avg latency: **179 ms**
- Avg TTFT: **73 ms**

#### Esempi
- Q `q19_city_tax`: La tassa di soggiorno e inclusa?
  R: Sì, la tassa di soggiorno è inclusa.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Nessuna nota
- Q `q05_breakfast_included`: La colazione e inclusa nella tariffa?
  R: Sì, la colazione è inclusa nella tariffa.
  Score finale=86.5, rule=90.0, judge=80.0. Nota: Risposta troppo generica, potrebbe essere più utile specificare i dettagli della colazione inclusa.
- Q `q44_restaurant_recommendation`: Mi consigliate un ristorante vicino?
  R: Prova il Ristorante Trattoria Roma, a due blocchi di strada. Buona cucina italiana.
  Score finale=56.2, rule=43.3, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: Prenota dal sito per maggior velocità e facilità.
  Score finale=41.0, rule=20.0, judge=80.0. Nota: Nessuna nota
- Q `q04_double_room_price`: Quanto costa una matrimoniale per una notte?
  R: Varia a seconda del hotel. Controlla il sito web o chiamate la reception per informazioni precise.
  Score finale=40.5, rule=30.0, judge=60.0. Nota: Nessuna nota

### 6. gpt-oss:20b
- Combined score: **0.703**
- Quality avg: **60.8/100**
- Avg latency: **1216 ms**
- Avg TTFT: **1088 ms**

#### Esempi
- Q `q06_parking_available`: Avete parcheggio privato?
  R: Sì, offriamo un parcheggio privato con 20 posti. È disponibile su prenotazione.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Risposta troppo dettagliata, potrebbe essere meglio dire semplicemente che il parcheggio è disponibile
- Q `q05_breakfast_included`: La colazione e inclusa nella tariffa?
  R: Sì, la colazione è inclusa nella tariffa. Se hai bisogno di ulteriori informazioni, chiedi pure.
  Score finale=86.5, rule=90.0, judge=80.0. Nota: Nessuna nota
- Q `q23_quiet_room_request`: Potrei avere una camera silenziosa?
  R: Sì, abbiamo una camera silenziosa disponibile. Vuole prenotarla ora?
  Score finale=62.7, rule=53.3, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere meglio dire semplicemente: Sì, abbiamo una camera silenziosa disponibile.
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: Entrambi i metodi sono validi; prenotando sul sito potete vedere disponibilità in tempo reale, mentre con noi potete chiedere assistenza personalizzata.
  Score finale=41.0, rule=20.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata più concisa
- Q `q34_no_show_policy`: Cosa succede se non mi presento?
  R: Se non ti presenti, la prenotazione può essere annullata, potresti perdere l'accesso e il personale dovrà gestire la situazione.
  Score finale=40.5, rule=30.0, judge=60.0. Nota: Nessuna nota

### 7. qwen3.6:35b
- Combined score: **0.654**
- Quality avg: **65.2/100**
- Avg latency: **4924 ms**
- Avg TTFT: **4722 ms**

#### Esempi
- Q `q05_breakfast_included`: La colazione e inclusa nella tariffa?
  R: La colazione non è inclusa nella tariffa base. Può aggiungerla al check-in per quindici euro a persona.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata abbreviata
- Q `q15_wifi_info`: Il wifi e gratuito?
  R: Il wifi è gratuito per tutti gli ospiti. Il codice di accesso è indicato alla reception o sul tuo biglietto.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Risposta breve e precisa, ma manca di specificità sul codice di accesso.
- Q `q16_airport_transfer`: Organizzate transfer da e per aeroporto?
  R: Sì, organizziamo transfer aeroportuali su richiesta. Indicami data, orario e numero di passeggeri per confermare.
  Score finale=62.7, rule=53.3, judge=80.0. Nota: Nessuna nota
- Q `q41_lost_item`: Ho dimenticato un caricatore in camera, come posso recuperarlo?
  R: Contatti subito la reception per segnalare l'oggetto. Un addetto verificherà la camera e le organizzerà il ritiro o la spedizione.
  Score finale=47.5, rule=30.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere meglio dire che il cliente contatta la reception per segnalare l'oggetto e poi chiedere al addetto di organizzare il ritiro o la spedizione
- Q `q49_booking_channel`: Conviene prenotare dal sito o direttamente con voi?
  R: Prenotare dal sito è più veloce e ti dà conferma immediata. Se preferisci assistenza personalizzata, chiamaci direttamente e ti guidiamo passo passo.
  Score finale=41.0, rule=20.0, judge=80.0. Nota: Nessuna nota

### 8. gemma4:31b
- Combined score: **0.576**
- Quality avg: **57.0/100**
- Avg latency: **5976 ms**
- Avg TTFT: **5576 ms**

#### Esempi
- Q `q15_wifi_info`: Il wifi e gratuito?
  R: Sì, il wifi è gratuito. Può trovare le credenziali in camera o richiederle alla reception.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere meglio dire che le credenziali sono disponibili in camera
- Q `q08_cancel_reservation`: Come posso cancellare la mia prenotazione?
  R: Può cancellare la prenotazione tramite l'area personale sul nostro sito o chiamando l'ufficio. Se preferisce, posso trasferirla all'operatore.
  Score finale=77.8, rule=76.7, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere meglio dire che la cancellazione è possibile tramite l'area personale sul sito o chiamando l'ufficio
- Q `q50_human_operator`: Vorrei parlare con una persona, mi puoi passare un operatore?
  R: La trasferisco immediatamente a un operatore. Resti in linea.
  Score finale=62.7, rule=53.3, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata migliorata con una sintesi più veloce.
- Q `q33_cancel_free_window`: Entro quando posso cancellare senza penali?
  R: Non dispongo dei dettagli della sua prenotazione. Fornisca il numero di pratica o controlli l'email di conferma.
  Score finale=40.5, rule=30.0, judge=60.0. Nota: Nessuna nota
- Q `q10_invoice_request`: Mi potete fare fattura intestata alla mia azienda?
  R: Non posso emettere fatture. Invii i dati aziendali all'ufficio amministrativo per procedere.
  Score finale=34.0, rule=20.0, judge=60.0. Nota: Risposta troppo lunga, potrebbe essere meglio dire che non è possibile emettere fatture e proporre il prossimo passo.

### 9. mistral-medium-3.5:latest
- Combined score: **0.405**
- Quality avg: **57.8/100**
- Avg latency: **14416 ms**
- Avg TTFT: **13313 ms**

#### Esempi
- Q `q15_wifi_info`: Il wifi e gratuito?
  R: Sì, il wifi è gratuito. Chieda la password alla reception.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere meglio dire semplicemente: Sì, il wifi è gratuito.
- Q `q19_city_tax`: La tassa di soggiorno e inclusa?
  R: No, la tassa di soggiorno non è inclusa. Verrà addebitata separatamente al check-out.
  Score finale=93.0, rule=100.0, judge=80.0. Nota: Nessuna nota
- Q `q29_business_trip_receipt`: Sono in trasferta lavoro, come ricevo ricevuta e fattura?
  R: Chiedi alla segreteria aziendale o all'ufficio amministrativo. Invia loro i dati della spesa.
  Score finale=56.2, rule=43.3, judge=80.0. Nota: Risposta troppo lunga, potrebbe essere stata meglio una risposta più breve e direttamente inviando i dati alla segreteria aziendale.
- Q `q41_lost_item`: Ho dimenticato un caricatore in camera, come posso recuperarlo?
  R: Contatti la reception al numero interno 0. Se non risponde, passa in portineria.
  Score finale=41.0, rule=20.0, judge=80.0. Nota: Nessuna nota
- Q `q34_no_show_policy`: Cosa succede se non mi presento?
  R: Verrai segnalato come assente. Controlla il regolamento per le conseguenze.
  Score finale=40.5, rule=30.0, judge=60.0. Nota: Risposta troppo lunga, potrebbe essere meglio dire 'Verrai segnalato' e fare riferimento al regolamento

