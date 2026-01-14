# Istruzioni per Creare Presentazione PowerPoint

## Opzione 1: Script Python (Automatico)

### Prerequisiti
```bash
pip install python-pptx
```

### Esecuzione
```bash
python convert_to_ppt.py
```

Questo creerà `PRESENTAZIONE_TECNICA.pptx` automaticamente.

### Personalizzazione
Dopo la creazione, apri il file in PowerPoint e:
- Personalizza colori e temi
- Aggiungi immagini/diagrammi
- Modifica layout se necessario

---

## Opzione 2: Pandoc (Alternativa)

### Prerequisiti
```bash
# macOS
brew install pandoc

# Linux
sudo apt-get install pandoc

# Windows
# Scarica da https://pandoc.org/installing.html
```

### Conversione
```bash
pandoc PRESENTAZIONE_TECNICA.md -o PRESENTAZIONE_TECNICA.pptx
```

**Nota**: Pandoc potrebbe non preservare perfettamente la formattazione. Lo script Python è più accurato.

---

## Opzione 3: Manuale (Più Controllo)

### Passi

1. **Apri PowerPoint** e crea nuova presentazione

2. **Per ogni slide in `PRESENTAZIONE_TECNICA.md`**:
   - Crea nuova slide
   - Copia titolo dalla sezione `## Slide X: Titolo`
   - Copia contenuto dalla slide
   - Formatta come necessario

3. **Aggiungi diagrammi**:
   - Slide 4: Architettura (usa diagramma ASCII o crea in PowerPoint)
   - Slide 10: Flusso end-to-end (crea diagramma di flusso)
   - Slide 14: Deployment (crea diagramma container)

4. **Personalizza**:
   - Aggiungi logo/immagini
   - Scegli tema colori
   - Aggiungi animazioni se necessario

---

## Opzione 4: Presentazione HTML (Reveal.js)

### Usa il file HTML direttamente

Il file `PRESENTAZIONE_TECNICA.html` può essere aperto direttamente nel browser.

**Vantaggi**:
- ✅ Nessuna conversione necessaria
- ✅ Funziona su qualsiasi dispositivo
- ✅ Navigazione con frecce tastiera
- ✅ Presentabile via browser

**Uso**:
```bash
# Apri nel browser
open PRESENTAZIONE_TECNICA.html

# Oppure
python -m http.server 8000
# Poi apri http://localhost:8000/PRESENTAZIONE_TECNICA.html
```

**Navigazione**:
- `→` o `Spazio`: Slide successiva
- `←`: Slide precedente
- `F`: Fullscreen
- `Esc`: Overview

---

## Suggerimenti per la Presentazione

### Timing Consigliato

- **Slide 1-5**: Introduzione (5 min)
- **Slide 6-12**: Architettura e Moduli (15 min)
- **Slide 13-18**: Deployment e API (10 min)
- **Slide 19-24**: Avanzato e Performance (10 min)
- **Slide 25-30**: Demo e Q&A (10 min)

**Totale**: ~50 minuti

### Slide da Evidenziare

1. **Slide 4**: Architettura - Mostra diagramma completo
2. **Slide 10**: Flusso end-to-end - Spiega il flusso passo-passo
3. **Slide 14**: Deployment - Mostra infrastruttura Docker
4. **Slide 26**: Demo - Mostra sistema funzionante

### Preparazione Demo

Prima della presentazione:
1. ✅ Sistema avviato e funzionante
2. ✅ Browser aperto su localhost:8443
3. ✅ Microfono testato
4. ✅ Esempi di conversazione pronti
5. ✅ API testate (curl o Postman)

### Elementi da Aggiungere

- **Screenshot**: Interfaccia web, dashboard, log
- **Diagrammi**: Architettura, flussi, deployment (crea in draw.io o PowerPoint)
- **Metriche**: Grafici performance, latenze
- **Code Snippets**: Esempi configurazione, API calls

---

## Struttura File

```
sophyai-live-server/
├── PRESENTAZIONE_TECNICA.md      # Markdown originale
├── PRESENTAZIONE_TECNICA.html    # Versione HTML (reveal.js)
├── PRESENTAZIONE_TECNICA.pptx    # PowerPoint (generato)
├── convert_to_ppt.py             # Script conversione
└── ISTRUZIONI_PRESENTAZIONE.md   # Questo file
```

---

## Troubleshooting

### Script Python non funziona

**Errore**: `ModuleNotFoundError: No module named 'pptx'`
```bash
pip install python-pptx
```

**Errore**: Formattazione non corretta
- Lo script è basico, potrebbe richiedere personalizzazione manuale
- Usa Opzione 3 (manuale) per più controllo

### Pandoc non preserva formattazione

- Pandoc è più adatto per documenti che presentazioni
- Usa lo script Python o crea manualmente

### HTML non si apre

- Assicurati che `reveal.js` sia nella stessa directory
- Oppure usa un server HTTP locale:
  ```bash
  python -m http.server 8000
  ```

---

## Supporto

Per problemi o domande:
- Consulta `DOCUMENTAZIONE.md` per dettagli tecnici
- Verifica che tutti i file siano nella stessa directory
- Controlla che le dipendenze siano installate
