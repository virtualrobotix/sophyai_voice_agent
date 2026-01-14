# Guida Generazione Grafici per Presentazione

## Panoramica

Sono disponibili **due metodi** per generare i grafici della presentazione:

1. **Python + Matplotlib** (grafici PNG ad alta risoluzione)
2. **Mermaid** (diagrammi vettoriali, convertibili in immagini)

---

## Metodo 1: Python + Matplotlib (Consigliato)

### Installazione

```bash
# Opzione A: Script automatico
./installa_dipendenze_grafici.sh

# Opzione B: Manuale
pip3 install matplotlib networkx pillow
```

### Generazione

```bash
python3 genera_grafici.py
```

### Output

I grafici verranno salvati in `grafici_presentazione/`:
- `01_architettura.png` - Diagramma architettura sistema
- `02_flusso_e2e.png` - Flusso end-to-end
- `03_stack_tecnologico.png` - Stack tecnologico
- `04_deployment_docker.png` - Deployment Docker
- `05_confronto_tts.png` - Confronto TTS engines
- `06_metriche_performance.png` - Metriche performance
- `07_database_schema.png` - Schema database
- `08_sequenza_multiuser.png` - Sequenza multi-user

**Risoluzione**: 300 DPI (adatta per presentazioni)

### Inserimento in PowerPoint

1. Apri PowerPoint
2. Per ogni slide che richiede un grafico:
   - Inserisci → Immagine → Da file
   - Seleziona il file PNG corrispondente
   - Ridimensiona e posiziona

---

## Metodo 2: Mermaid (Alternativa)

### Visualizzazione

I diagrammi Mermaid sono in `grafici_mermaid.md` e possono essere:
- Visualizzati direttamente in GitHub, VS Code, o qualsiasi viewer Markdown
- Convertiti in immagini usando tool online o CLI

### Conversione in Immagini

#### Opzione A: Mermaid Live Editor (Online)

1. Vai su https://mermaid.live
2. Copia il codice Mermaid da `grafici_mermaid.md`
3. Clicca "Actions" → "Download PNG" o "Download SVG"

#### Opzione B: Mermaid CLI

```bash
# Installa Node.js e npm
# Poi installa Mermaid CLI
npm install -g @mermaid-js/mermaid-cli

# Converti tutti i diagrammi
mmdc -i grafici_mermaid.md -o grafici_presentazione/ -b transparent
```

#### Opzione C: VS Code Extension

1. Installa extension "Markdown Preview Mermaid Support"
2. Apri `grafici_mermaid.md` in VS Code
3. Usa "Export Diagram" dal menu contestuale del diagramma

---

## Mappatura Slide → Grafici

| Slide | Grafico | File |
|-------|---------|------|
| **Slide 4**: Architettura | Diagramma architettura | `01_architettura.png` |
| **Slide 10**: Flusso E2E | Flusso end-to-end | `02_flusso_e2e.png` |
| **Slide 6**: Stack Tecnologico | Stack tecnologico | `03_stack_tecnologico.png` |
| **Slide 14**: Deployment | Deployment Docker | `04_deployment_docker.png` |
| **Slide 9**: TTS Engines | Confronto TTS | `05_confronto_tts.png` |
| **Slide 24**: Performance | Metriche performance | `06_metriche_performance.png` |
| **Slide 13**: Database | Schema database | `07_database_schema.png` |
| **Slide 11**: Multi-User | Sequenza multi-user | `08_sequenza_multiuser.png` |

---

## Personalizzazione

### Modificare Colori (Python)

Modifica `genera_grafici.py` e cambia i colori nelle variabili:
```python
color='#059669'  # Verde
color='#dc2626'  # Rosso
color='#2563eb'  # Blu
# ecc.
```

### Modificare Dimensioni

```python
fig, ax = plt.subplots(1, 1, figsize=(14, 10))  # Larghezza, Altezza
```

### Modificare DPI

```python
plt.savefig('file.png', dpi=300)  # 300 DPI per presentazioni
```

---

## Troubleshooting

### Errore: "matplotlib non installato"

```bash
pip3 install matplotlib
# oppure
python3 -m pip install matplotlib
```

### Errore: "Permission denied" su script

```bash
chmod +x installa_dipendenze_grafici.sh
```

### Grafici non si vedono bene in PowerPoint

- Assicurati di usare DPI 300 (già impostato)
- In PowerPoint, usa "Inserisci" → "Immagine" invece di copia/incolla
- Se necessario, aumenta dimensioni canvas in `genera_grafici.py`

### Mermaid non si converte

- Verifica sintassi Mermaid su https://mermaid.live
- Usa Mermaid CLI per conversione batch
- Alternativa: usa tool online per ogni diagramma

---

## Suggerimenti

1. **Genera prima i grafici** prima di creare la presentazione PowerPoint
2. **Rivedi i grafici** e personalizza colori/stili se necessario
3. **Aggiungi screenshot** del sistema funzionante
4. **Crea diagrammi aggiuntivi** se necessario (es. flusso API, architettura dettagliata)

---

## File Creati

```
sophyai-live-server/
├── genera_grafici.py              # Script generazione PNG
├── grafici_mermaid.md             # Diagrammi Mermaid
├── installa_dipendenze_grafici.sh  # Script installazione
├── README_GRAFICI.md              # Questo file
└── grafici_presentazione/         # Directory output (creata automaticamente)
    ├── 01_architettura.png
    ├── 02_flusso_e2e.png
    ├── 03_stack_tecnologico.png
    ├── 04_deployment_docker.png
    ├── 05_confronto_tts.png
    ├── 06_metriche_performance.png
    ├── 07_database_schema.png
    └── 08_sequenza_multiuser.png
```

---

## Prossimi Passi

1. ✅ Installa dipendenze: `./installa_dipendenze_grafici.sh`
2. ✅ Genera grafici: `python3 genera_grafici.py`
3. ✅ Apri PowerPoint
4. ✅ Inserisci grafici nelle slide corrispondenti
5. ✅ Personalizza colori/stili se necessario
