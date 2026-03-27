# Manuale Utente - SophyAI Voice Agent

## Indice

1. [Primo Accesso](#primo-accesso)
2. [Pagina di Chat](#pagina-di-chat)
3. [Pannello Amministrazione](#pannello-amministrazione)
4. [Gestione Utenti](#gestione-utenti)
5. [Configurazione SMTP / Email](#configurazione-smtp--email)
6. [Configurazione LLM](#configurazione-llm)
7. [Configurazione TTS](#configurazione-tts)
8. [Configurazione Whisper STT](#configurazione-whisper-stt)
9. [Configurazione SIP / Telefonia](#configurazione-sip--telefonia)
10. [Calibrazione Audio](#calibrazione-audio)
11. [Log Chiamate](#log-chiamate)
12. [Ascolto Live e Debug](#ascolto-live-e-debug)

---

## Primo Accesso

### Credenziali di Default

Al primo avvio del sistema viene creato automaticamente un utente amministratore:

- **Username**: `admin`
- **Password**: `admin123`

### Login

1. Apri il browser e vai a `https://<ip-server>:8443`
2. Verrai reindirizzato alla pagina di login
3. Inserisci username e password
4. Al primo accesso ti verra' richiesto di **cambiare la password** (minimo 8 caratteri)
5. Dopo il cambio password verrai reindirizzato alla pagina appropriata:
   - **Admin** -> Pannello di amministrazione
   - **Utente** -> Pagina di chat

### Ruoli

| Ruolo | Accesso |
|-------|---------|
| **Admin** | Pannello admin completo + pagina chat |
| **Utente** | Solo pagina di chat |

### Password Dimenticata

Se e' configurato un server SMTP (vedi sezione dedicata):
1. Nella pagina di login, clicca "Password dimenticata?"
2. Inserisci l'email associata al tuo account
3. Riceverai un'email con un link per reimpostare la password
4. Il link e' valido per 1 ora

---

## Pagina di Chat

La pagina principale (`/`) consente di interagire con l'assistente vocale.

### Connessione alla Room

1. Inserisci il tuo nome nel campo "Il tuo nome"
2. Clicca "Connetti" per entrare nella room WebRTC
3. Il microfono verra' attivato automaticamente (richiede permesso del browser)

### Modalita' di Interazione

- **Voce**: Parla direttamente, l'assistente rispondera' a voce
- **Chat testuale**: Scrivi nel campo di testo e premi Invio
- **Modalita' Agent**: Usa il toggle per forzare l'interazione con l'assistente (utile in room con piu' partecipanti)

### Interruzione Vocale (Barge-in)

Mentre l'assistente sta parlando, puoi interromperlo semplicemente parlando.
Il sistema rileva la tua voce e interrompe la risposta in corso.

### Pannello Calibrazione Audio

Clicca il pulsante con l'icona dell'oscilloscopio per aprire il pannello di calibrazione:
- **VU Meter**: Visualizza il livello audio in tempo reale
- **Soglie regolabili**: VAD, speech, silenzio, cooldown TTS
- **Calibrazione automatica**: Clicca "Calibra" per tarare automaticamente le soglie in base al rumore ambientale
- Le impostazioni vengono salvate nel database e ricaricate automaticamente

---

## Pannello Amministrazione

Accessibile solo agli admin da `/admin` o cliccando "Admin" nell'interfaccia.

### Tab Disponibili

| Tab | Descrizione |
|-----|-------------|
| Statistiche | Metriche real-time: latenza STT/LLM/TTS, contatori |
| Log Chiamate | Registro completo delle chiamate SIP |
| Whisper | Configurazione motore STT |
| LLM | Selezione provider e modello LLM |
| TTS | Selezione motore e voce TTS |
| Voice | Parametri VAD e soglie vocali |
| SIP | Configurazione telefonia SIP |
| Ascolto Live | Monitoraggio conversazioni in corso |
| Email / SMTP | Configurazione server email |
| Utenti | Gestione utenti della piattaforma |

---

## Gestione Utenti

Dal tab "Utenti" nel pannello admin puoi:

### Creare un Nuovo Utente

1. Clicca "Nuovo Utente"
2. Compila: username (min 3 caratteri), email (opzionale), ruolo
3. La password iniziale e' `changeme1` (modificabile)
4. L'utente dovra' cambiarla al primo accesso

### Modificare un Utente

- Clicca "Modifica" per cambiare email o ruolo
- Non puoi rimuovere il ruolo admin dal tuo stesso account

### Reset Password

- **Manuale**: Clicca "Reset PW", imposta una nuova password. L'utente dovra' cambiarla al prossimo accesso.
- **Via Email**: Se SMTP e' configurato e l'utente ha un'email, clicca il pulsante email per inviare un link di reset.

### Disabilitare/Abilitare

- Clicca "Disabilita" per impedire l'accesso a un utente senza eliminarlo
- Non puoi disabilitare il tuo stesso account

---

## Configurazione SMTP / Email

Dal tab "Email / SMTP" puoi configurare l'invio email per il reset password.

### Provider Supportati

| Provider | Host | Porta | Note |
|----------|------|-------|------|
| **Gmail** | smtp.gmail.com | 587 (TLS) | Richiede "App Password" (non la password Gmail) |
| **Brevo** | smtp-relay.brevo.com | 587 (TLS) | Gratis fino a 300 email/giorno |
| **Resend** | smtp.resend.com | 465 (SSL) | Gratis fino a 100 email/giorno |
| **Mailgun** | smtp.mailgun.org | 587 (TLS) | Sandbox gratuito per test |
| **Personalizzato** | Configurabile | Configurabile | Qualsiasi server SMTP |

### Configurazione

1. Seleziona il provider dal menu a tendina
2. I campi Host e Porta vengono pre-compilati
3. Inserisci Username (email o API key) e Password (password o API secret)
4. Imposta l'email mittente (From)
5. Clicca "Salva Configurazione"
6. Clicca "Invia Email di Test" per verificare

### Gmail - App Password

1. Vai su [myaccount.google.com](https://myaccount.google.com)
2. Sicurezza -> Verifica in 2 passaggi (deve essere attiva)
3. Sicurezza -> Password per le app
4. Crea una nuova password per "Posta"
5. Usa la password generata (16 caratteri) come password SMTP

### Brevo (Consigliato - Gratuito)

1. Registrati su [brevo.com](https://brevo.com)
2. Vai su Impostazioni -> SMTP & API
3. Copia la SMTP Key
4. Username: la tua email Brevo
5. Password: la SMTP Key

---

## Configurazione LLM

Dal tab "LLM" puoi scegliere il provider e il modello per le risposte dell'assistente.

### Ollama (Locale)

- Seleziona "Ollama" come provider
- Scegli il modello dalla lista dei modelli installati
- I modelli girano interamente in locale (privacy totale)

### OpenRouter (Cloud)

- Seleziona "OpenRouter" come provider
- Inserisci la tua API Key (da [openrouter.ai](https://openrouter.ai))
- Scegli tra 100+ modelli (GPT-4, Claude, Gemini, Mistral, ecc.)

### Remote LLM Server

- Configura URL, token e collection per un server LLM esterno con RAG
- Utile per integrare knowledge base aziendali

### System Prompt

Configura il comportamento dell'assistente modificando il system prompt.
Il prompt definisce personalita', regole di risposta, lingua e formato.

### Context Injection

Campo per iniettare contesto aggiuntivo in ogni conversazione (es. informazioni sull'azienda, menu, orari).

---

## Configurazione TTS

Dal tab "TTS" puoi scegliere il motore di sintesi vocale.

| Motore | Self-Hosted | Qualita' | Latenza | Note |
|--------|-------------|----------|---------|------|
| Edge | No (API MS) | Ottima | Bassa | Consigliato, gratuito |
| Piper | Si | Buona | Molto bassa | Leggero, offline |
| Kokoro | Si | Alta | Media | Multilingua |
| ElevenLabs | No (API) | Eccellente | Media | Richiede API key a pagamento |
| VibeVoice | Si | Eccellente | Bassa | Richiede GPU |
| Chatterbox | Si | Eccellente | Media | Voice cloning |
| Coqui | Si | Alta | Media | Open source |

Puoi cambiare motore e voce in tempo reale; il cambio viene applicato immediatamente.

---

## Configurazione Whisper STT

Dal tab "Whisper":
- **Modello**: tiny, base, small, medium, large-v3 (piu' grande = piu' preciso ma piu' lento)
- **Lingua**: Seleziona la lingua di riconoscimento
- **Auto-detect**: Rileva automaticamente la lingua parlata

---

## Configurazione SIP / Telefonia

Dal tab "SIP":
- Visualizza lo stato del trunk SIP (connesso/disconnesso)
- Configura parametri SIP (trunk, regole dispatch)
- Definisci contesti per numero di telefono (risposte personalizzate per diversi numeri)
- Testa la configurazione SIP

---

## Calibrazione Audio

Il pannello di calibrazione audio e' accessibile dalla pagina chat e permette di ottimizzare il riconoscimento vocale.

### Parametri

| Parametro | Descrizione | Default Web | Default SIP |
|-----------|-------------|-------------|-------------|
| VAD Energy Threshold | Soglia energia per rilevare attivita' vocale | 120 | 120 |
| Speech Energy Threshold | Soglia energia per riconoscere parlato | 25 | 100 |
| Silence Threshold | Durata silenzio (frame) per fine parlato | 60 | 30 |
| TTS Cooldown | Secondi di attesa dopo TTS prima di riascoltare | 5 | 5 |

### Calibrazione Automatica

1. Assicurati che l'ambiente sia in condizioni normali (rumore di fondo tipico)
2. Clicca "Calibra" nel pannello audio
3. Il sistema registra 3 secondi di rumore ambientale
4. Le soglie vengono calcolate e applicate automaticamente
5. Le impostazioni sono salvate separatamente per canale Web e SIP

---

## Log Chiamate

Dal tab "Log Chiamate":
- Visualizza tutte le chiamate SIP ricevute
- Filtra per stato (attiva, completata, fallita, persa)
- Clicca su una chiamata per vedere il dettaglio:
  - Numero chiamante/chiamato
  - Durata
  - Trascrizione completa della conversazione
  - Metadati

---

## Ascolto Live e Debug

Dal tab "Ascolto Live":
- Apri la pagina principale in un iframe per monitorare le conversazioni
- Utile per fare debug e verificare la qualita' delle risposte
- Disponibile anche in modalita' fullscreen

### Statistiche Real-time (Tab Statistiche)

- **STT**: Tempo di trascrizione (media, ultimo)
- **LLM**: Tempo di risposta, Time-to-First-Token
- **TTS**: Tempo di sintesi vocale
- **Latenza totale**: Tempo end-to-end dalla voce dell'utente alla risposta
