"""
Main Voice Agent
Agent principale che orchestra STT, LLM e TTS per conversazioni vocali.
"""

import asyncio
from datetime import datetime, timezone
import os
import re
import sys
import time
import urllib.parse
import uuid
import threading
import queue
from typing import Optional, Callable

import json
import aiohttp
from loguru import logger


from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    AutoSubscribe,
    cli,
    llm,
    stt,
    tts,
    APIConnectOptions,
)
from livekit.agents.utils import AudioBuffer
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.llm import function_tool
from livekit.agents import RunContext
from livekit.plugins import silero, openai
from livekit import rtc
import base64
import io
from PIL import Image

from .config import config
from .llm.remote_llm import RemoteLLM

# Callback globale per inviare messaggi al frontend
_send_transcript_callback = None
_sent_messages = set()  # Per evitare duplicati (memorizza message_id)
_sent_message_ids = set()  # Set di ID messaggi già inviati
_last_user_message = ""  # Per evitare duplicati STT
_detected_language = "it"  # Lingua rilevata da Whisper (default italiano)
_last_stt_end_time = None  # Timestamp fine STT per calcolo latenza
_last_stt_time_ms = 0  # Ultimo tempo STT in ms (per conversation tracking)
_last_tts_time_ms = 0  # Ultimo tempo TTS in ms (per conversation tracking)
_last_llm_ttft_ms = 0  # Ultimo LLM Time To First Token in ms
_component_info = {"stt": "whisper", "llm": "", "tts": ""}  # Info componenti attivi
_message_counter = 0  # Contatore progressivo per ID messaggi

# Variabili per tracciamento chiamate SIP
_current_call_log_id = None  # ID del log chiamata attivo (se è una chiamata SIP)
_is_sip_call = False  # True se siamo in una chiamata SIP

# Anti-duplicazione STT avanzata: traccia hash + timestamp degli ultimi messaggi
_stt_recent_hashes = {}  # hash -> timestamp di quando è stato processato
_STT_DEDUP_WINDOW_SECONDS = 5.0  # Ignora testi identici entro N secondi

# Variabili globali per pattern matching comandi video (fallback per modelli senza function calling)
_video_analysis_callback = None  # Callback per analisi video
_agent_session_global = None  # Sessione agent per TTS

# Variabili globali per gestione multi-utente
_human_participants_count = 1  # Numero di partecipanti umani (esclude agent)
_force_agent_response = False  # Se True, l'agent risponde sempre (toggle dal frontend)
_room_context = None  # Riferimento al contesto della room per contare partecipanti

# Webhook n8n per disponibilità camere
ROOM_AVAILABILITY_WEBHOOK_URL = os.getenv(
    "ROOM_AVAILABILITY_WEBHOOK_URL",
    "https://n8n.immodrone.it/webhook/aida-search-room-v2",
)

# ==================== WAKE WORD SYSTEM ====================
# Struttura per wake session per utente
# Formato: {participant_id: {"active": bool, "last_activity": float, "expires_at": float}}
from typing import Dict
_wake_sessions: Dict[str, dict] = {}
_wake_countdown_task = None  # Task asincrono per countdown
_send_wake_callback = None  # Callback per inviare aggiornamenti wake al frontend

# ==================== CONFIGURABLE VOICE SETTINGS ====================
# Questi valori vengono caricati dal database all'avvio
# Default values (saranno sovrascritti da load_voice_settings_from_db)
WAKE_TIMEOUT_SECONDS = 30  # Timeout di silenzio per disattivazione automatica
VAD_ENERGY_THRESHOLD = 120  # Soglia energia per barge-in VAD (alzata per evitare falsi positivi)
SPEECH_ENERGY_THRESHOLD = 25  # Soglia energia per rilevamento parlato (molto sensibile)
SILENCE_THRESHOLD = 60  # Frames di silenzio prima di terminare ascolto (~3s per frasi complete)

# ==================== BRANDING CONFIGURATION ====================
# Nome assistente e trigger (caricati da variabili ambiente)
from agent.config import config
ASSISTANT_NAME = config.branding.assistant_name
ASSISTANT_TRIGGERS = config.branding.assistant_triggers

# Pattern regex fuzzy per riconoscere wake word con varianti Whisper
# Cattura: "hey sophy", "ehi sophie", "e sofi", "a softie", "soffì", "e i soffi", "safi", ecc.
WAKE_WORD_PATTERNS = [
    # Pattern principale: prefisso opzionale + varianti di "sophy/sofi/sophie/safi"
    r'(hey|ehi|ei|e\s*i?|a|ok|ciao|ge|ghe)\s*,?\s*(soph[yie]+|sof[fìiy]+n?[ie]*|soft[iye]+|isof[iy]|saf[iy])',
    # Varianti con "e i" separato (Whisper spesso trascrive così)
    r'e\s+i\s+sof',
    # Varianti con spazi
    r'(hey|ehi)\s+soph',
    r'(hey|ehi)\s+sof',
    r'(hey|ehi)\s+saf',
    # Varianti scritte insieme
    r'(heysoph|ehisoph|eisoph)',
    r'(heysofi|ehisofi|eisofi)',
    r'(heysafi|ehisafi)',
    # Fallback per "sofi/soffi/soffini/safi" isolato con prefisso
    r'\b(e|a|ei)\s+sof[fiy]+n?[ie]*\b',
    r'\b(e|a|ei)\s+saf[iy]\b',
    # Varianti "soffini", "soffin", "soffi" 
    r'\bsoff[iy]n[ie]?\b',
    # Varianti russe/cirilliche che Whisper può generare
    r'офie',
]

# ==================== TTS INTERRUPT SYSTEM ====================
# NOTA: Usa FILE come flag invece di variabile globale
# perché LiveKit agents può isolare le variabili tra processi/task
_TTS_FLAG_FILE = "/tmp/sophyai_tts_speaking.flag"
_TTS_END_TIME_FILE = "/tmp/sophyai_tts_end_time.txt"
TTS_COOLDOWN_SECONDS = 5.0  # Scarta audio per Ns dopo fine TTS (configurabile da DB)


def set_tts_speaking(speaking: bool):
    """Imposta lo stato di speaking del TTS usando un file flag"""
    import os
    
    
    try:
        if speaking:
            # Crea il file flag
            with open(_TTS_FLAG_FILE, "w") as f:
                f.write("1")
            # Rimuovi il file di end time se esiste
            if os.path.exists(_TTS_END_TIME_FILE):
                os.remove(_TTS_END_TIME_FILE)
            logger.debug("🔊 TTS iniziato (file flag creato)")
        else:
            # Rimuovi il file flag
            if os.path.exists(_TTS_FLAG_FILE):
                os.remove(_TTS_FLAG_FILE)
            # Salva il timestamp di fine TTS per il cooldown
            with open(_TTS_END_TIME_FILE, "w") as f:
                f.write(str(time.time()))
            logger.debug("🔊 TTS terminato (file flag rimosso, cooldown iniziato)")
    except Exception as e:
        logger.error(f"Errore gestione file flag TTS: {e}")


def is_tts_speaking() -> bool:
    """Ritorna True se il TTS sta parlando (controlla file flag)"""
    import os
    result = os.path.exists(_TTS_FLAG_FILE)
    
    
    return result


def is_in_tts_cooldown() -> bool:
    """Ritorna True se siamo nel periodo di cooldown dopo il TTS"""
    import os
    if not os.path.exists(_TTS_END_TIME_FILE):
        return False
    try:
        with open(_TTS_END_TIME_FILE, "r") as f:
            end_time = float(f.read().strip())
        elapsed = time.time() - end_time
        in_cooldown = elapsed < TTS_COOLDOWN_SECONDS
        
        
        return in_cooldown
    except:
        return False


async def interrupt_tts_if_speaking():
    """Interrompe il TTS se sta parlando"""
    global _agent_session_global
    
    if is_tts_speaking() and _agent_session_global:
        logger.info("✋ Interruzione automatica TTS - utente sta parlando")
        try:
            await _agent_session_global.interrupt()
            set_tts_speaking(False)  # Reset flag
            return True
        except Exception as e:
            logger.error(f"Errore interruzione TTS: {e}")
    return False


async def _async_interrupt_from_vad(session):
    """
    Funzione async chiamata dal thread VAD per interrompere il TTS.
    Questa funzione viene eseguita nel loop asyncio principale.
    """
    if not is_tts_speaking():
        return False
    
    logger.info("🎤 [VAD] Esecuzione interrupt dal thread VAD")
    try:
        # Interrompi il TTS
        result = session.interrupt()
        if asyncio.iscoroutine(result):
            await result
        
        # Reset flag e cancella LLM
        set_tts_speaking(False)
        request_cancel_llm()
        
        logger.info("🎤 [VAD] Interrupt eseguito con successo")
        return True
    except Exception as e:
        logger.error(f"🎤 [VAD] Errore durante interrupt: {e}")
        return False


# ==================== LLM CANCELLATION SYSTEM ====================
_cancel_llm_response = False  # Flag per annullare risposte LLM in corso


def request_cancel_llm():
    """Richiede la cancellazione della risposta LLM in corso"""
    global _cancel_llm_response
    _cancel_llm_response = True
    logger.info("🛑 Richiesta cancellazione LLM")


def should_cancel_llm() -> bool:
    """
    Controlla se la risposta LLM deve essere cancellata.
    Resetta il flag dopo la lettura (one-shot).
    """
    global _cancel_llm_response
    if _cancel_llm_response:
        _cancel_llm_response = False
        return True
    return False


def reset_cancel_llm():
    """Resetta il flag di cancellazione LLM"""
    global _cancel_llm_response
    _cancel_llm_response = False


# ==================== VAD MONITOR (Thread Separato per Barge-in) ====================
class VADMonitor:
    """
    Monitora l'audio in un thread separato per rilevare barge-in.
    Questo thread gira indipendentemente dal loop asyncio principale,
    permettendo di rilevare la voce dell'utente anche durante il TTS.
    """
    
    def __init__(self, interrupt_callback: Callable[[], None], energy_threshold: float = 150):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._interrupt_callback = interrupt_callback
        self._audio_queue: queue.Queue = queue.Queue(maxsize=1000)  # Buffer limitato
        self._energy_threshold = energy_threshold
        self._last_interrupt_time = 0
        self._interrupt_cooldown = 1.0  # Minimo 1s tra interrupt (era 0.5s, troppo basso)
        self._consecutive_speech_frames = 0
        self._min_speech_frames = 6  # Richiedi almeno 6 frame consecutivi con voce (~300ms, era 3)
    
    def start(self):
        """Avvia il thread di monitoraggio VAD"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="VADMonitor")
        self._thread.start()
        logger.info("🎤 [VAD] Monitor thread avviato")
    
    def stop(self):
        """Ferma il thread di monitoraggio VAD"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("🎤 [VAD] Monitor thread fermato")
    
    def feed_audio(self, audio_data: bytes):
        """
        Alimenta il VAD monitor con dati audio.
        Chiamato dal loop audio principale per ogni frame.
        Non-blocking: scarta dati se la coda è piena.
        """
        if not self._running:
            return
        try:
            self._audio_queue.put_nowait(audio_data)
        except queue.Full:
            pass  # Scarta se la coda è piena (non bloccare mai il chiamante)
    
    def _calculate_energy(self, audio_data: bytes) -> float:
        """Calcola l'energia media dell'audio (16-bit PCM)"""
        try:
            samples = [int.from_bytes(audio_data[i:i+2], 'little', signed=True) 
                      for i in range(0, len(audio_data), 2)]
            if samples:
                return sum(abs(s) for s in samples) / len(samples)
        except Exception:
            pass
        return 0
    
    def _monitor_loop(self):
        """Loop principale del thread VAD"""
        logger.info("🎤 [VAD] Loop di monitoraggio avviato")
        frame_count = 0
        last_log_time = time.time()
        
        while self._running:
            try:
                # Attendi audio con timeout breve
                audio_data = self._audio_queue.get(timeout=0.05)
                frame_count += 1
                
                # Log ogni 2 secondi per debug
                now = time.time()
                if now - last_log_time >= 2.0:
                    tts_state = is_tts_speaking()
                    logger.info(f"🎤 [VAD] Frames ricevuti: {frame_count} negli ultimi 2s, TTS attivo: {tts_state}")
                    frame_count = 0
                    last_log_time = now
                
                # Calcola energia
                energy = self._calculate_energy(audio_data)
                
                # Se TTS è attivo e c'è voce significativa
                tts_on = is_tts_speaking()
                if tts_on and energy > self._energy_threshold:
                    self._consecutive_speech_frames += 1
                    
                    # Log per debug (ogni 5 frame per non spammare)
                    if self._consecutive_speech_frames % 5 == 1:
                        logger.debug(f"🎤 [VAD] Voce rilevata durante TTS: energia={energy:.0f}, threshold={self._energy_threshold}, frames={self._consecutive_speech_frames}/{self._min_speech_frames}")
                    
                    # Se abbastanza frame consecutivi con voce, interrompi
                    if self._consecutive_speech_frames >= self._min_speech_frames:
                        current_time = time.time()
                        if current_time - self._last_interrupt_time > self._interrupt_cooldown:
                            logger.info(f"🎤 [VAD] BARGE-IN RILEVATO! Energia={energy:.0f}, threshold={self._energy_threshold}, frames={self._consecutive_speech_frames}/{self._min_speech_frames}, cooldown={self._interrupt_cooldown}s")
                            self._interrupt_callback()
                            self._last_interrupt_time = current_time
                            self._consecutive_speech_frames = 0  # Reset
                elif tts_on and energy > 0:
                    # Reset contatore se non c'è voce o TTS non attivo
                    self._consecutive_speech_frames = 0
                else:
                    # Reset contatore se non c'è voce o TTS non attivo
                    self._consecutive_speech_frames = 0
                    
            except queue.Empty:
                # Timeout normale, continua
                self._consecutive_speech_frames = 0
                continue
            except Exception as e:
                logger.error(f"🎤 [VAD] Errore nel loop: {e}")
                time.sleep(0.1)  # Evita busy loop in caso di errori ripetuti
        
        logger.info("🎤 [VAD] Loop di monitoraggio terminato")


# Istanza globale del VAD monitor
_vad_monitor: Optional[VADMonitor] = None


def get_vad_monitor() -> Optional[VADMonitor]:
    """Ritorna l'istanza globale del VAD monitor"""
    return _vad_monitor


def set_human_participants_count(count: int):
    """Aggiorna il conteggio dei partecipanti umani"""
    global _human_participants_count
    _human_participants_count = count
    logger.info(f"👥 Partecipanti umani aggiornato: {count}")


def set_force_agent_response(force: bool):
    """Imposta se forzare la risposta dell'agent"""
    global _force_agent_response
    _force_agent_response = force
    logger.info(f"🔔 Forza risposta agent: {force}")


def get_should_require_mention() -> bool:
    """
    Determina se è richiesta la menzione @sophyai.
    Ritorna False se:
    - C'è solo 1 utente umano nella room
    - Il flag _force_agent_response è True
    """
    if _force_agent_response:
        return False
    if _human_participants_count <= 1:
        return False
    return True


# ==================== WAKE SESSION FUNCTIONS ====================

def set_wake_callback(callback):
    """Imposta il callback per inviare aggiornamenti wake al frontend"""
    global _send_wake_callback
    _send_wake_callback = callback
    logger.info("🎤 Wake callback impostato")


def is_wake_trigger(text: str) -> bool:
    """
    Verifica se il testo contiene un wake trigger usando pattern fuzzy.
    Gestisce varianti Whisper come "ehi sophie", "e sofi", "a softie", ecc.
    """
    import re
    
    # Normalizza: lowercase, rimuovi punteggiatura extra
    normalized = text.lower().strip()
    normalized = re.sub(r'[,.\-!?\'"]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Controlla ogni pattern
    for pattern in WAKE_WORD_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            logger.info(f"🎤 Wake trigger rilevato con pattern '{pattern}': '{text}'")
            return True
    
    return False


def get_wake_session(participant_id: str) -> dict:
    """Ottiene la wake session per un partecipante"""
    return _wake_sessions.get(participant_id, {"active": False, "last_activity": 0, "expires_at": 0})


def is_wake_active(participant_id: str = None) -> bool:
    """
    Verifica se c'è una wake session attiva E con timer partito.
    Se participant_id è None, controlla se qualsiasi sessione è attiva.
    NOTA: Una sessione è considerata attiva solo se timer_started=True.
    """
    current_time = time.time()
    
    if participant_id:
        session = _wake_sessions.get(participant_id)
        if (session and 
            session.get("active") and 
            session.get("timer_started", False) and 
            session.get("expires_at", 0) > current_time):
            return True
        return False
    
    # Controlla tutte le sessioni
    for pid, session in _wake_sessions.items():
        if (session.get("active") and 
            session.get("timer_started", False) and 
            session.get("expires_at", 0) > current_time):
            return True
    return False


def activate_wake_session(participant_id: str, start_timer: bool = False):
    """
    Attiva una wake session per un partecipante.
    Se start_timer=False, il timer NON parte subito (deve essere avviato dopo TTS "Dimmi").
    """
    global _wake_sessions
    current_time = time.time()
    
    # Se start_timer=False, expires_at è nel futuro lontano (timer non attivo)
    # Verrà impostato correttamente da start_wake_timer()
    expires_at = current_time + WAKE_TIMEOUT_SECONDS if start_timer else current_time + 9999
    
    _wake_sessions[participant_id] = {
        "active": True,
        "last_activity": current_time,
        "expires_at": expires_at,
        "timer_started": start_timer
    }
    
    logger.info(f"🎤 Wake session ATTIVATA per {participant_id} (timer_started={start_timer})")
    
    # Invia notifica al frontend
    if _send_wake_callback:
        remaining = WAKE_TIMEOUT_SECONDS if start_timer else 0
        asyncio.create_task(_send_wake_callback({
            "type": "wake_status",
            "active": True,
            "participant_id": participant_id,
            "remaining_seconds": remaining,
            "waiting_for_dimmi": not start_timer
        }))


def start_wake_timer(participant_id: str):
    """Avvia il timer per una wake session (da chiamare DOPO TTS 'Dimmi')"""
    global _wake_sessions
    
    if participant_id not in _wake_sessions:
        return False
    
    session = _wake_sessions[participant_id]
    if not session.get("active"):
        return False
    
    current_time = time.time()
    session["expires_at"] = current_time + WAKE_TIMEOUT_SECONDS
    session["timer_started"] = True
    session["last_activity"] = current_time
    
    logger.info(f"🎤 Wake timer AVVIATO per {participant_id} (scade tra {WAKE_TIMEOUT_SECONDS}s)")
    
    # Invia notifica al frontend
    if _send_wake_callback:
        asyncio.create_task(_send_wake_callback({
            "type": "wake_status",
            "active": True,
            "participant_id": participant_id,
            "remaining_seconds": WAKE_TIMEOUT_SECONDS,
            "waiting_for_dimmi": False
        }))
    
    return True


async def handle_wake_word_detected(participant_id: str):
    """
    Gestisce il rilevamento di un wake word:
    1. Attiva sessione (senza timer)
    2. Pronuncia "Dimmi"
    3. Avvia timer DOPO TTS
    """
    global _agent_session_global
    
    # Attiva sessione senza timer
    activate_wake_session(participant_id, start_timer=False)
    
    # Pronuncia "Dimmi" se abbiamo la sessione
    if _agent_session_global:
        try:
            set_tts_speaking(True)
            logger.info(f"🎤 Pronuncio 'Dimmi' per {participant_id}")
            await _agent_session_global.say("Dimmi")
            set_tts_speaking(False)
            
            # ORA avvia il timer
            start_wake_timer(participant_id)
            logger.info(f"🎤 Timer avviato dopo 'Dimmi' per {participant_id}")
        except Exception as e:
            set_tts_speaking(False)
            logger.error(f"Errore pronuncia 'Dimmi': {e}")
    else:
        # Fallback: avvia timer subito se non c'è sessione TTS
        logger.warning("⚠️ Nessuna sessione TTS disponibile, avvio timer subito")
        start_wake_timer(participant_id)


def refresh_wake_session(participant_id: str):
    """Resetta il timer di una wake session attiva"""
    global _wake_sessions
    
    if participant_id not in _wake_sessions:
        return False
    
    session = _wake_sessions[participant_id]
    if not session.get("active"):
        return False
    
    current_time = time.time()
    expires_at = current_time + WAKE_TIMEOUT_SECONDS
    
    session["last_activity"] = current_time
    session["expires_at"] = expires_at
    
    logger.debug(f"🎤 Wake session REFRESH per {participant_id} (scade tra {WAKE_TIMEOUT_SECONDS}s)")
    return True


def deactivate_wake_session(participant_id: str):
    """Disattiva una wake session"""
    global _wake_sessions
    
    if participant_id in _wake_sessions:
        _wake_sessions[participant_id]["active"] = False
        logger.info(f"🎤 Wake session DISATTIVATA per {participant_id}")
        
        # Invia notifica al frontend
        if _send_wake_callback:
            asyncio.create_task(_send_wake_callback({
                "type": "wake_status",
                "active": False,
                "participant_id": participant_id,
                "remaining_seconds": 0
            }))


def get_any_active_wake_participant() -> str:
    """Ritorna l'ID del primo partecipante con wake session attiva (timer partito), o None"""
    current_time = time.time()
    for pid, session in _wake_sessions.items():
        if (session.get("active") and 
            session.get("timer_started", False) and 
            session.get("expires_at", 0) > current_time):
            return pid
    return None


async def wake_countdown_loop():
    """
    Task asincrono che gestisce il countdown delle wake sessions.
    Invia aggiornamenti al frontend ogni secondo e disattiva sessioni scadute.
    NOTA: Solo sessioni con timer_started=True vengono contate e scadono.
    """
    global _wake_sessions
    
    logger.info("🎤 Wake countdown loop avviato")
    
    while True:
        try:
            await asyncio.sleep(1)  # Controlla ogni secondo
            
            current_time = time.time()
            sessions_to_deactivate = []
            
            for participant_id, session in _wake_sessions.items():
                if not session.get("active"):
                    continue
                
                # Ignora sessioni che aspettano ancora "Dimmi" (timer non partito)
                if not session.get("timer_started", False):
                    continue
                
                expires_at = session.get("expires_at", 0)
                remaining = int(expires_at - current_time)
                
                if remaining <= 0:
                    # Sessione scaduta
                    sessions_to_deactivate.append(participant_id)
                else:
                    # Invia countdown al frontend
                    if _send_wake_callback:
                        try:
                            await _send_wake_callback({
                                "type": "wake_countdown",
                                "participant_id": participant_id,
                                "remaining_seconds": remaining
                            })
                        except Exception as e:
                            logger.debug(f"Errore invio wake_countdown: {e}")
            
            # Disattiva sessioni scadute
            for participant_id in sessions_to_deactivate:
                logger.info(f"🎤 Wake session SCADUTA per {participant_id} (timeout {WAKE_TIMEOUT_SECONDS}s)")
                deactivate_wake_session(participant_id)
                
        except asyncio.CancelledError:
            logger.info("🎤 Wake countdown loop cancellato")
            break
        except Exception as e:
            logger.error(f"Errore nel wake countdown loop: {e}")
            await asyncio.sleep(1)


def start_wake_countdown_task():
    """Avvia il task di countdown wake (se non già avviato)"""
    global _wake_countdown_task
    
    if _wake_countdown_task is None or _wake_countdown_task.done():
        _wake_countdown_task = asyncio.create_task(wake_countdown_loop())
        logger.info("🎤 Wake countdown task avviato")
    
    return _wake_countdown_task


def stop_wake_countdown_task():
    """Ferma il task di countdown wake"""
    global _wake_countdown_task
    
    if _wake_countdown_task and not _wake_countdown_task.done():
        _wake_countdown_task.cancel()
        logger.info("🎤 Wake countdown task fermato")


def set_transcript_callback(callback):
    global _send_transcript_callback, _sent_messages, _sent_message_ids, _message_counter, _stt_recent_hashes, _last_user_message
    _send_transcript_callback = callback
    _sent_messages.clear()  # Reset quando si connette
    _sent_message_ids.clear()  # Reset ID messaggi
    _message_counter = 0  # Reset contatore
    _stt_recent_hashes.clear()  # Reset hash STT
    _last_user_message = ""  # Reset ultimo messaggio
    logger.info("🔄 Callback transcript impostato, tutti i set di dedup resettati")


def generate_message_id() -> str:
    """Genera un ID univoco per ogni messaggio"""
    global _message_counter
    _message_counter += 1
    # Formato: MSG-{timestamp_ms}-{counter}
    return f"MSG-{int(time.time() * 1000)}-{_message_counter}"

def set_video_analysis_callback(callback, session):
    """Imposta callback per gestire comandi video vocali (fallback per modelli senza function calling)"""
    global _video_analysis_callback, _agent_session_global
    _video_analysis_callback = callback
    _agent_session_global = session

def detect_video_command(text: str) -> str | None:
    """Rileva se il testo è un comando di analisi video. Ritorna il tipo o None."""
    text_lower = text.lower().strip()
    
    # Comandi per analisi generica
    if any(p in text_lower for p in ["cosa vedi", "che cosa vedi", "descrivi cosa vedi", "analizza il video", 
                                      "guarda il video", "cosa c'è nel video", "dimmi cosa vedi"]):
        return "generic"
    
    # Comandi per documenti
    if any(p in text_lower for p in ["leggi il documento", "analizza documento", "leggi la carta",
                                      "carta d'identità", "patente", "estrai i dati"]):
        return "document"
    
    # Comandi per età
    if any(p in text_lower for p in ["quanti anni", "età", "stima l'età", "che età ha"]):
        return "age"
    
    # Comandi per ambiente
    if any(p in text_lower for p in ["descrivi l'ambiente", "dove sono", "cosa c'è intorno",
                                      "descrivi la stanza", "descrivi il luogo"]):
        return "environment"
    
    return None


def should_agent_respond(text: str, participant_id: str = "default") -> tuple[bool, str, bool]:
    """
    Verifica se il messaggio deve attivare una risposta dell'agent.
    Ritorna (should_respond, testo_pulito, is_wake_trigger).
    
    Il sistema supporta:
    1. Wake word "Hey Sophy" - attiva sessione di ascolto per 20s
    2. Sessione wake attiva - risponde a tutto finché non scade
    3. Trigger espliciti (@sophyai, sophy, ecc.)
    4. Single user mode / force mode - risponde sempre
    
    is_wake_trigger è True solo se è stato rilevato un wake word (per non rispondere al wake stesso)
    """
    import re
    
    # Filtro per "hallucination" di Whisper - frasi spurie generate durante silenzio/rumore
    WHISPER_HALLUCINATIONS = [
        "sottotitoli e revisione a cura di qtss",
        "sottotitoli a cura di qtss",
        "sottotitoli creati dalla comunità di amara.org",
        "sottotitoli di amara.org",
        "grazie per aver guardato",
        "grazie per la visione",
        "iscriviti al canale",
        "metti mi piace",
        "lascia un commento",
        "thanks for watching",
        "subscribe to the channel",
        "like and subscribe",
        "thank you for watching",
        "music",
        "musica",
        "applausi",
        "applause",
        "silenzio",
        "...",
        "…",
    ]
    
    text_lower = text.lower().strip()

    # Ignora messaggi troppo corti o vuoti
    if len(text_lower) < 3:
        logger.debug(f"🔇 Messaggio troppo corto ignorato: '{text}'")
        return (False, text, False)
    
    # Ignora hallucination di Whisper
    for hallucination in WHISPER_HALLUCINATIONS:
        if hallucination in text_lower or text_lower in hallucination:
            logger.warning(f"🔇 Whisper hallucination ignorato: '{text}'")
            return (False, text, False)
    
    # ==================== SIP AUTO-RESPONSE ====================
    # Per chiamate SIP, rispondi SEMPRE automaticamente
    # I partecipanti SIP hanno identity che inizia con "sip_" (es: sip_+390111951786)
    if participant_id.startswith("sip_"):
        logger.info(f"📞 [SIP] Auto-risposta per {participant_id}: '{text[:50]}...'")
        return (True, text, False)
    
    # ==================== WAKE WORD DETECTION ====================
    # Controlla se è un wake trigger ("Hey Sophy", "Ehi Sophy", ecc.)
    if is_wake_trigger(text):
        # NON attivare sessione qui - sarà gestito da handle_wake_word_detected
        # Ritorna flag speciale: is_wake=True, should_respond=False
        logger.info(f"🎤 WAKE WORD rilevato da {participant_id}: '{text}'")
        # Ritorna (False, "", True) - non rispondere all'LLM, ma segnala wake word
        return (False, "", True)
    
    # ==================== WAKE SESSION CHECK ====================
    # Se c'è una wake session attiva per questo partecipante, rispondi e resetta timer
    if is_wake_active(participant_id):
        refresh_wake_session(participant_id)
        logger.info(f"🎤 Wake session attiva per {participant_id}, rispondo a: '{text[:50]}...'")
        return (True, text, False)
    
    # Controlla anche se c'è una wake session attiva per qualsiasi partecipante
    # (utile in single-user mode dove non abbiamo sempre l'ID)
    active_participant = get_any_active_wake_participant()
    if active_participant:
        refresh_wake_session(active_participant)
        logger.info(f"🎤 Wake session attiva (partecipante: {active_participant}), rispondo a: '{text[:50]}...'")
        return (True, text, False)
    
    # ==================== STANDARD TRIGGERS (FALLBACK) ====================
    # Il wake word è il metodo principale. I trigger testuali (@sophyai) sono solo fallback
    # per messaggi scritti, NON per il parlato
    
    # Se force mode è attivo dal pulsante, rispondi sempre (ma solo con pulsante, non single user)
    if _force_agent_response:
        logger.info(f"🔔 Agent risponde (force mode attivo dal pulsante): '{text[:50]}...'")
        return (True, text, False)

    # Trigger testuali (solo per chat scritta, non per parlato)
    # Questi sono meno prioritari del wake word
    # I trigger sono caricati dalla configurazione branding
    triggers = ASSISTANT_TRIGGERS

    for trigger in triggers:
        if trigger in text_lower:
            # Rimuovi il trigger dal testo per una risposta più naturale
            cleaned_text = re.sub(re.escape(trigger), '', text, flags=re.IGNORECASE).strip()
            cleaned_text = re.sub(r'^[,\s]+', '', cleaned_text).strip()
            logger.info(f"🔔 Agent attivato con trigger testuale '{trigger}': '{text[:50]}...'")
            return (True, cleaned_text if cleaned_text else text, False)

    # Nessun wake word attivo e nessun trigger trovato
    logger.debug(f"🔕 Nessun wake word/trigger attivo, ignoro: '{text[:50]}...'")
    return (False, text, False)

async def send_timing_to_server(timing_type: str, data: dict):
    """Invia timing stats al web server"""
    import aiohttp
    import ssl
    try:
        # Usa HTTPS con certificato self-signed
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            payload = {timing_type: data}
            async with session.post(
                "https://host.docker.internal:8443/api/timing",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=2)
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"Timing send failed: {resp.status}")
    except Exception as e:
        logger.debug(f"Timing send error: {e}")


async def send_conversation_to_server(conversation_data: dict):
    """Invia record conversazione completo al web server"""
    import aiohttp
    import ssl
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                "https://host.docker.internal:8443/api/conversations",
                json=conversation_data,
                timeout=aiohttp.ClientTimeout(total=2)
            ) as resp:
                if resp.status != 200:
                    logger.debug(f"Conversation send failed: {resp.status}")
    except Exception as e:
        logger.debug(f"Conversation send error: {e}")


async def send_transcript(text: str, role: str, message_id: str = None):
    """Invia trascrizione al frontend (con deduplicazione basata su ID)"""
    global _sent_messages, _sent_message_ids, _last_user_message
    
    if not text or not text.strip():
        return
    
    # Genera ID se non fornito
    if not message_id:
        message_id = generate_message_id()
    
    # Controlla se questo ID è già stato inviato (deduplicazione primaria)
    if message_id in _sent_message_ids:
        logger.warning(f"⚠️ DUPLICATO ID IGNORATO: {message_id} - {text[:30]}...")
        return
        
    # Crea chiave univoca per deduplicazione secondaria (testo+ruolo)
    msg_key = f"{role}:{text.strip()}"
    
    logger.info(f"📨 send_transcript: id={message_id}, role={role}, text='{text[:40]}...'")
    
    # Per messaggi utente, controlla anche similarità (anti-doppio STT)
    if role == "user":
        if text.strip() == _last_user_message:
            logger.warning(f"⚠️ DUPLICATO USER (stesso testo) IGNORATO: {text[:30]}...")
            return
        _last_user_message = text.strip()
    
    # Evita duplicati esatti per contenuto (fallback)
    if msg_key in _sent_messages:
        logger.warning(f"⚠️ DUPLICATO CONTENUTO IGNORATO: {text[:30]}...")
        return
    
    # Registra come inviato
    _sent_message_ids.add(message_id)
    _sent_messages.add(msg_key)
    logger.info(f"✅ Messaggio {message_id} aggiunto (ids={len(_sent_message_ids)}, keys={len(_sent_messages)})")
    
    # Limita dimensione dei set (evita memory leak)
    if len(_sent_messages) > 100:
        logger.info("🗑️ Set messaggi troppo grande, reset...")
        _sent_messages.clear()
    if len(_sent_message_ids) > 100:
        logger.info("🗑️ Set ID troppo grande, reset...")
        _sent_message_ids.clear()
    
    if _send_transcript_callback:
        try:
            await _send_transcript_callback(text, role, message_id)
        except Exception as e:
            logger.error(f"Errore invio trascrizione: {e}")
    
    # Salva nel database se è una chiamata SIP
    if _is_sip_call and _current_call_log_id:
        try:
            import aiohttp
            import os
            server_url = os.getenv("WEB_SERVER_URL", "http://voice-agent-web:8080")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{server_url}/api/calls/{_current_call_log_id}/message",
                    params={"role": role, "content": text}
                ) as resp:
                    if resp.status == 200:
                        logger.debug(f"📝 Messaggio salvato nel log chiamata")
                    else:
                        logger.warning(f"⚠️ Errore salvataggio messaggio: {resp.status}")
        except Exception as e:
            logger.warning(f"⚠️ Impossibile salvare messaggio nel log: {e}")


# Configura logging
logger.remove()
logger.add(
    sys.stderr,
    level=config.server.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)


class OllamaLLM(llm.LLM):
    """LLM che usa Ollama"""
    
    def __init__(self, model_name: str = "gpt-oss", host: str = "http://localhost:11434"):
        super().__init__()
        self._model_name = model_name
        self._host = host
        self._client = None
    
    async def _ensure_client(self):
        if self._client is None:
            import ollama
            self._client = ollama.AsyncClient(host=self._host)
        return self._client
    
    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list | None = None,
        conn_options: APIConnectOptions = APIConnectOptions(),
        parallel_tool_calls: bool | None = None,
        tool_choice: llm.ToolChoice | None = None,
        extra_kwargs: dict | None = None,
    ) -> "OllamaLLMStream":
        # Ritorna direttamente lo stream (non async)
        return OllamaLLMStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class OllamaLLMStream(llm.LLMStream):
    """Stream di risposta da Ollama"""
    
    def __init__(
        self,
        llm_instance: OllamaLLM,
        chat_ctx: llm.ChatContext,
        tools: list,
        conn_options: APIConnectOptions
    ):
        super().__init__(llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._llm = llm_instance
        self._chat_ctx = chat_ctx
    
    async def _run(self) -> None:
        logger.info("OllamaLLM._run() iniziato")
        client = await self._llm._ensure_client()
        
        # Converti messaggi in formato Ollama
        messages = []
        for msg in self._chat_ctx.items:
            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                role = "assistant" if msg.role == "assistant" else "user"
                content = ""
                if isinstance(msg.content, str):
                    content = msg.content
                elif isinstance(msg.content, list):
                    for c in msg.content:
                        if isinstance(c, str):
                            content += c
                        elif hasattr(c, 'text'):
                            content += c.text
                if content:
                    messages.append({"role": role, "content": content})
        
        if not messages:
            messages = [{"role": "user", "content": "Ciao"}]
        
        logger.info(f"OllamaLLM: invio {len(messages)} messaggi a {self._llm._model_name}")
        
        try:
            response = await client.chat(
                model=self._llm._model_name,
                messages=messages,
                stream=True,
                keep_alive=-1
            )
            
            logger.info("OllamaLLM: risposta ricevuta, inizio streaming")
            chunk_id = str(uuid.uuid4())
            full_response = ""
            async for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        full_response += content
                        self._event_ch.send_nowait(
                            llm.ChatChunk(
                                id=chunk_id,
                                choices=[
                                    llm.ChoiceDelta(content=content, role="assistant"),
                                ]
                            )
                        )
            
            # Invia chunk finale con finish_reason
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=chunk_id,
                    choices=[
                        llm.ChoiceDelta(content="", role="assistant", finish_reason="stop"),
                    ]
                )
            )
            logger.info(f"OllamaLLM: risposta completa ({len(full_response)} chars), inviato finish")
        except Exception as e:
            logger.error(f"Errore Ollama: {e}")
            import traceback
            traceback.print_exc()


class RemoteLLMAdapter(llm.LLM):
    """
    Adapter LiveKit-compatible per server LLM remoti custom.
    Converte l'interfaccia LiveKit LLM nel formato del server remoto.
    """
    
    def __init__(
        self,
        server_url: str,
        token: str = "",
        collection: str = "",
        fallback_model: str = "devstral-small-2:latest"
    ):
        super().__init__()
        self._server_url = server_url
        self._token = token
        self._collection = collection
        self._fallback_model = fallback_model
        self._remote_llm = RemoteLLM(
            server_url=server_url,
            token=token,
            collection=collection
        )
        logger.info(f"RemoteLLMAdapter inizializzato: url={server_url}, collection={collection}")
    
    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool] | None = None,
        conn_options: APIConnectOptions = APIConnectOptions(),
        parallel_tool_calls: bool | None = None,
        tool_choice: llm.ToolChoice | None = None,
        extra_kwargs: dict | None = None,
    ) -> "RemoteLLMStream":
        return RemoteLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options or APIConnectOptions()
        )


class RemoteLLMStream(llm.LLMStream):
    """Stream di risposta dal server LLM remoto"""
    
    def __init__(
        self,
        llm_instance: RemoteLLMAdapter,
        chat_ctx: llm.ChatContext,
        tools: list,
        conn_options: APIConnectOptions
    ):
        super().__init__(llm_instance, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._llm = llm_instance
        self._chat_ctx = chat_ctx
    
    
    async def _run(self) -> None:
        logger.info("RemoteLLMStream._run() iniziato")
        
        # Estrai l'ultimo messaggio utente dal contesto
        user_message = ""
        
        for msg in reversed(list(self._chat_ctx.items)):
            if hasattr(msg, 'role') and msg.role == "user":
                if isinstance(msg.content, str):
                    user_message = msg.content
                elif isinstance(msg.content, list):
                    for c in msg.content:
                        if isinstance(c, str):
                            user_message += c
                        elif hasattr(c, 'text'):
                            user_message += c.text
                if user_message:
                    break
        
        if not user_message:
            user_message = "Ciao"
        
        
        logger.info(f"RemoteLLM: invio messaggio al server remoto: {user_message[:50]}...")
        
        try:
            # Chiama il server remoto
            response = await self._llm._remote_llm.chat(user_message)
            
            
            if response.text:
                logger.info(f"RemoteLLM: risposta ricevuta ({len(response.text)} chars)")
                chunk_id = str(uuid.uuid4())
                
                # Invia la risposta come singolo chunk (il server remoto non supporta streaming)
                # NOTA: ChatChunk usa 'delta' (singolo), NON 'choices' (lista)
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=chunk_id,
                        delta=llm.ChoiceDelta(content=response.text, role="assistant")
                    )
                )
                
                # Invia chunk finale con finish_reason
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=chunk_id,
                        delta=llm.ChoiceDelta(content="", role="assistant", finish_reason="stop")
                    )
                )
                logger.info(f"RemoteLLM: risposta inviata, finish")
            else:
                logger.warning("RemoteLLM: risposta vuota dal server")
                # Invia messaggio di errore
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=str(uuid.uuid4()),
                        delta=llm.ChoiceDelta(content="Mi dispiace, non ho ricevuto risposta dal server.", role="assistant", finish_reason="stop")
                    )
                )
                
        except Exception as e:
            logger.error(f"Errore RemoteLLM: {e}")
            import traceback
            traceback.print_exc()
            # Invia messaggio di errore
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=str(uuid.uuid4()),
                    delta=llm.ChoiceDelta(content=f"Errore di connessione al server remoto: {str(e)}", role="assistant", finish_reason="stop")
                )
            )


class ExternalTTSLiveKit(tts.TTS):
    """
    Wrapper LiveKit-compatibile generico per TTS esterni.
    
    Usa un server TTS esterno (tts_server.py) per la sintesi,
    permettendo di sfruttare GPU/MPS sul host invece del container.
    Supporta: kokoro, piper, vibevoice, chatterbox, edge
    """

    SUPPORTED_LANGUAGES = {
        "it": "it-IT", "en": "en-US", "zh": "zh-CN",
        "es": "es-ES", "fr": "fr-FR", "de": "de-DE"
    }

    def __init__(
        self,
        engine: str = "edge",
        model: str = None,
        language: str = "it",
        speaker: str = None,
        speed: float = 1.0,
        auto_language: bool = True,
        tts_server_url: str = None
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self.engine = engine
        self.model_name = model
        self.language = language
        self.speaker = speaker
        self.speed = speed
        self.auto_language = auto_language

        # URL del server TTS esterno
        self.tts_server_url = tts_server_url or os.getenv("TTS_SERVER_URL", "http://host.docker.internal:8092")
        self._server_available = None

        logger.info(f"🎤 ExternalTTSLiveKit inizializzato: engine={engine}, server={self.tts_server_url}, language={language}")

    async def _check_server(self) -> bool:
        """Verifica se il server TTS è disponibile"""
        if self._server_available is not None:
            return self._server_available

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.tts_server_url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"🎤 TTS Server disponibile: engine={data.get('engine')}, device={data.get('device')}")
                        self._server_available = True
                        return True
        except Exception as e:
            logger.warning(f"⚠️ TTS Server non disponibile: {e}")
        
        self._server_available = False
        return False
    
    def get_current_language(self) -> str:
        """Ritorna la lingua corrente (globale se auto_language)"""
        if self.auto_language:
            return _detected_language or self.language
        return self.language
    
    def synthesize(self, text: str, *, conn_options: APIConnectOptions = APIConnectOptions()) -> "ExternalTTSStream":
        # Se auto_language è attivo, usa la lingua rilevata
        current_lang = self.get_current_language()
        if current_lang != self.language:
            logger.info(f"🎤 [{self.engine}] Cambio lingua: {self.language} → {current_lang}")
            self.language = current_lang
        
        return ExternalTTSStream(self, text, conn_options)


class ExternalTTSStream(tts.ChunkedStream):
    """Stream audio da TTS esterno"""
    
    def __init__(self, tts_instance: ExternalTTSLiveKit, text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)
        self._tts_instance = tts_instance
        self._text = text
    
    async def _run(self, output_emitter=None) -> None:
        import subprocess
        
        try:
            t_tts_start = time.time()
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            text_preview = self._text[:50] + "..." if len(self._text) > 50 else self._text
            engine = self._tts_instance.engine
            logger.info(f"🎤 [{engine}] [{timestamp}] Sintesi ({len(self._text)} chars): \"{text_preview}\"")
            
            # Genera ID univoco per questo messaggio TTS
            tts_message_id = generate_message_id()
            
            # Invia transcript con ID
            asyncio.create_task(send_transcript(self._text, "assistant", tts_message_id))
            
            pcm_data = None
            
            # Prova il server TTS esterno
            try:
                server_available = await self._tts_instance._check_server()
                
                if server_available:
                    # Usa il server TTS esterno
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            "text": self._text,
                            "language": self._tts_instance.language,
                            "engine": engine
                        }
                        if self._tts_instance.speaker:
                            payload["speaker"] = self._tts_instance.speaker
                        if self._tts_instance.speed:
                            payload["speed"] = self._tts_instance.speed
                        if self._tts_instance.model_name:
                            payload["model"] = self._tts_instance.model_name
                        
                        async with session.post(
                            f"{self._tts_instance.tts_server_url}/synthesize",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as resp:
                            if resp.status == 200:
                                pcm_data = await resp.read()
                                actual_engine = resp.headers.get("X-Engine", engine)
                                logger.info(f"🎤 [{engine}] Sintesi via TTS Server (engine usato={actual_engine})")
                            else:
                                error = await resp.text()
                                raise Exception(f"TTS Server error: {error}")
                else:
                    raise Exception("TTS Server non disponibile")
                    
            except Exception as e:
                # Fallback a Edge TTS locale
                logger.warning(f"⚠️ TTS Server non disponibile ({e}), uso Edge TTS locale")
                
                import edge_tts
                
                # Mappa lingua a voce Edge
                edge_voices = {
                    "it": "it-IT-DiegoNeural",
                    "en": "en-US-GuyNeural",
                    "zh": "zh-CN-YunxiNeural",
                    "es": "es-ES-AlvaroNeural",
                    "fr": "fr-FR-HenriNeural",
                    "de": "de-DE-ConradNeural"
                }
                voice = edge_voices.get(self._tts_instance.language, "it-IT-DiegoNeural")
                
                communicate = edge_tts.Communicate(self._text, voice)
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                
                # Converti MP3 in PCM
                process = subprocess.Popen(
                    ['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ar', '24000', '-ac', '1', 'pipe:1'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                pcm_data, _ = process.communicate(audio_data)
            
            t_tts_end = time.time()
            tts_time_ms = (t_tts_end - t_tts_start) * 1000
            duration = len(pcm_data) / (24000 * 2) if pcm_data else 0
            logger.info(f"🎤 [{engine}] Tempo: {tts_time_ms:.0f}ms | Audio: {duration:.2f}s")
            
            # ⏱️ Salva TTS time per conversation tracking
            global _last_tts_time_ms
            _last_tts_time_ms = tts_time_ms
            
            # ⏱️ LATENCY: Calcola latenze dal fine parlato
            latency_ms = 0
            first_audio_ms = 0
            if _last_stt_end_time:
                # latency_ms = STT end → fine sintesi = tempo fino al primo audio udibile
                latency_ms = (t_tts_end - _last_stt_end_time) * 1000
                # first_audio_ms = uguale a latency_ms (audio pushato subito dopo sintesi)
                first_audio_ms = latency_ms
                logger.info(f"⚡ [LATENCY] Primo audio: {first_audio_ms:.0f}ms dopo fine parlato")
            
            # Emetti l'audio
            if pcm_data:
                import uuid
                req_id = str(uuid.uuid4())
                seg_id = str(uuid.uuid4())
                
                if output_emitter is not None:
                    # API 1.3.x - inizializza e usa output_emitter
                    output_emitter.initialize(
                        request_id=req_id,
                        sample_rate=24000,
                        num_channels=1,
                        mime_type="audio/pcm",
                        stream=True
                    )
                    output_emitter.start_segment(segment_id=seg_id)
                    output_emitter.push(pcm_data)
                    output_emitter.end_segment()
                    output_emitter.end_input()
                    
                    # Invia timing stats
                    asyncio.create_task(send_timing_to_server("tts", {
                        "time_ms": int(tts_time_ms),
                        "audio_sec": round(duration, 2)
                    }))
                    if first_audio_ms > 0:
                        asyncio.create_task(send_timing_to_server("latency", {
                            "e2e_ms": int(latency_ms),
                            "to_first_audio_ms": int(first_audio_ms)
                        }))
                else:
                    # Fallback API 1.0.x - usa _event_ch
                    import numpy as np
                    frame = rtc.AudioFrame(
                        data=pcm_data,
                        sample_rate=24000,
                        num_channels=1,
                        samples_per_channel=len(pcm_data) // 2
                    )
                    audio_event = tts.SynthesizedAudio(
                        frame=frame,
                        request_id=req_id,
                        is_final=True
                    )
                    await self._event_ch.send(audio_event)
                
        except Exception as e:
            logger.error(f"❌ [{self._tts_instance.engine}] Errore TTS: {e}")
            import traceback
            traceback.print_exc()


class VibeVoiceLiveKit(tts.TTS):
    """
    Wrapper LiveKit-compatibile per Microsoft VibeVoice TTS.
    
    Usa un server TTS esterno (tts_server.py) per la sintesi,
    permettendo di sfruttare GPU/MPS sul host invece del container.
    """

    SUPPORTED_LANGUAGES = {
        "it": "it-IT", "en": "en-US", "zh": "zh-CN",
        "es": "es-ES", "fr": "fr-FR", "de": "de-DE"
    }

    def __init__(
        self,
        model: str = "realtime",
        language: str = "it",
        speaker: str = "carter",
        speed: float = 1.0,
        auto_language: bool = True,
        tts_server_url: str = None
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self.model_name = model
        self.language = language
        self.speaker = speaker
        self.speed = speed
        self.auto_language = auto_language

        # URL del server TTS esterno
        self.tts_server_url = tts_server_url or os.getenv("TTS_SERVER_URL", "http://host.docker.internal:8092")
        self._server_available = None

        logger.info(f"🎤 VibeVoiceLiveKit inizializzato: server={self.tts_server_url}, language={language}, speaker={speaker}")
    
    async def _check_server(self) -> bool:
        """Verifica se il server TTS è disponibile"""
        if self._server_available is not None:
            return self._server_available
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.tts_server_url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"🎤 TTS Server disponibile: engine={data.get('engine')}, device={data.get('device')}")
                        self._server_available = True
                        return True
        except Exception as e:
            logger.warning(f"⚠️ TTS Server non disponibile: {e}")
        
        self._server_available = False
        return False
    
    def get_current_language(self) -> str:
        """Ritorna la lingua corrente (globale se auto_language)"""
        if self.auto_language:
            return _detected_language or self.language
        return self.language
    
    def synthesize(self, text: str, *, conn_options: APIConnectOptions = APIConnectOptions()) -> "VibeVoiceTTSStream":
        # Se auto_language è attivo, usa la lingua rilevata
        current_lang = self.get_current_language()
        if current_lang != self.language:
            logger.info(f"🎤 [VibeVoice] Cambio lingua: {self.language} → {current_lang}")
            self.language = current_lang
        
        stream = VibeVoiceTTSStream(self, text, conn_options)
        return stream


class VibeVoiceTTSStream(tts.ChunkedStream):
    """Stream audio da VibeVoice TTS (via server esterno o fallback Edge)"""
    
    def __init__(self, tts_instance: VibeVoiceLiveKit, text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)
        self._tts_instance = tts_instance
        self._text = text
    
    async def _run(self, output_emitter=None) -> None:
        import subprocess
        import uuid
        
        
        try:
            t_tts_start = time.time()
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            text_preview = self._text[:50] + "..." if len(self._text) > 50 else self._text
            logger.info(f"🎤 [VibeVoice] [{timestamp}] Sintesi ({len(self._text)} chars): \"{text_preview}\"")
            
            # Genera ID univoco per questo messaggio TTS
            tts_message_id = generate_message_id()
            
            # Invia transcript con ID
            asyncio.create_task(send_transcript(self._text, "assistant", tts_message_id))
            
            pcm_data = None
            
            # Prova il server TTS esterno
            try:
                server_available = await self._tts_instance._check_server()
                
                if server_available:
                    # Usa il server TTS esterno
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            "text": self._text,
                            "language": self._tts_instance.language,
                            "speaker": self._tts_instance.speaker,
                            "speed": self._tts_instance.speed,
                            "engine": "vibevoice"
                        }
                        
                        async with session.post(
                            f"{self._tts_instance.tts_server_url}/synthesize",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as resp:
                            if resp.status == 200:
                                pcm_data = await resp.read()
                                engine = resp.headers.get("X-Engine", "unknown")
                                logger.info(f"🎤 [VibeVoice] Sintesi via TTS Server (engine={engine})")
                            else:
                                error = await resp.text()
                                raise Exception(f"TTS Server error: {error}")
                else:
                    raise Exception("TTS Server non disponibile")
                    
            except Exception as e:
                # Fallback a Edge TTS locale
                logger.warning(f"⚠️ TTS Server non disponibile ({e}), uso Edge TTS locale")
                
                import edge_tts
                
                # Mappa lingua a voce Edge
                edge_voices = {
                    "it": "it-IT-DiegoNeural",
                    "en": "en-US-GuyNeural",
                    "es": "es-ES-AlvaroNeural",
                    "fr": "fr-FR-HenriNeural",
                    "de": "de-DE-ConradNeural",
                    "zh": "zh-CN-YunxiNeural",
                }
                voice = edge_voices.get(self._tts_instance.language, "it-IT-DiegoNeural")
                
                communicate = edge_tts.Communicate(self._text, voice)
                
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                
                # Converti MP3 in PCM
                process = subprocess.Popen(
                    ['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ar', '24000', '-ac', '1', 'pipe:1'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                pcm_data, _ = process.communicate(audio_data)
            
            t_tts_end = time.time()
            tts_time_ms = (t_tts_end - t_tts_start) * 1000
            
            if pcm_data:
                req_id = str(uuid.uuid4())
                seg_id = str(uuid.uuid4())
                
                frame = rtc.AudioFrame(
                    data=pcm_data,
                    sample_rate=24000,
                    num_channels=1,
                    samples_per_channel=len(pcm_data) // 2
                )
                
                audio_duration = len(pcm_data) / (24000 * 2)
                logger.info(f"🎤 [VibeVoice] Tempo: {tts_time_ms:.0f}ms | Audio: {audio_duration:.2f}s")
                
                # Emetti audio
                if output_emitter is not None:
                    output_emitter.initialize(
                        request_id=req_id,
                        sample_rate=24000,
                        num_channels=1,
                        mime_type="audio/pcm",
                        stream=True
                    )
                    output_emitter.start_segment(segment_id=seg_id)
                    output_emitter.push(pcm_data)
                    output_emitter.end_segment()
                    output_emitter.end_input()
                else:
                    audio_event = tts.SynthesizedAudio(
                        frame=frame,
                        request_id=req_id,
                        is_final=True
                    )
                    await self._event_ch.send(audio_event)
            else:
                pass
                    
        except Exception as e:
            logger.error(f"❌ [VibeVoice] Errore: {e}")
            raise


class EdgeTTS(tts.TTS):
    """TTS che usa Edge TTS (Microsoft) con selezione automatica della lingua"""
    
    # Voci per lingua
    VOICES_BY_LANGUAGE = {
        "it": "it-IT-DiegoNeural",
        "en": "en-US-GuyNeural",
        "es": "es-ES-AlvaroNeural",
        "fr": "fr-FR-HenriNeural",
        "de": "de-DE-ConradNeural",
        "zh": "zh-CN-YunxiNeural",
        "pt": "pt-BR-AntonioNeural",
        "ru": "ru-RU-DmitryNeural",
        "ja": "ja-JP-KeitaNeural",
        "ko": "ko-KR-InJoonNeural",
    }
    
    def __init__(self, voice: str = "it-IT-DiegoNeural", auto_language: bool = True):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=24000,
            num_channels=1,
        )
        self.default_voice = voice
        self.voice = voice
        self.auto_language = auto_language
        logger.info(f"EdgeTTS inizializzato con voce: {voice}, auto_language: {auto_language}")
    
    def get_voice_for_language(self, language: str) -> str:
        """Ritorna la voce appropriata per la lingua"""
        return self.VOICES_BY_LANGUAGE.get(language, self.default_voice)
    
    def synthesize(self, text: str, *, conn_options: APIConnectOptions = APIConnectOptions()) -> "EdgeTTSStream":
        # Se auto_language è attivo, usa la lingua rilevata globalmente
        if self.auto_language:
            current_voice = self.get_voice_for_language(_detected_language)
            if current_voice != self.voice:
                logger.info(f"🔊 [TTS] Cambio voce: {self.voice} → {current_voice} (lingua: {_detected_language})")
                self.voice = current_voice
        
        stream = EdgeTTSStream(self, text, conn_options)
        return stream


class EdgeTTSStream(tts.ChunkedStream):
    """Stream audio da Edge TTS"""
    
    def __init__(self, tts_instance: EdgeTTS, text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)
        self._tts_instance = tts_instance
        self._text = text
    
    async def _run(self, output_emitter=None) -> None:
        import edge_tts
        import subprocess
        import uuid
        
        
        try:
            # ⏱️ TIMING: Inizio TTS
            t_tts_start = time.time()
            
            # Timestamp assoluto per tracciare il flusso LLM→TTS
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            text_preview = self._text[:50] + "..." if len(self._text) > 50 else self._text
            text_len = len(self._text)
            logger.info(f"🔊 [TTS] [{timestamp}] Ricevuta frase ({text_len} chars): \"{text_preview}\"")
            
            # Genera ID univoco per questo messaggio TTS
            tts_message_id = generate_message_id()
            
            # Invia risposta agent al frontend con ID
            asyncio.create_task(send_transcript(self._text, "assistant", tts_message_id))
            
            communicate = edge_tts.Communicate(self._text, self._tts_instance.voice)
            
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            # ⏱️ TIMING: Fine download audio
            t_download_end = time.time()
            download_time_ms = (t_download_end - t_tts_start) * 1000
            
            if audio_data:
                # Converti MP3 in PCM
                t_convert_start = time.time()
                process = subprocess.Popen(
                    ['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ar', '24000', '-ac', '1', 'pipe:1'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL
                )
                pcm_data, _ = process.communicate(audio_data)
                t_convert_end = time.time()
                convert_time_ms = (t_convert_end - t_convert_start) * 1000
                
                if pcm_data:
                    req_id = str(uuid.uuid4())
                    seg_id = str(uuid.uuid4())
                    
                    # Crea AudioFrame
                    frame = rtc.AudioFrame(
                        data=pcm_data,
                        sample_rate=24000,
                        num_channels=1,
                        samples_per_channel=len(pcm_data) // 2
                    )
                    
                    # Prova entrambi i metodi per compatibilità
                    if output_emitter is not None:
                        # API 1.3.x
                        output_emitter.initialize(
                            request_id=req_id,
                            sample_rate=24000,
                            num_channels=1,
                            mime_type="audio/pcm",
                            stream=True
                        )
                        output_emitter.start_segment(segment_id=seg_id)
                        output_emitter.push(pcm_data)
                        output_emitter.end_segment()
                        output_emitter.end_input()
                    else:
                        # API 1.0.x
                        audio_event = tts.SynthesizedAudio(
                            frame=frame,
                            request_id=req_id,
                            is_final=True
                        )
                        await self._event_ch.send(audio_event)
                    
                    # ⏱️ TIMING: Fine TTS
                    t_tts_end = time.time()
                    total_tts_time_ms = (t_tts_end - t_tts_start) * 1000
                    audio_duration_sec = len(pcm_data) / 2 / 24000  # 2 bytes/sample, 24kHz
                    
                    # ⏱️ LATENCY: Tempo dalla fine domanda all'inizio risposta
                    latency_ms = 0
                    if _last_stt_end_time:
                        latency_ms = (t_tts_end - _last_stt_end_time) * 1000
                        logger.info(f"⚡ [LATENCY] Domanda→Risposta: {latency_ms:.0f}ms")
                    
                    logger.info(f"🔊 [TTS] Tempo totale: {total_tts_time_ms:.0f}ms (API: {download_time_ms:.0f}ms, Convert: {convert_time_ms:.0f}ms) | Audio: {audio_duration_sec:.2f}s | {len(pcm_data)} bytes")
                    
                    # Invia timing stats al server
                    asyncio.create_task(send_timing_to_server("tts", {
                        "time_ms": int(total_tts_time_ms),
                        "audio_sec": round(audio_duration_sec, 2)
                    }))
                    
                    # Invia latency stats
                    if latency_ms > 0:
                        asyncio.create_task(send_timing_to_server("latency", {
                            "e2e_ms": int(latency_ms),
                            "to_first_audio_ms": int(total_tts_time_ms)  # Tempo solo TTS
                        }))
                    
        except Exception as e:
            logger.error(f"Errore Edge TTS: {e}")
            import traceback
            traceback.print_exc()


class WhisperSTT(stt.STT):
    """STT che usa server Whisper esterno con accelerazione GPU/MPS"""
    
    # Mapping lingue per TTS
    LANGUAGE_TTS_VOICES = {
        "it": "it-IT-DiegoNeural",
        "en": "en-US-GuyNeural",
        "es": "es-ES-AlvaroNeural",
        "fr": "fr-FR-HenriNeural",
        "de": "de-DE-ConradNeural",
        "zh": "zh-CN-YunxiNeural",
        "pt": "pt-BR-AntonioNeural",
    }
    
    def __init__(self, model_size: str = "base", language: str = "it", auto_detect: bool = True):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self.model_size = model_size
        self.language = language
        self.auto_detect = auto_detect
        self.last_detected_language = language
        # URL del server Whisper (host.docker.internal per accedere all'host da Docker)
        self.whisper_url = os.environ.get("WHISPER_SERVER_URL", "http://host.docker.internal:8091")
        logger.info(f"WhisperSTT inizializzato: model={model_size}, lang={language}, auto_detect={auto_detect}, server={self.whisper_url}")
    
    async def transcribe_only(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """Trascrizione audio SENZA invio al frontend - per uso multi-partecipante"""
        import aiohttp
        import io
        import struct
        
        if not audio_bytes or len(audio_bytes) < 1600:  # Almeno 50ms
            return ""
        
        audio_duration_sec = len(audio_bytes) / 2 / sample_rate
        
        # ⏱️ TIMING STT: Inizio
        t_stt_start = time.time()
        text = ""
        
        try:
            # Crea file WAV in memoria
            wav_buffer = io.BytesIO()
            wav_buffer.write(b'RIFF')
            wav_buffer.write(struct.pack('<I', 36 + len(audio_bytes)))
            wav_buffer.write(b'WAVE')
            wav_buffer.write(b'fmt ')
            wav_buffer.write(struct.pack('<I', 16))
            wav_buffer.write(struct.pack('<H', 1))
            wav_buffer.write(struct.pack('<H', 1))
            wav_buffer.write(struct.pack('<I', sample_rate))
            wav_buffer.write(struct.pack('<I', sample_rate * 2))
            wav_buffer.write(struct.pack('<H', 2))
            wav_buffer.write(struct.pack('<H', 16))
            wav_buffer.write(b'data')
            wav_buffer.write(struct.pack('<I', len(audio_bytes)))
            wav_buffer.write(audio_bytes)
            wav_data = wav_buffer.getvalue()
            
            async with aiohttp.ClientSession() as http_session:
                form_data = aiohttp.FormData()
                form_data.add_field('audio', wav_data, filename='audio.wav', content_type='audio/wav')
                form_data.add_field('language', self.language)
                
                async with http_session.post(
                    f"{self.whisper_url}/transcribe",
                    data=form_data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result.get("text", "").strip()
                        detected_lang = result.get("language", "?")
                        lang_prob = result.get("language_probability", 0)
                        segments = result.get("segments", [])
                        
                        logger.info(f"🎤 [WHISPER] Risposta: '{text}' (lang={detected_lang}, prob={lang_prob:.0%}, segments={len(segments)})")
                    else:
                        error_text = await response.text()
                        logger.warning(f"Whisper server error: {response.status} - {error_text[:100]}")
                        text = ""
        except Exception as e:
            logger.error(f"Errore transcribe_only: {e}")
            text = ""
        
        # ⏱️ TIMING STT: Fine - imposta globali per latenza e invia stats
        global _last_stt_end_time, _last_stt_time_ms
        t_stt_end = time.time()
        _last_stt_end_time = t_stt_end
        stt_time_ms = (t_stt_end - t_stt_start) * 1000
        _last_stt_time_ms = stt_time_ms
        
        if text:
            logger.info(f"🎤 [STT] Tempo: {stt_time_ms:.0f}ms | Trascritto: \"{text[:50]}\"")
            asyncio.create_task(send_timing_to_server("stt", {"time_ms": int(stt_time_ms)}))
        
        return text
    
    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = None,
    ) -> stt.SpeechEvent:
        global _detected_language
        import aiohttp
        import io
        import struct
        
        # ⏱️ TIMING: Inizio ricezione audio
        t_audio_received = time.time()
        
        # Converti buffer in bytes WAV
        audio_bytes = bytes(buffer.data)
        sample_rate = getattr(buffer, 'sample_rate', 16000)
        audio_duration_sec = len(audio_bytes) / 2 / sample_rate  # 2 bytes per sample (int16)
        
        logger.info(f"📥 [AUDIO] Ricevuto: {len(audio_bytes)} bytes ({audio_duration_sec:.2f}s di audio)")
        
        # ⏱️ TIMING: Inizio STT
        t_stt_start = time.time()
        
        detected_lang = language or self.language
        text = ""
        
        try:
            # Crea file WAV in memoria
            wav_buffer = io.BytesIO()
            
            # WAV header
            num_samples = len(audio_bytes) // 2
            wav_buffer.write(b'RIFF')
            wav_buffer.write(struct.pack('<I', 36 + len(audio_bytes)))
            wav_buffer.write(b'WAVE')
            wav_buffer.write(b'fmt ')
            wav_buffer.write(struct.pack('<I', 16))  # Subchunk1Size
            wav_buffer.write(struct.pack('<H', 1))   # AudioFormat (PCM)
            wav_buffer.write(struct.pack('<H', 1))   # NumChannels
            wav_buffer.write(struct.pack('<I', sample_rate))  # SampleRate
            wav_buffer.write(struct.pack('<I', sample_rate * 2))  # ByteRate
            wav_buffer.write(struct.pack('<H', 2))   # BlockAlign
            wav_buffer.write(struct.pack('<H', 16))  # BitsPerSample
            wav_buffer.write(b'data')
            wav_buffer.write(struct.pack('<I', len(audio_bytes)))
            wav_buffer.write(audio_bytes)
            
            wav_data = wav_buffer.getvalue()
            
            # Invia al server Whisper
            async with aiohttp.ClientSession() as session:
                form = aiohttp.FormData()
                form.add_field('audio', wav_data, filename='audio.wav', content_type='audio/wav')
                
                # Se auto_detect è attivo, non passiamo la lingua per forzare il rilevamento
                if self.auto_detect:
                    form.add_field('language', '')  # Whisper rileverà automaticamente
                    form.add_field('detect_language', 'true')
                else:
                    form.add_field('language', language or self.language)
                
                async with session.post(
                    f"{self.whisper_url}/transcribe",
                    data=form,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        text = result.get("text", "").strip()
                        detected_lang = result.get("language", self.language)
                        lang_probability = result.get("language_probability", 0)
                        whisper_duration = result.get("duration", 0)
                        
                        # Aggiorna lingua rilevata globalmente
                        if detected_lang and lang_probability > 0.7:
                            self.last_detected_language = detected_lang
                            _detected_language = detected_lang
                            logger.info(f"🌍 [LINGUA] Rilevata: {detected_lang} (confidenza: {lang_probability:.0%})")
                    else:
                        error = await resp.text()
                        logger.error(f"Errore server Whisper: {error}")
                        text = ""
            
        except aiohttp.ClientError as e:
            logger.error(f"Errore connessione server Whisper: {e}")
            # Fallback a Whisper locale se il server non è disponibile
            logger.info("Fallback a Whisper locale...")
            text, detected_lang = await self._local_transcribe_with_detection(buffer, language)
        except Exception as e:
            logger.error(f"Errore generico Whisper: {e}")
            text = ""
        
        # ⏱️ TIMING: Fine STT
        global _last_stt_end_time
        t_stt_end = time.time()
        _last_stt_end_time = t_stt_end  # Salva per calcolo latenza
        stt_time_ms = (t_stt_end - t_stt_start) * 1000
        
        logger.info(f"🎤 [STT] Tempo: {stt_time_ms:.0f}ms | Lingua: {detected_lang} | Trascritto: \"{text}\"")
        
        
        # Invia timing stats al server
        asyncio.create_task(send_timing_to_server("stt", {"time_ms": int(stt_time_ms)}))
        
        # Anti-duplicazione STT avanzata: controlla se questo testo è stato processato di recente
        if text:
            global _stt_recent_hashes
            import hashlib
            text_hash = hashlib.md5(text.strip().lower().encode()).hexdigest()
            current_time = time.time()
            
            # Pulisci hash vecchi (oltre la finestra di dedup)
            expired_hashes = [h for h, t in _stt_recent_hashes.items() if current_time - t > _STT_DEDUP_WINDOW_SECONDS]
            for h in expired_hashes:
                del _stt_recent_hashes[h]
            
            # Controlla se questo hash è già stato visto di recente
            if text_hash in _stt_recent_hashes:
                time_since = current_time - _stt_recent_hashes[text_hash]
                logger.warning(f"⚠️ DUPLICATO STT IGNORATO (stesso testo {time_since:.1f}s fa): '{text[:30]}...'")
                # Non inviare al frontend, ma restituisci comunque l'evento per l'LLM
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="", language=detected_lang)]
                )
            
            # Registra questo hash
            _stt_recent_hashes[text_hash] = current_time
            logger.info(f"📝 STT hash registrato: {text_hash[:8]}... (totale: {len(_stt_recent_hashes)} hashes)")
            
            # Invia trascrizione utente al frontend con ID univoco
            stt_message_id = generate_message_id()
            asyncio.create_task(send_transcript(text, "user", stt_message_id))
        
        # Intercetta comandi video vocali (fallback per modelli senza function calling come Gemma 3)
        if text and _video_analysis_callback:
            video_cmd = detect_video_command(text)
            if video_cmd:
                logger.info(f"📹 Comando video vocale rilevato: {video_cmd} - Gestione diretta")
                # Esegui analisi video in background (include TTS del risultato)
                asyncio.create_task(_video_analysis_callback(video_cmd))
                # Restituisci testo vuoto per evitare che l'LLM risponda
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="", language=detected_lang)]
                )
        
        # Verifica se l'agent deve rispondere (cerca @sophyai, wake word, o sessione attiva)
        # La trascrizione è già stata inviata al frontend sopra, qui filtriamo solo per l'LLM
        if text:
            # ==================== TTS INTERRUPT ====================
            # Se l'utente sta parlando e il TTS è attivo, interrompi il TTS SEMPRE
            if is_tts_speaking():
                logger.info(f"✋ Utente parla mentre TTS attivo - STOP immediato")
                asyncio.create_task(interrupt_tts_if_speaking())
                # NON processare questo messaggio, era solo per fermare il TTS
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="", language=detected_lang)]
                )
            
            should_respond, cleaned_text, is_wake = should_agent_respond(text)
            
            if is_wake:
                # Wake word rilevato - pronuncia "Dimmi" e avvia timer
                logger.info(f"🎤 Wake word rilevato, pronuncio 'Dimmi'...")
                asyncio.create_task(handle_wake_word_detected("default"))
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="", language=detected_lang)]
                )
            
            if not should_respond:
                logger.info(f"🔕 Messaggio senza wake attivo, ignoro: '{text[:50]}...'")
                # Restituisci testo vuoto all'LLM per evitare che risponda
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="", language=detected_lang)]
                )
            else:
                # Wake session attiva, passa il testo all'LLM
                logger.info(f"🔔 Wake attivo, invio a LLM: '{cleaned_text[:50]}...'")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text=cleaned_text, language=detected_lang)]
                )
        
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=detected_lang)]
        )
    
    async def _local_transcribe(self, buffer: AudioBuffer, language: str | None) -> str:
        """Fallback trascrizione locale"""
        text, _ = await self._local_transcribe_with_detection(buffer, language)
        return text
    
    async def _local_transcribe_with_detection(self, buffer: AudioBuffer, language: str | None) -> tuple[str, str]:
        """Fallback trascrizione locale con rilevamento lingua"""
        global _detected_language
        import numpy as np
        from faster_whisper import WhisperModel
        
        if not hasattr(self, '_model') or self._model is None:
            logger.info(f"Caricamento modello Whisper locale {self.model_size}...")
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        
        audio_data = np.frombuffer(buffer.data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Se auto_detect, non forzare la lingua
        lang_param = None if self.auto_detect else (language or self.language)
        
        segments, info = self._model.transcribe(audio_data, language=lang_param, beam_size=1)
        text = " ".join([seg.text for seg in segments]).strip()
        
        detected_lang = info.language if hasattr(info, 'language') else (language or self.language)
        
        # Aggiorna lingua rilevata
        if detected_lang:
            self.last_detected_language = detected_lang
            _detected_language = detected_lang
        
        return text, detected_lang
    
    def get_tts_voice_for_language(self, language: str) -> str:
        """Ritorna la voce TTS appropriata per la lingua rilevata"""
        return self.LANGUAGE_TTS_VOICES.get(language, self.LANGUAGE_TTS_VOICES.get("it"))


def create_chatterbox_livekit_wrapper(
    model: str = "multilingual",
    language: str = "it",
    device: str = "auto",
    exaggeration: Optional[float] = None,
    audio_prompt_path: Optional[str] = None,
    auto_language: bool = True
) -> "tts.TTS":
    """
    Crea un wrapper LiveKit-compatibile per Chatterbox TTS.
    
    Questo wrapper usa ChatterboxTTS internamente e lo adatta all'API LiveKit.
    """
    from agent.tts.chatterbox_tts import ChatterboxTTS
    
    # Salva i parametri in variabili locali per la closure
    _model = model
    _language = language
    _device = device
    _exaggeration = exaggeration
    _audio_prompt_path = audio_prompt_path
    _auto_language = auto_language
    
    class ChatterboxLiveKit(tts.TTS):
        """Wrapper LiveKit-compatibile per Chatterbox TTS"""
        
        def __init__(self):
            super().__init__(
                capabilities=tts.TTSCapabilities(streaming=False),
                sample_rate=24000,
                num_channels=1,
            )
            self.chatterbox = ChatterboxTTS(
                model=_model,
                language=_language,
                sample_rate=24000,
                device=_device,
                exaggeration=_exaggeration,
                audio_prompt_path=_audio_prompt_path
            )
            self.language = _language
            self.auto_language = _auto_language
        
        def synthesize(self, text: str, *, conn_options: APIConnectOptions = APIConnectOptions()) -> "ChatterboxTTSStream":
            # Se auto_language è attivo, usa la lingua rilevata
            if self.auto_language:
                current_lang = _detected_language or self.language
                if current_lang != self.language:
                    logger.info(f"🎭 [Chatterbox] Cambio lingua: {self.language} → {current_lang}")
                    self.language = current_lang
                    self.chatterbox.language = current_lang
            
            return ChatterboxTTSStream(self, text, conn_options)
    
    class ChatterboxTTSStream(tts.ChunkedStream):
        """Stream audio da Chatterbox TTS"""
        
        def __init__(self, tts_instance: ChatterboxLiveKit, text: str, conn_options: APIConnectOptions):
            super().__init__(tts=tts_instance, input_text=text, conn_options=conn_options)
            self._tts_instance = tts_instance
            self._text = text
        
        async def _run(self, output_emitter=None) -> None:
            import numpy as np
            try:
                # Sintetizza con Chatterbox
                result = await self._tts_instance.chatterbox.synthesize_async(self._text)
                
                # Converti numpy array in bytes (PCM 16-bit)
                audio_data = result.audio_data
                pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
                
                req_id = str(uuid.uuid4())
                seg_id = str(uuid.uuid4())
                
                frame = rtc.AudioFrame(
                    data=pcm_data,
                    sample_rate=result.sample_rate,
                    num_channels=1,
                    samples_per_channel=len(pcm_data) // 2
                )
                
                # Emetti audio
                if output_emitter is not None:
                    output_emitter.initialize(
                        request_id=req_id,
                        sample_rate=result.sample_rate,
                        num_channels=1,
                        mime_type="audio/pcm",
                        stream=True
                    )
                    output_emitter.start_segment(segment_id=seg_id)
                    output_emitter.push(pcm_data)
                    output_emitter.end_segment()
                    output_emitter.end_input()
                else:
                    audio_event = tts.SynthesizedAudio(
                        frame=frame,
                        request_id=req_id,
                        is_final=True
                    )
                    await self._event_ch.send(audio_event)
                
                logger.info(f"🎭 [Chatterbox] Sintesi completata: {len(pcm_data)} bytes, {result.duration_seconds:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ [Chatterbox] Errore: {e}")
                raise
    
    return ChatterboxLiveKit()


class MultimodalLLM:
    """Wrapper per LLM multimodale che supporta analisi immagini/video"""
    
    def __init__(self, base_llm, llm_provider: str, db_settings: dict):
        """
        Inizializza MultimodalLLM.
        
        Args:
            base_llm: LLM base (OpenAI-compatible)
            llm_provider: "openrouter", "ollama" o "remote"
            db_settings: Settings dal database
        """
        self.base_llm = base_llm
        self.llm_provider = llm_provider
        self.db_settings = db_settings
        self.ollama_host = config.ollama.host if hasattr(config, 'ollama') else "http://localhost:11434"
        
        # Verifica se il modello supporta vision
        self.supports_vision = self._check_vision_support()
        logger.info(f"🔍 MultimodalLLM: provider={llm_provider}, supports_vision={self.supports_vision}")
    
    def _check_vision_support(self) -> bool:
        """Verifica se il modello corrente supporta vision usando info dal database"""
        if self.llm_provider == "remote":
            # Server remoto custom - assumiamo no vision support
            return False
        elif self.llm_provider == "openrouter":
            # Prima controlla se abbiamo l'info salvata dal database (da API OpenRouter)
            db_vision_support = self.db_settings.get("openrouter_supports_vision", "")
            if db_vision_support:
                return db_vision_support.lower() == "true"
            
            # Fallback: controlla nome modello
            model = self.db_settings.get("openrouter_model", "")
            vision_models = [
                "gpt-4-vision", "gpt-4o", "gpt-4-turbo",
                "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3.5",
                "gemma-3", "gemma3", "gemma-2", "gemma2",
                "gemini-pro-vision", "gemini-1.5", "gemini-2",
                "pixtral", "llava", "qwen-vl", "qwen2-vl"
            ]
            return any(vm in model.lower().replace("_", "-") for vm in vision_models)
        elif self.llm_provider == "ollama":
            model = self.db_settings.get("ollama_model", config.ollama.model)
            vision_models = ["llava", "bakllava", "moondream", "gemma3", "gemma-3", "llama3.2-vision", "minicpm-v"]
            return any(vm in model.lower() for vm in vision_models)
        return False
    
    async def analyze_image(self, image_base64: str, prompt: str) -> str:
        """
        Analizza un'immagine usando LLM vision.
        
        Args:
            image_base64: Immagine in base64
            prompt: Prompt per l'analisi
        
        Returns:
            Risposta del LLM
        """
        if not self.supports_vision:
            return "Errore: Il modello LLM configurato non supporta l'analisi di immagini. Usa un modello vision (es. GPT-4 Vision, Claude 3, o llava per Ollama)."
        
        try:
            if self.llm_provider == "openrouter":
                return await self._analyze_with_openrouter(image_base64, prompt)
            elif self.llm_provider == "ollama":
                return await self._analyze_with_ollama(image_base64, prompt)
            else:
                return "Errore: Provider LLM non supportato per vision"
        except Exception as e:
            logger.error(f"Errore analisi immagine: {e}")
            import traceback
            traceback.print_exc()
            return f"Errore durante l'analisi: {str(e)}"
    
    async def _analyze_with_openrouter(self, image_base64: str, prompt: str) -> str:
        """Analizza con OpenRouter usando formato OpenAI vision API"""
        import aiohttp
        
        model = self.db_settings.get("openrouter_model", "openai/gpt-4-vision-preview")
        api_key = self.db_settings.get("openrouter_api_key", "")
        
        if not api_key:
            return "Errore: OpenRouter API key non configurata"
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://sophyai.local",
                    "X-Title": "SophyAi Voice Agent"
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1000
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"OpenRouter error: {resp.status} - {error_text}")
                    return f"Errore API OpenRouter: {resp.status}"
                
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
    
    async def _analyze_with_ollama(self, image_base64: str, prompt: str) -> str:
        """Analizza con Ollama usando modelli vision"""
        try:
            import ollama
            from ollama import AsyncClient
            
            model = self.db_settings.get("ollama_model", config.ollama.model)
            client = AsyncClient(host=self.ollama_host)
            
            # Decodifica base64
            image_bytes = base64.b64decode(image_base64)
            
            # Ollama API per modelli vision
            response = await client.generate(
                model=model,
                prompt=prompt,
                images=[image_bytes],
                stream=False,
                keep_alive=-1
            )
            
            if hasattr(response, 'response'):
                return response.response
            elif isinstance(response, dict):
                return response.get('response', '')
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Errore Ollama vision: {e}")
            # Fallback: prova con API chat se disponibile
            try:
                client = AsyncClient(host=self.ollama_host)
                image_bytes = base64.b64decode(image_base64)
                
                response = await client.chat(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_bytes]
                        }
                    ],
                    keep_alive=-1
                )
                
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    return response.message.content
                elif isinstance(response, dict):
                    return response.get('message', {}).get('content', '')
                return str(response)
            except Exception as e2:
                logger.error(f"Errore fallback Ollama: {e2}")
                return f"Errore analisi Ollama: {str(e)}"


class VideoFrameExtractor:
    """Estrae frame dai video tracks LiveKit per analisi"""
    
    def __init__(self, max_rate: float = None):
        self.video_tracks = {}  # {participant_identity: track}
        self.last_frame_time = {}  # Rate limiting
        self.frame_buffer = {}  # Ultimi frame estratti
        self.max_rate = max_rate or (config.vision.max_frame_rate if hasattr(config, 'vision') else 1.0)
    
    def register_video_track(self, participant_identity: str, track: rtc.VideoTrack):
        """Registra un video track per l'estrazione frame"""
        self.video_tracks[participant_identity] = track
        logger.info(f"📹 Video track registrato per {participant_identity}")
    
    def unregister_video_track(self, participant_identity: str):
        """Rimuove un video track"""
        if participant_identity in self.video_tracks:
            del self.video_tracks[participant_identity]
        if participant_identity in self.frame_buffer:
            del self.frame_buffer[participant_identity]
        if participant_identity in self.last_frame_time:
            del self.last_frame_time[participant_identity]
        logger.info(f"📹 Video track rimosso per {participant_identity}")
    
    async def extract_frame(self, participant_identity: str = None, max_rate: float = None) -> Optional[bytes]:
        """
        Estrae un frame dal video track usando rtc.VideoStream.
        
        Args:
            participant_identity: Identità del partecipante (None = primo disponibile)
            max_rate: Massimo frame al secondo (rate limiting)
        
        Returns:
            Frame come bytes (PNG) o None se non disponibile
        """
        import time
        
        # Seleziona track
        track = None
        identity_key = participant_identity
        if participant_identity:
            track = self.video_tracks.get(participant_identity)
        else:
            # Prendi il primo track disponibile
            if self.video_tracks:
                identity_key = list(self.video_tracks.keys())[0]
                track = list(self.video_tracks.values())[0]
        
        if not track:
            logger.warning("Nessun video track disponibile")
            return None
        
        # Rate limiting
        rate_limit = max_rate or self.max_rate
        now = time.time()
        last_time = self.last_frame_time.get(identity_key, 0)
        if now - last_time < 1.0 / rate_limit:
            # Usa frame bufferizzato se disponibile
            if identity_key in self.frame_buffer:
                logger.info("📹 Usando frame bufferizzato")
                return self.frame_buffer[identity_key]
            return None
        
        try:
            logger.info(f"📹 Estrazione frame da track: {type(track).__name__}")
            
            # Usa VideoStream per estrarre frame (API corretta di livekit-agents)
            video_stream = rtc.VideoStream(track)
            frame_data = None
            
            try:
                # Ottieni il primo frame disponibile con timeout
                async def get_first_frame():
                    async for frame_event in video_stream:
                        return frame_event.frame
                    return None
                
                frame = await asyncio.wait_for(get_first_frame(), timeout=3.0)
                
                if frame:
                    logger.info(f"📹 Frame ricevuto: {frame.width}x{frame.height}")
                    
                    # Converti VideoFrame in ARGB buffer
                    argb_frame = frame.convert(rtc.VideoBufferType.RGBA)
                    
                    # Crea immagine PIL dai dati RGBA
                    img = Image.frombytes(
                        'RGBA',
                        (argb_frame.width, argb_frame.height),
                        argb_frame.data
                    )
                    
                    # Converti in RGB per rimuovere alpha
                    img = img.convert('RGB')
                    
                    # Converti in PNG bytes
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG', optimize=True)
                    frame_data = buffer.getvalue()
                    
                    logger.info(f"📹 Frame convertito: {len(frame_data)} bytes")
                    
            except asyncio.TimeoutError:
                logger.warning("📹 Timeout attesa frame video")
            finally:
                await video_stream.aclose()
            
            if frame_data:
                # Salva nel buffer
                self.frame_buffer[identity_key] = frame_data
                self.last_frame_time[identity_key] = now
                return frame_data
            
            return None
            
        except Exception as e:
            logger.error(f"Errore estrazione frame: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def frame_to_base64(self, frame_bytes: bytes) -> str:
        """Converte frame bytes in base64 string per LLM"""
        return base64.b64encode(frame_bytes).decode('utf-8')


class VisionAgent(Agent):
    """
    Agent con capacità vision. Sottoclasse di Agent che aggiunge
    funzioni tool per l'analisi di immagini/video.
    """
    
    # Variabili di classe per le dipendenze (impostate dopo l'inizializzazione)
    _frame_extractor: Optional[VideoFrameExtractor] = None
    _multimodal_llm: Optional[MultimodalLLM] = None
    _db_settings: dict = {}
    _base_llm = None
    _session: Optional[AgentSession] = None
    
    @classmethod
    def set_vision_dependencies(cls, frame_extractor: VideoFrameExtractor, base_llm, db_settings: dict, session: AgentSession = None):
        """Imposta le dipendenze per le funzioni vision (metodo di classe)"""
        cls._frame_extractor = frame_extractor
        cls._db_settings = db_settings
        cls._base_llm = base_llm
        cls._session = session
        # Crea MultimodalLLM
        llm_provider = db_settings.get("llm_provider", "ollama")
        cls._multimodal_llm = MultimodalLLM(base_llm, llm_provider, db_settings)
        logger.info(f"📹 VisionAgent: dipendenze vision impostate, provider={llm_provider}, session={'presente' if session else 'assente'}")
    
    def _has_video(self) -> bool:
        """Verifica se c'è un video track disponibile"""
        return self._frame_extractor is not None and bool(self._frame_extractor.video_tracks)
    
    async def _analyze_with_prompt(self, prompt: str) -> str:
        """Esegue analisi con il prompt specificato"""
        if not self._has_video():
            return "Non vedo nessun video attivo. Attiva la webcam o condividi lo schermo prima."
        
        if self._multimodal_llm is None:
            return "Il sistema di analisi immagini non è configurato."
        
        try:
            # Estrai frame
            frame_bytes = await self._frame_extractor.extract_frame()
            if not frame_bytes:
                return "Non sono riuscito a catturare un frame dal video."
            
            # Converti in base64
            image_base64 = self._frame_extractor.frame_to_base64(frame_bytes)
            
            # Analizza con LLM multimodale
            result = await self._multimodal_llm.analyze_image(image_base64, prompt)
            
            # Pulisci il risultato da caratteri markdown
            result = result.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
            result = result.replace("\n\n", ". ").replace("\n", ". ")
            
            return result
            
        except Exception as e:
            logger.error(f"Errore analisi vision: {e}")
            return f"Si è verificato un errore durante l'analisi: {str(e)}"
    
    @function_tool(description="Analizza cosa è visibile nel video o nella webcam. Usa questa funzione quando l'utente chiede di vedere, guardare, o descrivere cosa c'è nel video.")
    async def analyze_video(self, context: RunContext) -> str:
        """Analizza il video/immagine dalla webcam o screen sharing.
        
        Args:
            context: Contesto di esecuzione dell'agent.
        """
        logger.info(f"📹 FUNCTION TOOL analyze_video CHIAMATO: has_video={self._has_video()}")
        prompt = """Descrivi in modo naturale e conversazionale cosa vedi in questa immagine.
Sii conciso, usa 2-3 frasi al massimo.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo, come se stessi parlando a voce.
Rispondi come se stessi parlando direttamente a qualcuno."""
        
        result = await self._analyze_with_prompt(prompt)
        logger.info(f"📹 analyze_video risultato: {len(result)} chars")
        
        # Invia direttamente al TTS senza passare dall'LLM
        if self._session and result:
            try:
                set_tts_speaking(True)
                speech_handle = self._session.say(result, allow_interruptions=True)
                await speech_handle
                set_tts_speaking(False)
                logger.info(f"📹 Risultato analyze_video pronunciato direttamente via TTS")
            except Exception as tts_error:
                set_tts_speaking(False)
                logger.error(f"Errore TTS analyze_video: {tts_error}")
        
        # Restituisce stringa vuota per evitare che l'LLM interpreti la risposta
        return ""
    
    @function_tool(description="Leggi e estrai dati da documenti come carte d'identità, patenti, o altri documenti. Usa quando l'utente mostra un documento e chiede di leggerlo.")
    async def analyze_document(self, context: RunContext) -> str:
        """Leggi e estrai dati da documenti.
        
        Args:
            context: Contesto di esecuzione dell'agent.
        """
        prompt = """Analizza questo documento e leggi i dati visibili.
Elenca i dati in modo naturale, come se li stessi leggendo a voce alta.
Per esempio: Il nome è Mario Rossi, nato il 15 marzo 1985, numero documento AB123456.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo, senza JSON o markdown. Sii conversazionale."""
        
        result = await self._analyze_with_prompt(prompt)
        
        # Invia direttamente al TTS senza passare dall'LLM
        if self._session and result:
            try:
                set_tts_speaking(True)
                speech_handle = self._session.say(result, allow_interruptions=True)
                await speech_handle
                set_tts_speaking(False)
                logger.info(f"📹 Risultato analyze_document pronunciato direttamente via TTS")
            except Exception as tts_error:
                set_tts_speaking(False)
                logger.error(f"Errore TTS analyze_document: {tts_error}")
        
        # Restituisce stringa vuota per evitare che l'LLM interpreti la risposta
        return ""
    
    @function_tool(description="Stima l'età approssimativa della persona visibile nel video. Usa quando l'utente chiede quanti anni ha qualcuno.")
    async def estimate_age(self, context: RunContext) -> str:
        """Stima l'età della persona visibile.
        
        Args:
            context: Contesto di esecuzione dell'agent.
        """
        prompt = """Osserva la persona in questa immagine e stima la sua età approssimativa.
Rispondi in modo naturale, per esempio: Direi che ha circa trenta trentacinque anni, basandomi sui lineamenti del viso.
Se non vedi una persona chiaramente, dillo. Sii conversazionale e breve.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo. Scrivi i numeri in lettere."""
        
        result = await self._analyze_with_prompt(prompt)
        
        # Invia direttamente al TTS senza passare dall'LLM
        if self._session and result:
            try:
                set_tts_speaking(True)
                speech_handle = self._session.say(result, allow_interruptions=True)
                await speech_handle
                set_tts_speaking(False)
                logger.info(f"📹 Risultato estimate_age pronunciato direttamente via TTS")
            except Exception as tts_error:
                set_tts_speaking(False)
                logger.error(f"Errore TTS estimate_age: {tts_error}")
        
        # Restituisce stringa vuota per evitare che l'LLM interpreti la risposta
        return ""
    
    @function_tool(description="Descrivi l'ambiente, la stanza o il luogo visibile nel video. Usa quando l'utente chiede dove si trova o cosa c'è intorno.")
    async def describe_environment(self, context: RunContext) -> str:
        """Descrivi l'ambiente/stanza visibile.
        
        Args:
            context: Contesto di esecuzione dell'agent.
        """
        prompt = """Descrivi l'ambiente o la stanza che vedi in questa immagine.
Menziona gli elementi principali come mobili, oggetti, colori, illuminazione.
Sii conciso e conversazionale, come se stessi descrivendo a voce. Due o tre frasi massimo.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo."""
        
        result = await self._analyze_with_prompt(prompt)
        
        # Invia direttamente al TTS senza passare dall'LLM
        if self._session and result:
            try:
                set_tts_speaking(True)
                speech_handle = self._session.say(result, allow_interruptions=True)
                await speech_handle
                set_tts_speaking(False)
                logger.info(f"📹 Risultato describe_environment pronunciato direttamente via TTS")
            except Exception as tts_error:
                set_tts_speaking(False)
                logger.error(f"Errore TTS describe_environment: {tts_error}")
        
        # Restituisce stringa vuota per evitare che l'LLM interpreti la risposta
        return ""

    @function_tool(description="Restituisce la data e l'ora corrente del sistema. Usa questo tool SEMPRE prima di check_room_availability per conoscere la data odierna e calcolare le date corrette quando l'utente usa espressioni relative come 'domani', 'la prossima settimana', 'tra 3 giorni', 'questo weekend'.")
    async def get_current_datetime(
        self,
        context: RunContext,
    ) -> str:
        """Restituisce data/ora corrente del server in formato leggibile e ISO."""
        now = datetime.now()
        utc_now = datetime.now(timezone.utc)
        weekdays_it = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
        months_it = ["", "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
                      "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"]
        weekday = weekdays_it[now.weekday()]
        return (
            f"Data odierna: {weekday} {now.day} {months_it[now.month]} {now.year}, "
            f"ore {now.strftime('%H:%M')}. "
            f"Formato ISO: {now.strftime('%Y-%m-%d')}. "
            f"UTC: {utc_now.strftime('%Y-%m-%dT%H:%M:%SZ')}"
        )

    @function_tool(description="Controlla la disponibilità delle camere in hotel per un periodo. Usa questo tool quando l'utente chiede camere libere, disponibilità o vuole cercare camere tra una data di arrivo e una di partenza.")
    async def check_room_availability(
        self,
        context: RunContext,
        start_date: str,
        end_date: str,
        count: int = 2
    ) -> str:
        """Verifica disponibilità camere tramite webhook n8n.

        Args:
            context: Contesto di esecuzione dell'agent.
            start_date: Data check-in in formato YYYY-MM-DD.
            end_date: Data check-out in formato YYYY-MM-DD.
            count: Numero ospiti/camere richieste.
        """
        try:
            start_date = start_date.strip()
            end_date = end_date.strip()

            if not re.match(r"^\d{4}-\d{2}-\d{2}$", start_date):
                return "La data di arrivo non e valida. Usa il formato anno mese giorno, per esempio duemilaventisei trattino zero cinque trattino diciannove."

            if not re.match(r"^\d{4}-\d{2}-\d{2}$", end_date):
                return "La data di partenza non e valida. Usa il formato anno mese giorno, per esempio duemilaventisei trattino zero cinque trattino ventisei."

            if count <= 0:
                count = 1

            search_room_value = f'Start="{start_date}" End="{end_date}" Count="{count}"'

            logger.info(
                "🏨 check_room_availability: start={} end={} count={}",
                start_date,
                end_date,
                count,
            )

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ROOM_AVAILABILITY_WEBHOOK_URL,
                    params={"SEARCH_ROOM_SB": search_room_value},
                    json={},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as response:
                    status = response.status
                    body_text = await response.text()

            if status >= 400:
                logger.error(
                    "🏨 Webhook disponibilita camere errore: status={} body={}",
                    status,
                    body_text[:500],
                )
                return "Non riesco a verificare la disponibilita camere in questo momento. Riprova tra poco."

            # Prova parsing JSON; in fallback usa testo grezzo
            parsed_payload = None
            try:
                parsed_payload = json.loads(body_text) if body_text else {}
            except Exception:
                parsed_payload = None

            if parsed_payload is None:
                clean_text = body_text.strip()
                if not clean_text:
                    return "Ho inviato la richiesta disponibilita camere, ma il sistema non ha restituito dettagli."
                return f"Risultato disponibilita camere: {clean_text}"

            if isinstance(parsed_payload, dict):
                for key in ("message", "result", "response", "text"):
                    if key in parsed_payload and parsed_payload.get(key):
                        return str(parsed_payload[key])
                return json.dumps(parsed_payload, ensure_ascii=True)

            if isinstance(parsed_payload, list):
                if not parsed_payload:
                    return "Non risultano camere disponibili nel periodo richiesto."
                return json.dumps(parsed_payload, ensure_ascii=True)

            return str(parsed_payload)

        except asyncio.TimeoutError:
            logger.warning("🏨 Timeout webhook disponibilita camere")
            return "La verifica disponibilita camere sta impiegando troppo tempo. Riprova tra poco."
        except Exception as e:
            logger.error(f"Errore check_room_availability: {e}")
            return "Si e verificato un errore durante il controllo disponibilita camere."


async def handle_video_analysis(
    analysis_type: str,
    frame_extractor: VideoFrameExtractor,
    send_callback,
    base_llm,
    db_settings: dict,
    session: AgentSession = None
):
    """Gestisce richiesta analisi video e pronuncia il risultato via TTS"""
    # Genera ID univoco per questa risposta
    video_analysis_id = generate_message_id()
    
    try:
        logger.info(f"📹 Inizio analisi video: {analysis_type} (id={video_analysis_id})")
        
        # Estrai frame
        frame_bytes = await frame_extractor.extract_frame()
        if not frame_bytes:
            result = "Nessun video disponibile per l'analisi. Assicurati che la webcam o lo screen sharing sia attivo."
            await send_callback(json.dumps({
                "type": "video_analysis_result",
                "analysis_type": analysis_type,
                "result": result,
                "id": video_analysis_id
            }), "system", video_analysis_id)
            return
        
        # Converti in base64
        image_base64 = frame_extractor.frame_to_base64(frame_bytes)
        
        # Seleziona prompt in base al tipo
        prompts = {
            "document": """Analizza questo documento e leggi i dati visibili.
Elenca i dati in modo naturale, come se li stessi leggendo a voce alta.
Per esempio: Il nome è Mario Rossi, nato il quindici marzo millenovecentottantacinque, numero documento AB centoventitremilaquattrocentocinquantasei.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo, senza JSON o markdown. Sii conversazionale. Scrivi i numeri in lettere.""",
            
            "age": """Osserva la persona in questa immagine e stima la sua età approssimativa.
Rispondi in modo naturale, per esempio: Direi che ha circa trenta trentacinque anni, basandomi sui lineamenti del viso.
Se non vedi una persona chiaramente, dillo. Sii conversazionale e breve.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo. Scrivi i numeri in lettere.""",
            
            "environment": """Descrivi l'ambiente o la stanza che vedi in questa immagine.
Menziona gli elementi principali come mobili, oggetti, colori, illuminazione.
Sii conciso e conversazionale, come se stessi descrivendo a voce. Due o tre frasi massimo.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo.""",
            
            "generic": """Descrivi in modo naturale e conversazionale cosa vedi in questa immagine.
Sii conciso, usa due o tre frasi al massimo.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo, come se stessi parlando a voce."""
        }
        
        prompt = prompts.get(analysis_type, prompts["generic"])
        
        # Crea MultimodalLLM
        llm_provider = db_settings.get("llm_provider", "ollama")
        multimodal_llm = MultimodalLLM(base_llm, llm_provider, db_settings)
        
        # Analizza
        result = await multimodal_llm.analyze_image(image_base64, prompt)
        
        logger.info(f"📹 Analisi completata: {result[:100]}...")
        
        # Pulisci il risultato da caratteri markdown per TTS
        tts_result = result.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
        tts_result = tts_result.replace("\n\n", ". ").replace("\n", ". ")
        
        # Invia risultato al frontend con ID
        await send_callback(json.dumps({
            "type": "video_analysis_result",
            "analysis_type": analysis_type,
            "result": result,
            "id": video_analysis_id
        }), "system", video_analysis_id)
        
        # Pronuncia il risultato via TTS
        if session:
            try:
                set_tts_speaking(True)
                speech_handle = session.say(tts_result, allow_interruptions=True)
                # Attendi che il TTS finisca
                await speech_handle
                set_tts_speaking(False)
                logger.info(f"📹 Risultato pronunciato via TTS (id={video_analysis_id})")
            except Exception as tts_error:
                set_tts_speaking(False)
                logger.error(f"Errore TTS analisi video: {tts_error}")
        
    except Exception as e:
        logger.error(f"Errore analisi video: {e}")
        import traceback
        traceback.print_exc()
        
        error_msg = f"Mi dispiace, si è verificato un errore durante l'analisi."
        error_id = generate_message_id()
        await send_callback(json.dumps({
            "type": "video_analysis_result",
            "analysis_type": analysis_type,
            "result": f"Errore durante l'analisi: {str(e)}",
            "id": error_id
        }), "system", error_id)
        
        # Pronuncia errore via TTS
        if session:
            try:
                set_tts_speaking(True)
                speech_handle = session.say(error_msg, allow_interruptions=True)
                await speech_handle
                set_tts_speaking(False)
            except:
                set_tts_speaking(False)


async def handle_image_analysis(
    image_base64: str,
    analysis_type: str,
    custom_prompt: str,
    send_callback,
    base_llm,
    db_settings: dict,
    session: AgentSession = None
):
    """Gestisce analisi di immagine caricata dall'utente"""
    # Genera ID univoco per questa risposta
    image_analysis_id = generate_message_id()
    
    try:
        logger.info(f"🖼️ Inizio analisi immagine caricata: {analysis_type} (id={image_analysis_id})")
        
        if not image_base64:
            result = "Nessuna immagine ricevuta per l'analisi."
            await send_callback(json.dumps({
                "type": "image_analysis_result",
                "analysis_type": analysis_type,
                "result": result,
                "id": image_analysis_id
            }), "system", image_analysis_id)
            return
        
        # Se c'è un prompt personalizzato, usalo (con aggiunta di istruzioni sui caratteri speciali)
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt.strip() + "\nIMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione. Scrivi solo testo semplice e discorsivo."
        else:
            # Seleziona prompt in base al tipo (stessi prompt di handle_video_analysis)
            prompts = {
                "document": """Analizza questo documento e leggi i dati visibili.
Elenca i dati in modo naturale, come se li stessi leggendo a voce alta.
Per esempio: Il nome è Mario Rossi, nato il quindici marzo millenovecentottantacinque, numero documento AB centoventitremilaquattrocentocinquantasei.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo, senza JSON o markdown. Sii conversazionale. Scrivi i numeri in lettere.""",
                
                "age": """Osserva la persona in questa immagine e stima la sua età approssimativa.
Rispondi in modo naturale, per esempio: Direi che ha circa trenta trentacinque anni, basandomi sui lineamenti del viso.
Se non vedi una persona chiaramente, dillo. Sii conversazionale e breve.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo. Scrivi i numeri in lettere.""",
                
                "environment": """Descrivi l'ambiente o la stanza che vedi in questa immagine.
Menziona gli elementi principali come mobili, oggetti, colori, illuminazione.
Sii conciso e conversazionale, come se stessi descrivendo a voce. Due o tre frasi massimo.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo.""",
                
                "generic": """Descrivi in modo naturale e conversazionale cosa vedi in questa immagine.
Sii conciso, usa due o tre frasi al massimo.
IMPORTANTE: Non usare MAI caratteri speciali come asterischi, hashtag, trattini, elenchi puntati, parentesi, virgolette, simboli matematici o qualsiasi formattazione.
Scrivi solo testo semplice e discorsivo, come se stessi parlando a voce."""
            }
            prompt = prompts.get(analysis_type, prompts["generic"])
        
        # Crea MultimodalLLM
        llm_provider = db_settings.get("llm_provider", "ollama")
        multimodal_llm = MultimodalLLM(base_llm, llm_provider, db_settings)
        
        # Analizza l'immagine
        result = await multimodal_llm.analyze_image(image_base64, prompt)
        
        logger.info(f"🖼️ Analisi immagine completata: {result[:100]}...")
        
        # Pulisci il risultato da caratteri markdown per TTS
        tts_result = result.replace("**", "").replace("*", "").replace("#", "").replace("`", "")
        tts_result = tts_result.replace("\n\n", ". ").replace("\n", ". ")
        
        # Invia risultato al frontend con ID
        await send_callback(json.dumps({
            "type": "image_analysis_result",
            "analysis_type": analysis_type,
            "result": result,
            "id": image_analysis_id
        }), "system", image_analysis_id)
        
        # Pronuncia il risultato via TTS
        if session:
            try:
                set_tts_speaking(True)
                speech_handle = session.say(tts_result, allow_interruptions=True)
                await speech_handle
                set_tts_speaking(False)
                logger.info(f"🖼️ Risultato analisi immagine pronunciato via TTS (id={image_analysis_id})")
            except Exception as tts_error:
                set_tts_speaking(False)
                logger.error(f"Errore TTS analisi immagine: {tts_error}")
        
    except Exception as e:
        logger.error(f"Errore analisi immagine: {e}")
        import traceback
        traceback.print_exc()
        
        error_msg = f"Mi dispiace, si è verificato un errore durante l'analisi dell'immagine."
        error_id = generate_message_id()
        await send_callback(json.dumps({
            "type": "image_analysis_result",
            "analysis_type": analysis_type,
            "result": f"Errore durante l'analisi: {str(e)}",
            "id": error_id
        }), "system", error_id)
        
        # Pronuncia errore via TTS
        if session:
            try:
                set_tts_speaking(True)
                speech_handle = session.say(error_msg, allow_interruptions=True)
                await speech_handle
                set_tts_speaking(False)
            except:
                set_tts_speaking(False)


async def load_settings_from_server() -> dict:
    """Carica impostazioni dal web server (database)"""
    import ssl
    
    settings = {
        "llm_provider": "ollama",
        "ollama_model": config.ollama.model,
        "openrouter_model": "",
        "openrouter_api_key": "",
        "system_prompt": "",
        "context_injection": "",
        "whisper_model": config.whisper.model,
        "whisper_language": config.whisper.language,
        # Voice Activation defaults (molto sensibili per chiamate SIP)
        "wake_timeout_seconds": "30",
        "vad_energy_threshold": "70",
        "speech_energy_threshold": "25",
        "silence_threshold": "60",
        "tts_cooldown_seconds": "1.5",
        # TTS defaults
        "tts_engine": config.tts.default_engine,
        "tts_language": "it",
        # ElevenLabs defaults
        "elevenlabs_api_key": "",
        "elevenlabs_voice": "",
        "elevenlabs_model": "eleven_multilingual_v2",
        "elevenlabs_stability": "50",
        "elevenlabs_similarity": "75",
        "sip_context_injection": "",
    }
    
    server_candidates = []
    web_server_url = os.getenv("WEB_SERVER_URL", "").strip()
    if web_server_url:
        server_candidates.append(web_server_url.rstrip("/"))
    server_candidates.extend([
        "http://voice-agent-web:8080",
        "http://host.docker.internal:8080",
        "https://host.docker.internal:8443",
        "http://127.0.0.1:8080",
    ])
    # Dedup mantenendo ordine
    server_candidates = list(dict.fromkeys(server_candidates))

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ssl_context)

    loaded = False
    last_error = ""
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            for base_url in server_candidates:
                try:
                    async with session.get(
                        f"{base_url}/api/settings",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status != 200:
                            raise RuntimeError(f"/api/settings status={resp.status}")
                        data = await resp.json()
                        settings.update(data)

                    async with session.get(
                        f"{base_url}/api/prompt",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            settings["system_prompt"] = data.get("prompt", "")

                    async with session.get(
                        f"{base_url}/api/context",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            settings["context_injection"] = data.get("context", "")

                    loaded = True
                    logger.info(f"📥 Settings caricati da database via {base_url}")
                    break
                except Exception as endpoint_error:
                    last_error = str(endpoint_error)
                    continue
    except Exception as e:
        last_error = str(e)

    if not loaded:
        logger.warning(f"⚠️ Impossibile caricare settings da DB: {last_error}")
    
    return settings


async def request_handler(request: JobRequest) -> AutoSubscribe:
    """
    Gestisce le richieste di job.
    Accetta automaticamente i job nelle room SIP per permettere all'agent
    di rispondere alle chiamate telefoniche.
    """
    
    room_name = request.room.name
    
    # Accetta automaticamente job in room SIP
    if room_name.startswith("sip-") or room_name == "sip-call":
        logger.info(f"📞 Accetto automaticamente job SIP per room: {room_name}")
        await request.accept()
        return AutoSubscribe.SUBSCRIBE_ALL
    
    # Per altre room, accetta normalmente (dispatch esplicito)
    logger.info(f"✅ Accetto job per room: {room_name}")
    await request.accept()
    return AutoSubscribe.SUBSCRIBE_ALL


async def entrypoint(ctx: JobContext):
    """Entry point per l'agent LiveKit"""
    await ctx.connect()
    
    # ==================== CHECK DUPLICATI ====================
    # Verifica se c'è già un altro agent (bot) nella room
    # Se sì, questo agent si disconnette per evitare duplicati
    existing_agents = [p for p in ctx.room.remote_participants.values() 
                       if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT]
    if existing_agents:
        logger.warning(f"⚠️ Room {ctx.room.name} ha già {len(existing_agents)} agent(s), questo agent si disconnette")
        ctx.shutdown(reason="duplicate-agent")
        return
    
    logger.info(f"Agent connesso alla room: {ctx.room.name}")
    
    # ==================== RILEVAMENTO CHIAMATA SIP ====================
    global _is_sip_call, _current_call_log_id
    
    room_name = ctx.room.name
    _is_sip_call = room_name.startswith("sip-") or room_name == "sip-call"
    _current_call_log_id = None
    
    if _is_sip_call:
        logger.info(f"📞 CHIAMATA SIP RILEVATA in room: {room_name}")
        logger.info(f"🔄 Contesto resettato per nuova chiamata SIP")
        
        # Le chiamate SIP partono con contesto pulito
        # Il call_log viene creato dal webhook, qui lo recuperiamo per salvare i messaggi
        try:
            import aiohttp
            import os
            server_url = os.getenv("WEB_SERVER_URL", "http://voice-agent-web:8080")
            async with aiohttp.ClientSession() as session:
                # Cerca il call_log attivo per questa room
                async with session.get(f"{server_url}/api/calls?status=active&limit=10") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for call in data.get('calls', []):
                            if call.get('room_name') == room_name:
                                _current_call_log_id = call.get('call_id')
                                logger.info(f"📝 Trovato call_log attivo: {_current_call_log_id}")
                                break
        except Exception as e:
            logger.warning(f"⚠️ Impossibile recuperare call_log: {e}")
    else:
        _is_sip_call = False
        _current_call_log_id = None
    
    # Carica impostazioni dal database
    db_settings = await load_settings_from_server()
    logger.info(f"📥 LLM Provider: {db_settings.get('llm_provider', 'ollama')}")

    # Risolvi contesto: per SIP usa il numero chiamato, per web usa il default
    try:
        import aiohttp
        import os
        web_base = os.getenv("WEB_SERVER_URL", "http://voice-agent-web:8080").rstrip("/")

        if _is_sip_call:
            resolve_url = f"{web_base}/api/sip/context/resolve?room_name={urllib.parse.quote(room_name, safe='')}"
        else:
            resolve_url = f"{web_base}/api/sip/context/resolve?called_number=__default__"

        # #region agent log
        import json as _dbg_json, time as _dbg_time
        _dbg_log_path = "/app/.cursor/debug-ffaca3.log"
        with open(_dbg_log_path, "a") as _f: _f.write(_dbg_json.dumps({"sessionId":"ffaca3","hypothesisId":"H1","location":"main.py:ctx-resolve","message":"context_resolve_start","data":{"is_sip":_is_sip_call,"room_name":room_name,"resolve_url":resolve_url},"timestamp":int(_dbg_time.time()*1000)})+"\n")
        # #endregion

        sip_context = ""
        async with aiohttp.ClientSession() as _ctx_session:
            async with _ctx_session.get(resolve_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                # #region agent log
                _resp_text = await resp.text()
                with open(_dbg_log_path, "a") as _f: _f.write(_dbg_json.dumps({"sessionId":"ffaca3","hypothesisId":"H1","location":"main.py:ctx-resp","message":"primary_resolve_response","data":{"status":resp.status,"body":_resp_text[:500]},"timestamp":int(_dbg_time.time()*1000)})+"\n")
                # #endregion
                if resp.status == 200:
                    payload = await resp.json()
                    sip_context = (payload.get("context") or "").strip()
                    if sip_context:
                        db_settings["sip_context_injection"] = sip_context
                        logger.info(
                            f"🏨 Context risolto per {'SIP ' + str(payload.get('matched_number')) if _is_sip_call else 'default'}: "
                            f"{len(sip_context)} caratteri"
                        )

            # #region agent log
            with open(_dbg_log_path, "a") as _f: _f.write(_dbg_json.dumps({"sessionId":"ffaca3","hypothesisId":"H2","location":"main.py:ctx-fallback-check","message":"after_primary_resolve","data":{"sip_context_len":len(sip_context),"is_sip":_is_sip_call,"will_fallback":not sip_context and _is_sip_call},"timestamp":int(_dbg_time.time()*1000)})+"\n")
            # #endregion

            if not sip_context and _is_sip_call:
                fallback_url = f"{web_base}/api/sip/context/resolve?called_number=__default__"
                async with _ctx_session.get(fallback_url, timeout=aiohttp.ClientTimeout(total=5)) as resp2:
                    if resp2.status == 200:
                        payload2 = await resp2.json()
                        sip_context = (payload2.get("context") or "").strip()
                        if sip_context:
                            db_settings["sip_context_injection"] = sip_context
                            logger.info(f"🏨 Context SIP fallback __default__: {len(sip_context)} caratteri")
                # #region agent log
                with open(_dbg_log_path, "a") as _f: _f.write(_dbg_json.dumps({"sessionId":"ffaca3","hypothesisId":"H2","location":"main.py:ctx-fallback-result","message":"fallback_resolve_result","data":{"sip_context_len":len(sip_context)},"timestamp":int(_dbg_time.time()*1000)})+"\n")
                # #endregion
    except Exception as e:
        # #region agent log
        with open(_dbg_log_path, "a") as _f: _f.write(_dbg_json.dumps({"sessionId":"ffaca3","hypothesisId":"H1","location":"main.py:ctx-error","message":"context_resolve_error","data":{"error":str(e)},"timestamp":int(_dbg_time.time()*1000)})+"\n")
        # #endregion
        logger.warning(f"⚠️ Impossibile risolvere context per room {room_name}: {e}")
    
    
    # Applica Voice Settings dalle impostazioni caricate
    global WAKE_TIMEOUT_SECONDS, VAD_ENERGY_THRESHOLD, SPEECH_ENERGY_THRESHOLD, SILENCE_THRESHOLD, TTS_COOLDOWN_SECONDS
    try:
        WAKE_TIMEOUT_SECONDS = int(db_settings.get('wake_timeout_seconds', '30'))
        VAD_ENERGY_THRESHOLD = int(db_settings.get('vad_energy_threshold', '120'))
        SPEECH_ENERGY_THRESHOLD = int(db_settings.get('speech_energy_threshold', '25'))
        SILENCE_THRESHOLD = int(db_settings.get('silence_threshold', '60'))
        TTS_COOLDOWN_SECONDS = float(db_settings.get('tts_cooldown_seconds', '1.5'))
        logger.info(f"🎙️ Voice Settings: wake_timeout={WAKE_TIMEOUT_SECONDS}s, vad={VAD_ENERGY_THRESHOLD}, speech={SPEECH_ENERGY_THRESHOLD}, silence={SILENCE_THRESHOLD}, cooldown={TTS_COOLDOWN_SECONDS}s")
        
    except Exception as e:
        logger.warning(f"⚠️ Errore parsing voice settings: {e}, uso default")
    
    # Inizializza LLM in base al provider configurato
    llm_provider = db_settings.get("llm_provider", "ollama")
    
    llm_chat_extra_kwargs = {}

    if llm_provider == "remote" and db_settings.get("remote_server_url"):
        # Usa Server Remoto Custom con adapter LiveKit-compatible
        remote_url = db_settings.get("remote_server_url", "")
        remote_token = db_settings.get("remote_server_token", "")
        remote_collection = db_settings.get("remote_server_collection", "")
        
        base_llm = RemoteLLMAdapter(
            server_url=remote_url,
            token=remote_token,
            collection=remote_collection
        )
        logger.info(f"🖥️ LLM: Server Remoto ({remote_url}), collection={remote_collection}")
        
    elif llm_provider == "openrouter" and db_settings.get("openrouter_api_key"):
        # Usa OpenRouter
        openrouter_model = db_settings.get("openrouter_model", "openai/gpt-3.5-turbo")
        openrouter_key = db_settings.get("openrouter_api_key", "")
        
        base_llm = openai.LLM(
            model=openrouter_model,
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key,
        )
        logger.info(f"🌐 LLM: OpenRouter ({openrouter_model})")
    else:
        # Usa Ollama (default)
        ollama_base_url = config.ollama.host + "/v1"
        ollama_model = db_settings.get("ollama_model", config.ollama.model)
        ollama_extra_body = None
        if str(ollama_model).lower().startswith("qwen3"):
            # Qwen3 in modalita' "thinking" aumenta molto la latenza e causa timeout.
            ollama_extra_body = {"think": False}
            llm_chat_extra_kwargs = {"extra_body": ollama_extra_body}
        
        ollama_llm_kwargs = {
            "model": ollama_model,
            "base_url": ollama_base_url,
            "api_key": "ollama",  # Ollama non richiede API key
        }
        if ollama_extra_body:
            ollama_llm_kwargs["extra_body"] = ollama_extra_body
        base_llm = openai.LLM(**ollama_llm_kwargs)
        logger.info(f"🦙 LLM: Ollama ({ollama_model})")
    
    # Wrapper per timing LLM - usa callback su eventi stream
    class TimedLLMStream:
        """Wrapper per stream LLM che traccia i timing"""
        def __init__(self, wrapped_stream, t_start):
            self._wrapped = wrapped_stream
            self._t_start = t_start
            self._first_chunk = True
            self._ttfb = 0
            
        def __aiter__(self):
            return self
            
        async def __anext__(self):
            try:
                chunk = await self._wrapped.__anext__()
                if self._first_chunk:
                    self._ttfb = (time.time() - self._t_start) * 1000
                    global _last_llm_ttft_ms
                    _last_llm_ttft_ms = self._ttfb
                    logger.info(f"🤖 [LLM] Time to first token: {self._ttfb:.0f}ms")
                    self._first_chunk = False
                return chunk
            except StopAsyncIteration:
                # Stream finito - invia timing
                total_time = (time.time() - self._t_start) * 1000
                logger.info(f"🤖 [LLM] Tempo totale: {total_time:.0f}ms")
                asyncio.create_task(send_timing_to_server("llm", {
                    "time_ms": int(total_time),
                    "ttft_ms": int(self._ttfb)
                }))
                raise
        
        # Supporto async context manager (async with)
        async def __aenter__(self):
            if hasattr(self._wrapped, '__aenter__'):
                await self._wrapped.__aenter__()
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # Invia timing alla fine del context
            total_time = (time.time() - self._t_start) * 1000
            if self._first_chunk:
                # Non c'erano chunk, ma registriamo comunque il tempo
                logger.info(f"🤖 [LLM] Tempo totale (no chunks): {total_time:.0f}ms")
            asyncio.create_task(send_timing_to_server("llm", {
                "time_ms": int(total_time),
                "ttft_ms": int(self._ttfb)
            }))
            if hasattr(self._wrapped, '__aexit__'):
                return await self._wrapped.__aexit__(exc_type, exc_val, exc_tb)
            return False
        
        # Proxy tutti gli altri attributi/metodi allo stream originale
        def __getattr__(self, name):
            return getattr(self._wrapped, name)
    
    class TimedLLM(llm.LLM):
        def __init__(self, wrapped_llm):
            super().__init__()
            self._wrapped = wrapped_llm
        
        def chat(self, **kwargs) -> llm.LLMStream:
            t_start = time.time()
            logger.info(f"🤖 [LLM] Inizio richiesta...")
            stream = self._wrapped.chat(**kwargs)
            return TimedLLMStream(stream, t_start)
    
    my_llm = TimedLLM(base_llm)
    logger.info(f"LLM configurato e pronto")
    
    # Leggi configurazione TTS dal file condiviso (se esiste)
    tts_config_file = "/app/config/tts_config.json"
    tts_from_file = None
    tts_language = "it"
    
    try:
        import json
        if os.path.exists(tts_config_file):
            with open(tts_config_file, "r") as f:
                tts_from_file = json.load(f)
            logger.info(f"📁 Config TTS caricata da file: {tts_from_file}")
    except Exception as e:
        logger.warning(f"⚠️ Errore lettura config TTS: {e}")
    
    # Seleziona TTS: priorità al file, poi database, poi variabile d'ambiente
    if tts_from_file:
        tts_engine = tts_from_file.get("engine", "edge").lower()
        tts_language = tts_from_file.get("language", "it")
        logger.info(f"📁 TTS config da file: engine={tts_engine}, language={tts_language}")
    elif db_settings.get("tts_engine"):
        tts_engine = db_settings.get("tts_engine", "edge").lower()
        tts_language = db_settings.get("tts_language", "it")
        logger.info(f"🗄️ TTS config da database: engine={tts_engine}, language={tts_language}")
    else:
        tts_engine = config.tts.default_engine.lower()
        tts_language = config.tts.vibevoice_language
        logger.info(f"⚙️ TTS config da env: engine={tts_engine}, language={tts_language}")
    
    
    logger.info(f"🔊 ======================================")
    logger.info(f"🔊 CONFIGURAZIONE TTS")
    logger.info(f"🔊 Engine selezionato: {tts_engine}")
    logger.info(f"🔊 Lingua: {tts_language}")
    logger.info(f"🔊 Fonte config: {'file' if tts_from_file else ('db' if db_settings.get('tts_engine') else 'env')}")
    logger.info(f"🔊 ======================================")
    
    if tts_engine == "vibevoice":
        try:
            # Usa il wrapper VibeVoice LiveKit-compatibile
            my_tts = VibeVoiceLiveKit(
                model=tts_from_file.get("model", "realtime") if tts_from_file else "realtime",
                language=tts_language,
                speaker=tts_from_file.get("speaker", "speaker_1") if tts_from_file else "speaker_1",
                speed=tts_from_file.get("speed", 1.0) if tts_from_file else 1.0,
                auto_language=True
            )
            logger.info(f"🎤 TTS attivo: VibeVoiceLiveKit (wrapper nativo)")
            logger.info(f"🎤 Lingua: {tts_language}, Speaker: {my_tts.speaker}")
        except Exception as e:
            logger.error(f"❌ Errore configurazione VibeVoice: {e}")
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
    elif tts_engine == "kokoro":
        try:
            # Kokoro usa il server TTS esterno
            kokoro_speed = tts_from_file.get("speed", 1.0) if tts_from_file else 1.0
            my_tts = ExternalTTSLiveKit(
                engine="kokoro",
                language=tts_language,
                speed=kokoro_speed,
                auto_language=True
            )
            logger.info(f"🔊 TTS attivo: Kokoro (via server esterno)")
            logger.info(f"🎤 Lingua: {tts_language}")
        except Exception as e:
            logger.error(f"❌ Errore configurazione Kokoro: {e}")
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
    elif tts_engine == "piper":
        try:
            # Piper usa il server TTS esterno
            piper_model = tts_from_file.get("model", "it_IT-riccardo-x_low") if tts_from_file else "it_IT-riccardo-x_low"
            my_tts = ExternalTTSLiveKit(
                engine="piper",
                model=piper_model,
                language=tts_language,
                auto_language=True
            )
            logger.info(f"🔊 TTS attivo: Piper (via server esterno)")
            logger.info(f"🎤 Lingua: {tts_language}, Model: {piper_model}")
        except Exception as e:
            logger.error(f"❌ Errore configurazione Piper: {e}")
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
    elif tts_engine == "qwen":
        try:
            qwen_speed = tts_from_file.get("speed", 1.0) if tts_from_file else 1.0
            qwen_speaker = tts_from_file.get("speaker", "Ryan") if tts_from_file else "Ryan"
            my_tts = ExternalTTSLiveKit(
                engine="qwen",
                language=tts_language,
                speaker=qwen_speaker,
                speed=qwen_speed,
                auto_language=True
            )
            logger.info(f"🔊 TTS attivo: Qwen (via server esterno)")
            logger.info(f"🎤 Lingua: {tts_language}, Speaker: {qwen_speaker}")
        except Exception as e:
            logger.error(f"❌ Errore configurazione Qwen: {e}")
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
    elif tts_engine == "elevenlabs":
        try:
            # ElevenLabs usa il server TTS esterno come proxy
            el_voice = tts_from_file.get("voice", "") if tts_from_file else ""
            el_model = tts_from_file.get("model", "eleven_multilingual_v2") if tts_from_file else "eleven_multilingual_v2"
            
            # Carica API key e voice dalle impostazioni DB
            el_api_key = db_settings.get("elevenlabs_api_key", "")
            el_voice_id = db_settings.get("elevenlabs_voice", el_voice)
            el_stability = float(db_settings.get("elevenlabs_stability", "50")) / 100
            el_similarity = float(db_settings.get("elevenlabs_similarity", "75")) / 100
            
            if el_api_key and el_voice_id:
                from agent.tts.elevenlabs_livekit import ElevenLabsLiveKit
                my_tts = ElevenLabsLiveKit(
                    api_key=el_api_key,
                    voice_id=el_voice_id,
                    model=el_model,
                    stability=el_stability,
                    similarity_boost=el_similarity,
                    language=tts_language,
                    auto_language=True
                )
                logger.info(f"🌟 TTS attivo: ElevenLabs (model={el_model}, voice={el_voice_id})")
            else:
                logger.warning(f"⚠️ ElevenLabs: API key o voice mancante, fallback a EdgeTTS")
                my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
        except ImportError:
            logger.warning(f"⚠️ ElevenLabs wrapper non disponibile, fallback a EdgeTTS")
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
        except Exception as e:
            logger.error(f"❌ Errore configurazione ElevenLabs: {e}")
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
    elif tts_engine == "chatterbox":
        try:
            from agent.tts.chatterbox_tts import ChatterboxTTS
            # Usa parametri da file o default da config
            chatterbox_model = tts_from_file.get("model", config.tts.chatterbox_model) if tts_from_file else config.tts.chatterbox_model
            chatterbox_language = tts_from_file.get("language", config.tts.chatterbox_language) if tts_from_file else config.tts.chatterbox_language
            chatterbox_device = tts_from_file.get("device", config.tts.chatterbox_device) if tts_from_file else config.tts.chatterbox_device
            chatterbox_exaggeration = tts_from_file.get("exaggeration") if tts_from_file else config.tts.chatterbox_exaggeration
            chatterbox_audio_prompt_path = tts_from_file.get("audio_prompt_path") if tts_from_file else config.tts.chatterbox_audio_prompt_path
            
            # Crea wrapper LiveKit-compatibile
            my_tts = create_chatterbox_livekit_wrapper(
                model=chatterbox_model,
                language=chatterbox_language,
                device=chatterbox_device,
                exaggeration=chatterbox_exaggeration,
                audio_prompt_path=chatterbox_audio_prompt_path,
                auto_language=True
            )
            logger.info(f"🎭 TTS attivo: Chatterbox (model={chatterbox_model}, language={chatterbox_language})")
        except Exception as e:
            logger.error(f"❌ Errore configurazione Chatterbox: {e}")
            import traceback
            traceback.print_exc()
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
    else:
        # Default: Edge TTS
        my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
        logger.info(f"🔊 TTS attivo: EdgeTTS")
        logger.info(f"🔊 Voce: {config.tts.edge_voice}")
        logger.info(f"🔊 Auto-language: attivo (cambio voce automatico)")
    
    logger.info(f"🔊 ======================================")
    
    
    # ⏱️ Imposta info componenti per conversation tracking
    global _component_info
    _component_info["llm"] = f"{llm_provider}"
    if llm_provider == "ollama":
        _component_info["llm"] = f"ollama/{db_settings.get('ollama_model', config.ollama.model)}"
    elif llm_provider == "openrouter":
        _component_info["llm"] = f"openrouter/{db_settings.get('openrouter_model', 'gpt-3.5-turbo')}"
    elif llm_provider == "remote":
        _component_info["llm"] = "remote-server"
    try:
        _component_info["tts"] = tts_engine
    except NameError:
        _component_info["tts"] = "edge"
    _component_info["stt"] = "whisper"
    
    # Configura Whisper STT usando settings dal database
    whisper_model = db_settings.get("whisper_model", config.whisper.model)
    whisper_language = db_settings.get("whisper_language", config.whisper.language)
    whisper_auto_detect = db_settings.get("whisper_auto_detect", "false") == "true"
    
    my_stt = WhisperSTT(
        model_size=whisper_model,
        language=whisper_language,
        auto_detect=whisper_auto_detect
    )
    logger.info(f"🎤 Whisper: model={whisper_model}, language={whisper_language}, auto_detect={whisper_auto_detect}")
    
    # VAD
    vad = silero.VAD.load()
    
    logger.info("Componenti caricati, creo Agent...")
    
    # Costruisci il prompt usando quello dal database (se disponibile)
    # I trigger per l'attivazione vengono dalla configurazione branding
    triggers_str = ", ".join([f'"{t}"' for t in ASSISTANT_TRIGGERS[:3]])  # Primi 3 trigger
    default_prompt = f"""Sei {ASSISTANT_NAME}, assistente vocale ultra-veloce. PRIORITÀ ASSOLUTA: VELOCITÀ E SINTESI.

ATTIVAZIONE:
IMPORTANTE: Rispondi SOLO quando vieni menzionato esplicitamente con {triggers_str} o varianti simili.
Se il messaggio NON contiene il tuo nome o una menzione diretta a te, NON rispondere affatto.
Quando sei menzionato, rispondi in modo utile e conciso.

REGOLE FONDAMENTALI:
1. RISPOSTE ULTRA-BREVI: massimo 1-2 frasi, mai più di 30 parole
2. VAI DRITTO AL PUNTO: niente preamboli, saluti inutili o ripetizioni
3. LINGUA: rispondi nella stessa lingua dell'utente

CAPACITÀ VISION:
Hai accesso a webcam e screen sharing. Quando l'utente ti chiede di:
- Vedere, guardare, o descrivere cosa c'è nel video: usa analyze_video
- Leggere documenti, carte d'identità, patenti: usa analyze_document
- Stimare l'età di qualcuno: usa estimate_age
- Descrivere l'ambiente o la stanza: usa describe_environment
Usa sempre le funzioni appropriate quando l'utente fa richieste visive.

CAPACITÀ HOTEL:
Quando l'utente chiede disponibilità camere, camere libere o ricerca camere tra date specifiche:
1. PRIMA chiama get_current_datetime per conoscere la data odierna
2. POI usa la data odierna per calcolare le date corrette (es. "domani" = data odierna + 1 giorno)
3. INFINE chiama check_room_availability passando:
   - start_date in formato YYYY-MM-DD
   - end_date in formato YYYY-MM-DD
   - count come numero richiesto
IMPORTANTE: NON inventare o indovinare mai la data corrente. Usa SEMPRE get_current_datetime.

STILE:
- Rispondi come un amico esperto: diretto, chiaro, utile
- Se non sai qualcosa, dillo in 5 parole
- Preferisci risposte secche e precise

FORMATO TTS:
- NO simboli: * # @ euro dollaro percentuale ampersand / | minore maggiore parentesi graffe quadre tilde
- NO emoji
- Numeri in parole (ventitre, non 23)
- NO elenchi puntati, scrivi discorsivo"""

    # Usa prompt dal database se disponibile
    system_prompt = db_settings.get("system_prompt", "").strip()
    if not system_prompt:
        system_prompt = default_prompt
    
    # Aggiungi context injection se presente
    context_injection = db_settings.get("context_injection", "").strip()
    if context_injection:
        system_prompt = f"{system_prompt}\n\n--- INFORMAZIONI AGGIUNTIVE ---\n{context_injection}"
        logger.info(f"📝 Context injection aggiunto: {len(context_injection)} caratteri")

    sip_context_injection = db_settings.get("sip_context_injection", "").strip()
    # #region agent log
    try:
        import json as _dbg_json2, time as _dbg_time2
        _dbg_log_path2 = "/app/.cursor/debug-ffaca3.log"
        with open(_dbg_log_path2, "a") as _f: _f.write(_dbg_json2.dumps({"sessionId":"ffaca3","hypothesisId":"H3","location":"main.py:prompt-inject","message":"sip_context_at_prompt_build","data":{"sip_ctx_len":len(sip_context_injection),"has_ctx":bool(sip_context_injection),"first100":sip_context_injection[:100] if sip_context_injection else ""},"timestamp":int(_dbg_time2.time()*1000)})+"\n")
    except: pass
    # #endregion
    if sip_context_injection:
        system_prompt = f"{system_prompt}\n\n--- CONTESTO SIP PER NUMERO CHIAMATO ---\n{sip_context_injection}"
        logger.info(f"🏨 Context SIP aggiunto: {len(sip_context_injection)} caratteri")
    
    # #region agent log
    try:
        with open(_dbg_log_path2, "a") as _f: _f.write(_dbg_json2.dumps({"sessionId":"ffaca3","hypothesisId":"H3","location":"main.py:prompt-final","message":"system_prompt_final_size","data":{"prompt_len":len(system_prompt),"has_sip_ctx":bool(sip_context_injection)},"timestamp":int(_dbg_time2.time()*1000)})+"\n")
    except: pass
    # #endregion
    logger.info(f"📝 System prompt: {len(system_prompt)} caratteri")
    
    # Inizializza VideoFrameExtractor PRIMA dell'Agent
    frame_extractor = VideoFrameExtractor()
    logger.info("📹 VideoFrameExtractor inizializzato")
    
    # Crea l'agent con capacità vision (sottoclasse di Agent con function tools)
    agent = VisionAgent(
        instructions=system_prompt,
        vad=vad,
        stt=my_stt,
        llm=my_llm,
        tts=my_tts,
    )

    session_conversation = {
        "room_name": room_name,
        "is_sip": _is_sip_call,
        "session_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "system_prompt": system_prompt,
        "llm_provider": _component_info.get("llm", "unknown"),
        "turns": [],
    }

    def _post_debug_snapshot(sess_conv):
        """Invia snapshot sessione al server per debug API."""
        try:
            import urllib.request
            data = json.dumps(sess_conv).encode()
            req = urllib.request.Request(
                "http://127.0.0.1:8080/api/debug/session",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass

    _post_debug_snapshot(session_conversation)

    
    # Verifica tools registrati
    if hasattr(agent, '_tools') and agent._tools:
        logger.info(f"📹 Tools registrati: {len(agent._tools)} funzioni")
    else:
        logger.warning("⚠️ Nessun tool registrato nell'agent!")
    
    logger.info("VisionAgent creato con function tools, creo AgentSession...")
    
    # Crea sessione
    session = AgentSession()
    
    logger.info("AgentSession creata, avvio...")
    
    # Import RoomOptions per configurazione avanzata (API moderna)
    from livekit.agents.voice.room_io import RoomOptions
    
    # Configura room options: 
    # - NON chiudere la sessione quando un partecipante si disconnette
    # - DISABILITA audio input automatico per gestire manualmente TUTTI i partecipanti
    room_opts = RoomOptions(
        close_on_disconnect=False,
        audio_input=False  # Disabilita per gestire manualmente tutti i partecipanti
    )
    
    # Avvia la sessione con le opzioni configurate
    await session.start(agent, room=ctx.room, room_options=room_opts)
    
    logger.info("AgentSession avviata!")
    
    # ==================== INIZIALIZZA VAD MONITOR ====================
    # Crea callback thread-safe per l'interrupt
    main_loop = asyncio.get_event_loop()
    
    def vad_interrupt_callback():
        """Callback chiamato dal thread VAD quando rileva barge-in"""
        try:
            # Usa run_coroutine_threadsafe per chiamare l'interrupt dal thread VAD
            future = asyncio.run_coroutine_threadsafe(
                _async_interrupt_from_vad(session),
                main_loop
            )
            # Non aspettiamo il risultato per non bloccare il thread VAD
        except Exception as e:
            logger.error(f"🎤 [VAD] Errore nel callback interrupt: {e}")
    
    # Crea e avvia il VAD monitor
    global _vad_monitor
    _vad_monitor = VADMonitor(
        interrupt_callback=vad_interrupt_callback,
        energy_threshold=VAD_ENERGY_THRESHOLD  # Soglia configurabile da database
    )
    _vad_monitor.start()
    logger.info(f"🎤 [VAD] Monitor inizializzato: threshold={VAD_ENERGY_THRESHOLD}, min_frames=6, cooldown=1.0s")
    
    # Imposta dipendenze per VisionAgent (frame_extractor, llm, settings, session)
    VisionAgent.set_vision_dependencies(frame_extractor, base_llm, db_settings, session)
    logger.info("📹 VisionAgent dipendenze vision impostate")
    
    # Imposta callback per comandi video vocali (fallback per modelli senza function calling)
    async def video_analysis_callback(analysis_type: str):
        """Callback per gestire comandi video vocali"""
        await handle_video_analysis(analysis_type, frame_extractor, send_to_frontend, base_llm, db_settings, session)
    
    set_video_analysis_callback(video_analysis_callback, session)
    logger.info("📹 Callback analisi video vocale impostato")
    
    # === GESTIONE AUDIO MULTI-PARTECIPANTE ===
    # Dizionario per tracciare i task audio attivi per ogni partecipante
    audio_processing_tasks: dict[str, asyncio.Task] = {}
    
    async def process_participant_audio(participant_identity: str, track: rtc.Track):
        """Processa l'audio di un singolo partecipante con STT"""
        is_sip_participant = participant_identity.startswith("sip_")
        if is_sip_participant:
            logger.info(f"📞 [SIP-AUDIO] Avvio processing audio per chiamata SIP: {participant_identity}")
        else:
            logger.info(f"🎤 [MULTI-AUDIO] Avvio processing audio per {participant_identity}")
        
        try:
            # Crea AudioStream per questa traccia
            audio_stream = rtc.AudioStream(
                track,
                sample_rate=16000,  # Whisper usa 16kHz
                num_channels=1
            )
            
            # Buffer per accumulare audio (Voice Activity Detection semplice)
            audio_buffer = bytearray()
            silence_frames = 0
            speech_frames = 0
            MIN_SPEECH_FRAMES = 30  # ~1.5 secondi di speech prima di trascrivere
            MIN_AUDIO_BYTES = 32000  # Almeno 1 secondo di audio (16kHz * 16bit = 32 bytes/ms * 1000ms)
            # NOTA: SILENCE_THRESHOLD è ora globale e configurabile da database (default 60 = ~3 sec silenzio)
            
            async for event in audio_stream:
                if not isinstance(event, rtc.AudioFrameEvent):
                    continue
                    
                frame = event.frame
                audio_data = bytes(frame.data)
                
                # ==================== ALIMENTA VAD MONITOR (SEMPRE) ====================
                # Il VAD monitor gira in un thread separato e può rilevare barge-in
                # anche durante il TTS, perché non è bloccato dal loop asyncio
                vad = get_vad_monitor()
                if vad:
                    vad.feed_audio(audio_data)
                
                # Calcola energia audio per VAD semplice
                samples = [int.from_bytes(audio_data[i:i+2], 'little', signed=True) 
                          for i in range(0, len(audio_data), 2)]
                if samples:
                    energy = sum(abs(s) for s in samples) / len(samples)
                else:
                    energy = 0
                
                # Soglia energia per rilevare speech (configurabile da database)
                # NOTA: SPEECH_ENERGY_THRESHOLD è globale e configurabile
                
                # ==================== CHECK FLAG TTS MANUALE ====================
                # NOTA: session.agent_state NON ritorna 'speaking' durante TTS (bug LiveKit?)
                # Uso il flag manuale is_tts_speaking() impostato prima di session.say()
                tts_active = is_tts_speaking()
                tts_cooldown = is_in_tts_cooldown()
                
                # Scarta audio sia durante TTS che durante cooldown (per evitare eco)
                should_discard = tts_active or tts_cooldown
                
                
                # ==================== SCARTA AUDIO DURANTE TTS E COOLDOWN ====================
                # Se il TTS è attivo o siamo in cooldown, SCARTA TUTTO l'audio in ingresso
                if should_discard:
                    
                    # Se c'è voce significativa durante TTS (non cooldown), interrompi
                    if tts_active and energy > VAD_ENERGY_THRESHOLD:  # Soglia configurabile da DB (era hardcoded 70)
                        
                        logger.info(f"✋ [BARGE-IN] Voce durante TTS (energia: {energy:.0f}) - INTERRUPT")
                        try:
                            await session.interrupt()
                            set_tts_speaking(False)  # Reset flag dopo interrupt
                            logger.info(f"✋ [BARGE-IN] TTS interrotto")
                        except Exception as e:
                            logger.error(f"✋ [BARGE-IN] Errore: {e}")
                        request_cancel_llm()
                    # Scarta buffer e resetta contatori
                    audio_buffer.clear()
                    speech_frames = 0
                    silence_frames = 0
                    continue  # SEMPRE scarta durante TTS
                
                # ==================== PROCESSING NORMALE (agent non sta parlando) ====================
                if energy > SPEECH_ENERGY_THRESHOLD:
                    # Speech rilevato, aggiungi al buffer
                    audio_buffer.extend(audio_data)
                    speech_frames += 1
                    silence_frames = 0
                    # Log periodico per SIP (ogni 20 frames = ~1 secondo)
                    if is_sip_participant and speech_frames % 20 == 1:
                        logger.debug(f"📞 [SIP-AUDIO] Speech da {participant_identity}: energy={energy:.0f}, frames={speech_frames}")
                elif len(audio_buffer) > 0:
                    # Silenzio dopo speech
                    audio_buffer.extend(audio_data)
                    silence_frames += 1
                    
                    if silence_frames >= SILENCE_THRESHOLD and speech_frames >= MIN_SPEECH_FRAMES:
                        # Fine utterance - trascrivi
                        audio_bytes = bytes(audio_buffer)
                        audio_buffer.clear()
                        silence_frames = 0
                        speech_frames = 0
                        
                        if len(audio_bytes) > MIN_AUDIO_BYTES:  # Almeno 500ms di audio
                            # Calcola statistiche audio per debug
                            audio_duration_ms = len(audio_bytes) / 32  # 16kHz, 16bit = 32 bytes/ms
                            audio_samples = [int.from_bytes(audio_bytes[i:i+2], 'little', signed=True) for i in range(0, min(len(audio_bytes), 10000), 2)]
                            avg_energy = sum(abs(s) for s in audio_samples) / len(audio_samples) if audio_samples else 0
                            max_energy = max(abs(s) for s in audio_samples) if audio_samples else 0
                            
                            
                            if is_sip_participant:
                                logger.info(f"📞 [SIP-AUDIO] {participant_identity}: {len(audio_bytes)} bytes ({audio_duration_ms:.0f}ms), energy avg={avg_energy:.0f} max={max_energy}")
                            else:
                                logger.info(f"🎤 [MULTI-AUDIO] {participant_identity}: {len(audio_bytes)} bytes ({audio_duration_ms:.0f}ms), energy avg={avg_energy:.0f}")
                            
                            # Trascrivi con WhisperSTT (metodo dedicato senza invio automatico)
                            try:
                                text = await my_stt.transcribe_only(audio_bytes, 16000)
                                
                                if text and len(text) > 1:
                                    if is_sip_participant:
                                        logger.info(f"📞 [SIP-AUDIO] Trascrizione Whisper da {participant_identity}: '{text}'")
                                    else:
                                        logger.info(f"🎤 [MULTI-AUDIO] {participant_identity} dice: {text}")
                                    
                                    # ==================== TTS INTERRUPT ====================
                                    # Se TTS attivo, interrompi SEMPRE e non processare
                                    if is_tts_speaking():
                                        logger.info(f"✋ [MULTI-AUDIO] TTS attivo - STOP immediato")
                                        await interrupt_tts_if_speaking()
                                        continue  # Non processare questo messaggio
                                    
                                    # Invia al frontend con il nome del partecipante
                                    msg_id = generate_message_id()
                                    await send_to_frontend(text, "user", msg_id, participant_identity)
                                    
                                    # ==================== SALVA MESSAGGIO UTENTE NEL DB ====================
                                    # Per chiamate SIP, salva anche i messaggi utente nel log chiamata
                                    if _is_sip_call and _current_call_log_id:
                                        try:
                                            import aiohttp
                                            server_url = os.getenv("WEB_SERVER_URL", "http://voice-agent-web:8080")
                                            async with aiohttp.ClientSession() as http_session:
                                                async with http_session.post(
                                                    f"{server_url}/api/calls/{_current_call_log_id}/message",
                                                    params={"role": "user", "content": text}
                                                ) as resp:
                                                    if resp.status == 200:
                                                        logger.debug(f"📝 [SIP] Messaggio utente salvato nel log")
                                                    else:
                                                        logger.warning(f"⚠️ Errore salvataggio messaggio utente: {resp.status}")
                                        except Exception as e:
                                            logger.warning(f"⚠️ Impossibile salvare messaggio utente: {e}")
                                    
                                    # ==================== SIP ROOM DEBUG MODE ====================
                                    # Per stanze SIP, solo i partecipanti SIP attivano risposte
                                    # I client web in stanze SIP sono in modalità debug/ascolto
                                    if _is_sip_call and not participant_identity.startswith("sip_"):
                                        logger.info(f"🔇 [SIP-DEBUG] Partecipante web {participant_identity} in ascolto - no LLM response")
                                        continue  # Non processare con LLM, la trascrizione è già inviata al frontend
                                    
                                    # Verifica se l'agent deve rispondere (con participant_id per wake sessions)
                                    should_respond, cleaned_text, is_wake = should_agent_respond(text, participant_identity)
                                    
                                    if is_wake:
                                        # Wake word rilevato - pronuncia "Dimmi"
                                        logger.info(f"🎤 [MULTI-AUDIO] Wake word per {participant_identity} - pronuncio Dimmi")
                                        await handle_wake_word_detected(participant_identity)
                                        continue  # Non processare oltre
                                    
                                    if should_respond and cleaned_text:
                                        # Wake session attiva o partecipante SIP, rispondi
                                        if is_sip_participant:
                                            logger.info(f"📞 [SIP-AUDIO] Invio a LLM per risposta: '{cleaned_text[:50]}...'")
                                        await handle_agent_response_only(session, cleaned_text, send_to_frontend, participant_identity)
                            except Exception as e:
                                logger.error(f"🎤 [MULTI-AUDIO] Errore STT per {participant_identity}: {e}")
                                
        except asyncio.CancelledError:
            logger.info(f"🎤 [MULTI-AUDIO] Processing audio cancellato per {participant_identity}")
        except Exception as e:
            logger.error(f"🎤 [MULTI-AUDIO] Errore processing audio per {participant_identity}: {e}")
    
    # Handler per video E audio tracks
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.info(f"📹 Video track ricevuto da {participant.identity}")
            frame_extractor.register_video_track(participant.identity, track)
            logger.info(f"📹 Video tracks attivi: {len(frame_extractor.video_tracks)}")
        
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            # Avvia processing audio per questo partecipante
            participant_id = participant.identity
            # FILTRO: Ignora audio da altri agent (evita loop di auto-risposta)
            if participant_id.startswith("agent-"):
                logger.info(f"🎤 [MULTI-AUDIO] Ignoro traccia audio da agent: {participant_id}")
                return
            if participant_id not in audio_processing_tasks:
                logger.info(f"🎤 [MULTI-AUDIO] Nuova traccia audio da {participant_id}")
                task = asyncio.create_task(process_participant_audio(participant_id, track))
                audio_processing_tasks[participant_id] = task
    
    @ctx.room.on("track_unsubscribed")
    def on_track_unsubscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.info(f"📹 Video track rimosso da {participant.identity}")
            frame_extractor.unregister_video_track(participant.identity)
        
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            # Cancella processing audio per questo partecipante
            participant_id = participant.identity
            if participant_id in audio_processing_tasks:
                logger.info(f"🎤 [MULTI-AUDIO] Rimuovo traccia audio da {participant_id}")
                audio_processing_tasks[participant_id].cancel()
                del audio_processing_tasks[participant_id]
    
    # ==================== GESTIONE PARTECIPANTI GIA' PRESENTI ====================
    # Se ci sono partecipanti già nella room quando l'agent si connette,
    # le loro tracce potrebbero essere già state sottoscritte PRIMA della registrazione
    # dell'handler. Iteriamo sui partecipanti esistenti per gestire le loro tracce.
    logger.info(f"🔍 Controllo partecipanti già presenti nella room...")
    for participant in ctx.room.remote_participants.values():
        logger.info(f"👤 Partecipante trovato: {participant.identity}")
        for publication in participant.track_publications.values():
            if publication.track is not None:
                track = publication.track
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    participant_id = participant.identity
                    # Ignora agent
                    if participant_id.startswith("agent-"):
                        logger.info(f"🎤 [MULTI-AUDIO] Ignoro traccia audio da agent esistente: {participant_id}")
                        continue
                    if participant_id not in audio_processing_tasks:
                        logger.info(f"🎤 [MULTI-AUDIO] Sottoscrivo traccia audio esistente da {participant_id}")
                        task = asyncio.create_task(process_participant_audio(participant_id, track))
                        audio_processing_tasks[participant_id] = task
                elif track.kind == rtc.TrackKind.KIND_VIDEO:
                    logger.info(f"📹 Video track esistente da {participant.identity}")
                    frame_extractor.register_video_track(participant.identity, track)
    
    logger.info(f"🎤 Audio processing tasks attivi: {len(audio_processing_tasks)}")
    
    # Imposta callback per inviare trascrizioni al frontend
    async def send_to_frontend(text: str, role: str, message_id: str = None, sender: str = None):
        """Invia trascrizione al frontend via data channel con ID univoco"""
        try:
            # Genera ID se non fornito
            if not message_id:
                message_id = generate_message_id()
            
            # Se il testo è già un JSON raw (es. video_analysis_result), aggiungi ID se mancante
            if text.startswith('{') and '"type":' in text:
                try:
                    obj = json.loads(text)
                    if 'id' not in obj:
                        obj['id'] = message_id
                    data = json.dumps(obj)
                except:
                    data = text
            else:
                # Includi sender per identificare chi ha inviato il messaggio
                # Per agent usa sempre il nome configurato, per user usa il sender passato
                sender_name = ASSISTANT_NAME if role == "assistant" else sender
                data = json.dumps({"type": "transcript", "text": text, "role": role, "id": message_id, "sender": sender_name})
            await ctx.room.local_participant.publish_data(data.encode(), reliable=True)
            logger.info(f"📤 [FRONTEND] id={message_id} {role} (sender={sender_name}): {text[:50]}...")
        except Exception as e:
            logger.error(f"Errore invio al frontend: {e}")
    
    set_transcript_callback(send_to_frontend)
    
    # ==================== WAKE WORD SYSTEM SETUP ====================
    async def send_wake_update(wake_data: dict):
        """Invia aggiornamenti wake session al frontend"""
        try:
            data = json.dumps(wake_data)
            await ctx.room.local_participant.publish_data(data.encode(), reliable=True)
            logger.debug(f"📤 [WAKE] {wake_data.get('type')}: {wake_data}")
        except Exception as e:
            logger.error(f"Errore invio wake update: {e}")
    
    set_wake_callback(send_wake_update)
    start_wake_countdown_task()
    logger.info("🎤 Wake word system inizializzato")
    
    # Handler per messaggi dal frontend (es. interrupt, text_message)
    @ctx.room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        try:
            msg = json.loads(data.data.decode())
            msg_type = msg.get("type")
            
            if msg_type == "interrupt":
                logger.info("✋ Richiesta interruzione dal frontend")
                try:
                    # session.interrupt() può essere sync o async
                    result = session.interrupt()
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                    set_tts_speaking(False)  # Reset flag TTS
                    request_cancel_llm()  # Cancella anche LLM
                    logger.info("✋ Interrupt eseguito con successo")
                except Exception as e:
                    logger.error(f"✋ Errore durante interrupt: {e}")
            
            elif msg_type == "text_message":
                # Messaggio testuale dall'utente
                text = msg.get("text", "").strip()
                sender_identity = msg.get("sender") or (data.participant.identity if hasattr(data, 'participant') and data.participant else None)
                if text:
                    logger.info(f"📝 Messaggio testuale ricevuto da {sender_identity}: {text}")
                    # Processa come se fosse stato detto vocalmente
                    asyncio.create_task(handle_text_message(session, text, send_to_frontend, sender_identity))
            
            elif msg_type == "video_analysis":
                # Richiesta analisi video
                analysis_type = msg.get("analysis_type", "generic")
                logger.info(f"📹 Richiesta analisi video: {analysis_type}")
                asyncio.create_task(handle_video_analysis(analysis_type, frame_extractor, send_to_frontend, base_llm, db_settings, session))

            elif msg_type == "image_analysis":
                # Richiesta analisi immagine caricata
                image_base64 = msg.get("image_base64", "")
                analysis_type = msg.get("analysis_type", "generic")
                custom_prompt = msg.get("custom_prompt", "")
                logger.info(f"🖼️ Richiesta analisi immagine caricata: {analysis_type}")
                asyncio.create_task(handle_image_analysis(image_base64, analysis_type, custom_prompt, send_to_frontend, base_llm, db_settings, session))
            
            elif msg_type == "participants_count":
                # Aggiornamento conteggio partecipanti umani
                count = msg.get("count", 1)
                set_human_participants_count(count)
            
            elif msg_type == "force_agent_response":
                # Toggle per forzare risposta agent (bottone "Parla con Sophy")
                force = msg.get("force", False)
                set_force_agent_response(force)

            elif msg_type == "reset_context":
                session_conversation["turns"].clear()
                logger.info("🔄 Reset storico conversazione (nuova chat)")
                _post_debug_snapshot(session_conversation)

        except Exception as e:
            logger.error(f"Errore parsing messaggio frontend: {e}")

    async def _speak_tts_task(session: AgentSession, text: str):
        """Task helper per eseguire TTS senza bloccare il loop audio"""
        try:
            await session.say(text, allow_interruptions=True)
        finally:
            set_tts_speaking(False)

    async def handle_text_message(session: AgentSession, user_text: str, send_callback, sender_identity: str = None):
        """Gestisce un messaggio testuale dal frontend e genera sempre una risposta agent."""
        # Genera ID univoco per il messaggio utente
        user_message_id = generate_message_id()
        
        # SEMPRE invia il messaggio utente a tutti i partecipanti (broadcast)
        logger.info(f"💬 Broadcast messaggio utente: {user_text[:50]}... (sender={sender_identity})")
        await send_callback(user_text, "user", user_message_id, sender_identity)
        
        # Verifica se l'agent deve rispondere (cerca @sophyai, wake word, o sessione attiva)
        should_respond, cleaned_text, is_wake = should_agent_respond(user_text, sender_identity or "text_user")
        
        if is_wake:
            logger.info(f"🎤 Wake session attivata da messaggio testuale di {sender_identity}")

        # Per i messaggi testuali espliciti dal frontend rispondiamo sempre.
        # Il gate wake/mention resta valido per il solo canale vocale.
        if not should_respond or not cleaned_text.strip():
            cleaned_text = user_text
        
        # Genera ID univoco per la risposta dell'agent
        text_response_id = generate_message_id()
        
        try:
            logger.info(f"💬 Elaboro messaggio testuale (menzionato @sophyai): {cleaned_text} (response_id={text_response_id})")
            
            # Nota: l'analisi video è ora gestita via function calling dall'LLM
            # Non serve più pattern matching manuale
            
            # Costruisci chat context con storico multi-turno
            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="system", content=agent._instructions)
            for prev_turn in session_conversation["turns"]:
                chat_ctx.add_message(role="user", content=prev_turn["user_message"])
                chat_ctx.add_message(role="assistant", content=prev_turn["assistant_response"])
            chat_ctx.add_message(role="user", content=cleaned_text)

            # Cattura snapshot raw del contesto inviato al LLM
            llm_raw_messages = [{"role": "system", "content": agent._instructions}]
            for prev_turn in session_conversation["turns"]:
                llm_raw_messages.append({"role": "user", "content": prev_turn["user_message"]})
                llm_raw_messages.append({"role": "assistant", "content": prev_turn["assistant_response"]})
            llm_raw_messages.append({"role": "user", "content": cleaned_text})

            tools_for_chat = []
            raw_tools = getattr(agent, "_tools", None)
            if isinstance(raw_tools, dict):
                tools_for_chat = list(raw_tools.values())
            elif isinstance(raw_tools, (list, tuple)):
                tools_for_chat = list(raw_tools)
            
            t_start = time.time()
            response_text = ""
            llm_cancelled = False
            turn_tool_calls = []
            reset_cancel_llm()
            llm_conn_options = APIConnectOptions(max_retry=1, retry_interval=1.0, timeout=60.0)
            stream = my_llm.chat(
                chat_ctx=chat_ctx,
                tools=tools_for_chat,
                conn_options=llm_conn_options,
                extra_kwargs=llm_chat_extra_kwargs,
            )
            
            from livekit.agents.llm.chat_context import FunctionCall as _FunctionCall, FunctionCallOutput as _FunctionCallOutput

            MAX_TOOL_ROUNDS = 3
            for _tool_round in range(MAX_TOOL_ROUNDS + 1):
                pending_tool_calls = []
                async for chunk in stream:
                    if should_cancel_llm():
                        logger.info("🛑 Risposta LLM ANNULLATA (utente ha interrotto)")
                        llm_cancelled = True
                        break

                    if hasattr(chunk, 'delta') and chunk.delta:
                        d = chunk.delta
                        if hasattr(d, 'content') and d.content:
                            response_text += d.content
                        if hasattr(d, 'tool_calls') and d.tool_calls:
                            pending_tool_calls.extend(d.tool_calls)
                    elif hasattr(chunk, 'choices') and chunk.choices:
                        for choice in chunk.choices:
                            if hasattr(choice, 'content') and choice.content:
                                response_text += choice.content
                            if hasattr(choice, 'tool_calls') and choice.tool_calls:
                                pending_tool_calls.extend(choice.tool_calls)

                if llm_cancelled:
                    break

                if not pending_tool_calls:
                    break

                logger.info(f"🔧 [TOOL] Round {_tool_round+1}: {len(pending_tool_calls)} tool call(s)")

                for tc in pending_tool_calls:
                    fn_name = tc.name if hasattr(tc, 'name') else "unknown"
                    fn_args_raw = tc.arguments if hasattr(tc, 'arguments') else "{}"
                    tc_id = tc.call_id if hasattr(tc, 'call_id') else "unknown"
                    logger.info(f"🔧 [TOOL] Eseguo {fn_name}({fn_args_raw}) call_id={tc_id}")

                    # Invia evento tool_call request al frontend
                    try:
                        tc_args_parsed = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except Exception:
                        tc_args_parsed = fn_args_raw
                    await send_callback(json.dumps({"type": "tool_call", "phase": "request", "tool_name": fn_name, "call_id": tc_id, "arguments": tc_args_parsed, "round": _tool_round + 1}), "system", generate_message_id())

                    chat_ctx.insert(_FunctionCall(call_id=tc_id, name=fn_name, arguments=fn_args_raw))

                    tool_result = f"Tool {fn_name} non trovato."
                    is_error = False
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                        if fn_name == "get_current_datetime":
                            tool_result = await agent.get_current_datetime(context=None)
                        elif fn_name == "check_room_availability":
                            tool_result = await agent.check_room_availability(
                                context=None,
                                start_date=fn_args.get("start_date", ""),
                                end_date=fn_args.get("end_date", ""),
                                count=int(fn_args.get("count", 2)),
                            )
                        else:
                            logger.warning(f"🔧 [TOOL] Funzione {fn_name} non trovata tra i tools registrati")
                            is_error = True
                    except Exception as tool_err:
                        logger.error(f"🔧 [TOOL] Errore esecuzione {fn_name}: {tool_err}")
                        tool_result = f"Errore nell'esecuzione del tool: {tool_err}"
                        is_error = True

                    logger.info(f"🔧 [TOOL] Risultato {fn_name}: {str(tool_result)[:200]}")
                    chat_ctx.insert(_FunctionCallOutput(call_id=tc_id, name=fn_name, output=str(tool_result), is_error=is_error))

                    # Invia evento tool_call response al frontend
                    await send_callback(json.dumps({"type": "tool_call", "phase": "response", "tool_name": fn_name, "call_id": tc_id, "result": str(tool_result)[:500], "is_error": is_error, "round": _tool_round + 1}), "system", generate_message_id())
                    turn_tool_calls.append({"function_name": fn_name, "arguments": tc_args_parsed, "result": str(tool_result)[:500], "is_error": is_error})

                response_text = ""
                stream = my_llm.chat(
                    chat_ctx=chat_ctx,
                    tools=tools_for_chat,
                    conn_options=llm_conn_options,
                    extra_kwargs=llm_chat_extra_kwargs,
                )

            if llm_cancelled:
                logger.info("🛑 TTS saltato - risposta annullata")
                return

            t_llm = time.time()
            llm_elapsed_ms = int((t_llm - t_start) * 1000)
            logger.info(f"🤖 [LLM] Risposta in {llm_elapsed_ms}ms: {response_text[:100]}...")

            await send_callback(response_text, "assistant", text_response_id)

            session_conversation["turns"].append({
                "turn_id": len(session_conversation["turns"]) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_message": cleaned_text,
                "assistant_response": response_text,
                "tool_calls": turn_tool_calls,
                "llm_elapsed_ms": llm_elapsed_ms,
                "llm_context_messages": llm_raw_messages,
                "llm_provider": _component_info.get("llm", "unknown"),
                "channel": "chat",
            })
            _post_debug_snapshot(session_conversation)

            asyncio.create_task(send_conversation_to_server({
                "stt_ms": 0,
                "llm_ms": llm_elapsed_ms,
                "llm_ttft_ms": 0,
                "tts_ms": 0,
                "e2e_ms": llm_elapsed_ms,
                "speech_to_tts_ms": 0,
                "stt_type": "text",
                "llm_type": _component_info.get("llm", "unknown"),
                "tts_type": "text",
                "user_text": cleaned_text[:100] if cleaned_text else "",
                "agent_text": response_text[:100] if response_text else "",
                "sender": sender_identity or "chat",
            }))

            set_tts_speaking(True)
            asyncio.create_task(_speak_tts_task(session, response_text))
            
        except Exception as e:
            set_tts_speaking(False)
            logger.error(f"Errore gestione messaggio testuale: {e}")
            try:
                fallback_id = generate_message_id()
                fallback_text = "Sto impiegando più tempo del previsto per elaborare la richiesta. Riprova tra pochi secondi."
                await send_callback(fallback_text, "assistant", fallback_id, "Receptionist")
            except Exception:
                pass
    
    async def handle_agent_response_only(session: AgentSession, user_text: str, send_callback, sender_identity: str = None):
        """Gestisce solo la risposta dell'agent (messaggio user già inviato da multi-audio)"""
        # Il testo è già pulito dal trigger/wake word, usa direttamente
        cleaned_text = user_text
        
        # Genera ID univoco per la risposta dell'agent
        text_response_id = generate_message_id()
        
        try:
            logger.info(f"🤖 [MULTI-AUDIO] Genero risposta per {sender_identity}: {cleaned_text[:50]}...")
            
            # Costruisci chat context con storico multi-turno
            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="system", content=agent._instructions)
            for prev_turn in session_conversation["turns"]:
                chat_ctx.add_message(role="user", content=prev_turn["user_message"])
                chat_ctx.add_message(role="assistant", content=prev_turn["assistant_response"])
            chat_ctx.add_message(role="user", content=cleaned_text)

            llm_raw_messages = [{"role": "system", "content": agent._instructions}]
            for prev_turn in session_conversation["turns"]:
                llm_raw_messages.append({"role": "user", "content": prev_turn["user_message"]})
                llm_raw_messages.append({"role": "assistant", "content": prev_turn["assistant_response"]})
            llm_raw_messages.append({"role": "user", "content": cleaned_text})

            tools_for_chat = []
            raw_tools = getattr(agent, "_tools", None)
            if isinstance(raw_tools, dict):
                tools_for_chat = list(raw_tools.values())
            elif isinstance(raw_tools, (list, tuple)):
                tools_for_chat = list(raw_tools)
            
            t_start = time.time()
            response_text = ""
            llm_cancelled = False
            turn_tool_calls = []
            reset_cancel_llm()
            llm_conn_options = APIConnectOptions(max_retry=1, retry_interval=1.0, timeout=60.0)
            from livekit.agents.llm.chat_context import FunctionCall as _FunctionCall, FunctionCallOutput as _FunctionCallOutput

            stream = my_llm.chat(
                chat_ctx=chat_ctx,
                tools=tools_for_chat,
                conn_options=llm_conn_options,
                extra_kwargs=llm_chat_extra_kwargs,
            )

            _MAX_TOOL_ROUNDS = 3
            for _tround in range(_MAX_TOOL_ROUNDS + 1):
                _pending_tc = []
                async for chunk in stream:
                    if should_cancel_llm():
                        logger.info("🛑 [MULTI-AUDIO] Risposta LLM ANNULLATA (utente ha interrotto)")
                        llm_cancelled = True
                        break

                    if hasattr(chunk, 'delta') and chunk.delta:
                        d = chunk.delta
                        if hasattr(d, 'content') and d.content:
                            response_text += d.content
                        if hasattr(d, 'tool_calls') and d.tool_calls:
                            _pending_tc.extend(d.tool_calls)
                    elif hasattr(chunk, 'choices') and chunk.choices:
                        for choice in chunk.choices:
                            if hasattr(choice, 'content') and choice.content:
                                response_text += choice.content
                            if hasattr(choice, 'tool_calls') and choice.tool_calls:
                                _pending_tc.extend(choice.tool_calls)

                if llm_cancelled:
                    break

                if not _pending_tc:
                    break

                logger.info(f"🔧 [MULTI-AUDIO TOOL] Round {_tround+1}: {len(_pending_tc)} tool call(s)")

                for tc in _pending_tc:
                    fn_name = tc.name if hasattr(tc, 'name') else "unknown"
                    fn_args_raw = tc.arguments if hasattr(tc, 'arguments') else "{}"
                    tc_id = tc.call_id if hasattr(tc, 'call_id') else "unknown"
                    logger.info(f"🔧 [MULTI-AUDIO TOOL] Eseguo {fn_name}({fn_args_raw}) call_id={tc_id}")

                    try:
                        tc_args_parsed = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                    except Exception:
                        tc_args_parsed = fn_args_raw
                    await send_callback(json.dumps({"type": "tool_call", "phase": "request", "tool_name": fn_name, "call_id": tc_id, "arguments": tc_args_parsed, "round": _tround + 1}), "system", generate_message_id())

                    chat_ctx.insert(_FunctionCall(call_id=tc_id, name=fn_name, arguments=fn_args_raw))

                    tool_result = f"Tool {fn_name} non trovato."
                    is_error = False
                    try:
                        fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
                        if fn_name == "get_current_datetime":
                            tool_result = await agent.get_current_datetime(context=None)
                        elif fn_name == "check_room_availability":
                            tool_result = await agent.check_room_availability(
                                context=None,
                                start_date=fn_args.get("start_date", ""),
                                end_date=fn_args.get("end_date", ""),
                                count=int(fn_args.get("count", 2)),
                            )
                        else:
                            logger.warning(f"🔧 [MULTI-AUDIO TOOL] Funzione {fn_name} non trovata")
                            is_error = True
                    except Exception as tool_err:
                        logger.error(f"🔧 [MULTI-AUDIO TOOL] Errore {fn_name}: {tool_err}")
                        tool_result = f"Errore nell'esecuzione del tool: {tool_err}"
                        is_error = True

                    logger.info(f"🔧 [MULTI-AUDIO TOOL] Risultato {fn_name}: {str(tool_result)[:200]}")
                    chat_ctx.insert(_FunctionCallOutput(call_id=tc_id, name=fn_name, output=str(tool_result), is_error=is_error))

                    await send_callback(json.dumps({"type": "tool_call", "phase": "response", "tool_name": fn_name, "call_id": tc_id, "result": str(tool_result)[:500], "is_error": is_error, "round": _tround + 1}), "system", generate_message_id())
                    turn_tool_calls.append({"function_name": fn_name, "arguments": tc_args_parsed, "result": str(tool_result)[:500], "is_error": is_error})

                response_text = ""
                stream = my_llm.chat(
                    chat_ctx=chat_ctx,
                    tools=tools_for_chat,
                    conn_options=llm_conn_options,
                    extra_kwargs=llm_chat_extra_kwargs,
                )

            if llm_cancelled:
                logger.info("🛑 [MULTI-AUDIO] TTS saltato - risposta annullata")
                return
            
            t_llm = time.time()
            llm_elapsed_ms = int((t_llm - t_start) * 1000)
            logger.info(f"🤖 [LLM] Risposta in {llm_elapsed_ms}ms: {response_text[:100]}...")
            
            await send_callback(response_text, "assistant", text_response_id)

            session_conversation["turns"].append({
                "turn_id": len(session_conversation["turns"]) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_message": cleaned_text,
                "assistant_response": response_text,
                "tool_calls": turn_tool_calls,
                "llm_elapsed_ms": llm_elapsed_ms,
                "llm_context_messages": llm_raw_messages,
                "llm_provider": _component_info.get("llm", "unknown"),
                "channel": "voice",
            })
            _post_debug_snapshot(session_conversation)
            
            tts_text = response_text
            # Rimuovi contenuti tra parentesi quadre [...]
            tts_text = re.sub(r'\[.*?\]', '', tts_text)
            # Rimuovi contenuti tra parentesi tonde (...)
            tts_text = re.sub(r'\(.*?\)', '', tts_text)
            # Rimuovi asterischi * e **
            tts_text = re.sub(r'\*+', '', tts_text)
            # Rimuovi underscore _ e __
            tts_text = re.sub(r'_+', ' ', tts_text)
            # Rimuovi hashtag #
            tts_text = re.sub(r'#+\s*', '', tts_text)
            # Rimuovi backtick ` e ```
            tts_text = re.sub(r'`+', '', tts_text)
            # Rimuovi caratteri speciali comuni che non si pronunciano
            tts_text = re.sub(r'[<>{}|\\^~]', '', tts_text)
            # Rimuovi URL http/https
            tts_text = re.sub(r'https?://\S+', '', tts_text)
            # Rimuovi emoji (range Unicode comuni)
            tts_text = re.sub(r'[\U0001F300-\U0001F9FF]', '', tts_text)
            # Rimuovi spazi multipli
            tts_text = re.sub(r' +', ' ', tts_text)
            # Rimuovi righe vuote multiple
            tts_text = re.sub(r'\n\s*\n', '\n', tts_text).strip()
            
            # Traccia stato TTS speaking
            # NOTA: Non usiamo await per non bloccare - il TTS gira in parallelo
            set_tts_speaking(True)
            
            # Cattura dati per conversation tracking
            _turn_stt_end = _last_stt_end_time
            _turn_stt_ms = _last_stt_time_ms
            _turn_llm_ms = (t_llm - t_start) * 1000
            _turn_llm_ttft = _last_llm_ttft_ms  # TTFT catturato dal TimedLLMStream
            _turn_user_text = user_text
            _turn_agent_text = response_text
            
            async def speak_and_reset():
                try:
                    global _last_tts_time_ms
                    _last_tts_time_ms = 0
                    t_tts_start_wall = time.time()
                    await session.say(tts_text, allow_interruptions=True)
                    t_tts_end_wall = time.time()
                    
                    # Usa il tempo TTS misurato internamente, o il wall clock come fallback
                    tts_ms = _last_tts_time_ms if _last_tts_time_ms > 0 else (t_tts_end_wall - t_tts_start_wall) * 1000
                    
                    # Calcola latenze
                    e2e_ms = 0
                    speech_to_tts_ms = 0
                    if _turn_stt_end:
                        e2e_ms = (t_tts_end_wall - _turn_stt_end) * 1000
                        # Tempo dalla fine del parlato all'inizio della sintesi TTS
                        speech_to_tts_ms = (t_tts_start_wall - _turn_stt_end) * 1000
                    
                    # Invia record conversazione completo
                    asyncio.create_task(send_conversation_to_server({
                        "stt_ms": int(_turn_stt_ms),
                        "llm_ms": int(_turn_llm_ms),
                        "llm_ttft_ms": int(_turn_llm_ttft),
                        "tts_ms": int(tts_ms),
                        "e2e_ms": int(e2e_ms),
                        "speech_to_tts_ms": int(speech_to_tts_ms),
                        "stt_type": _component_info.get("stt", "whisper"),
                        "llm_type": _component_info.get("llm", "unknown"),
                        "tts_type": _component_info.get("tts", "unknown"),
                        "user_text": _turn_user_text[:100] if _turn_user_text else "",
                        "agent_text": _turn_agent_text[:100] if _turn_agent_text else "",
                        "sender": sender_identity or ""
                    }))
                    
                    logger.info(f"📊 [CONV] STT:{int(_turn_stt_ms)}ms TTFT:{int(_turn_llm_ttft)}ms LLM:{int(_turn_llm_ms)}ms TTS:{int(tts_ms)}ms E2E:{int(e2e_ms)}ms Inizio TTS:{int(speech_to_tts_ms)}ms")
                finally:
                    set_tts_speaking(False)
            
            # Avvia TTS in task separato - NON attendiamo per permettere al loop audio di continuare
            asyncio.create_task(speak_and_reset())
            
        except Exception as e:
            set_tts_speaking(False)  # Reset in caso di errore
            logger.error(f"Errore gestione risposta agent: {e}")
            try:
                fallback_id = generate_message_id()
                fallback_text = "Sto impiegando più tempo del previsto per elaborare la richiesta. Riprova tra pochi secondi."
                await send_callback(fallback_text, "assistant", fallback_id, "Receptionist")
            except Exception:
                pass
    
    # Messaggio di benvenuto
    try:
        set_tts_speaking(True)
        await session.say("Ciao! Come posso aiutarti?")
        set_tts_speaking(False)
    except Exception as e:
        set_tts_speaking(False)
        logger.error(f"❌ Errore nel messaggio di benvenuto: {e}")
        raise
    
    # Mantieni attivo
    await asyncio.Event().wait()


def main():
    """Funzione principale per avviare l'agent"""
    logger.info("Avvio Voice Agent...")
    logger.info(f"LiveKit URL: {config.livekit.url}")
    logger.info(f"Ollama Host: {config.ollama.host}")
    
    # Mostra configurazione TTS all'avvio
    tts_config_file = "/app/config/tts_config.json"
    if os.path.exists(tts_config_file):
        try:
            import json
            with open(tts_config_file, "r") as f:
                tts_cfg = json.load(f)
            logger.info(f"🔊 TTS CONFIG: engine={tts_cfg.get('engine')}, language={tts_cfg.get('language')}")
        except:
            pass
    else:
        logger.info(f"🔊 TTS CONFIG: engine={config.tts.default_engine} (default)")
    
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_handler,  # Accetta automaticamente job SIP
        api_key=config.livekit.api_key,
        api_secret=config.livekit.api_secret,
        ws_url=config.livekit.url,
    )
    
    cli.run_app(worker_options)


if __name__ == "__main__":
    main()
