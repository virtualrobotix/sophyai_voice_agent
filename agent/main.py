"""
Main Voice Agent
Agent principale che orchestra STT, LLM e TTS per conversazioni vocali.
"""

import asyncio
import os
import re
import sys
import time
import uuid
import threading
import queue
from typing import Optional, Callable

import json
import aiohttp
from loguru import logger

# #region debug logging
# Debug log disabilitato in Docker - usa solo logger
def debug_log(hypothesis_id, location, message, data=None):
    """Debug log - ora usa solo il logger standard"""
    try:
        logger.debug(f"[{hypothesis_id}] {location}: {message} | {data}")
    except Exception:
        pass  # Ignora errori di logging
# #endregion
from livekit.agents import (
    JobContext,
    WorkerOptions,
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
_message_counter = 0  # Contatore progressivo per ID messaggi

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
WAKE_TIMEOUT_SECONDS = 20  # Timeout di silenzio per disattivazione automatica
VAD_ENERGY_THRESHOLD = 40  # Soglia energia per barge-in VAD
SPEECH_ENERGY_THRESHOLD = 100  # Soglia energia per rilevamento parlato
SILENCE_THRESHOLD = 30  # Frames di silenzio prima di terminare ascolto

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
    
    # #region agent log - H11: traccia chiamate set_tts_speaking
    import json as _json
    try:
        with open("/app/config/debug.log", "a") as _f:
            _f.write(_json.dumps({"hypothesisId": "H11", "location": "set_tts_speaking", "message": f"SET TTS SPEAKING = {speaking}", "data": {"speaking": speaking, "pid": os.getpid()}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
    except: pass
    # #endregion
    
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
    
    # #region agent log - H11: traccia lettura flag (solo se True)
    if result:
        import json as _json
        try:
            with open("/app/config/debug.log", "a") as _f:
                _f.write(_json.dumps({"hypothesisId": "H11", "location": "is_tts_speaking", "message": "READ TTS FLAG = True (file exists)", "data": {"pid": os.getpid()}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
        except: pass
    # #endregion
    
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
        
        # #region agent log - H14: traccia cooldown
        if in_cooldown:
            import json as _json
            try:
                with open("/app/config/debug.log", "a") as _f:
                    _f.write(_json.dumps({"hypothesisId": "H14", "location": "is_in_tts_cooldown", "message": "IN TTS COOLDOWN", "data": {"elapsed": round(elapsed, 2), "remaining": round(TTS_COOLDOWN_SECONDS - elapsed, 2)}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
            except: pass
        # #endregion
        
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
        self._interrupt_cooldown = 0.5  # Minimo 500ms tra interrupt
        self._consecutive_speech_frames = 0
        self._min_speech_frames = 3  # Richiedi almeno 3 frame consecutivi con voce
    
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
                if is_tts_speaking() and energy > self._energy_threshold:
                    self._consecutive_speech_frames += 1
                    
                    # Log per debug (ogni 5 frame per non spammare)
                    if self._consecutive_speech_frames % 5 == 1:
                        logger.debug(f"🎤 [VAD] Voce rilevata durante TTS: energia={energy:.0f}, frames={self._consecutive_speech_frames}")
                    
                    # Se abbastanza frame consecutivi con voce, interrompi
                    if self._consecutive_speech_frames >= self._min_speech_frames:
                        current_time = time.time()
                        if current_time - self._last_interrupt_time > self._interrupt_cooldown:
                            logger.info(f"🎤 [VAD] BARGE-IN RILEVATO! Energia={energy:.0f}, frames={self._consecutive_speech_frames}")
                            self._interrupt_callback()
                            self._last_interrupt_time = current_time
                            self._consecutive_speech_frames = 0  # Reset
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
    triggers = ["@sophyai", "@sophy"]

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
                stream=True
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
    
    # #region agent log - debug helper
    def _debug_log(self, hypothesis_id, location, message, data=None):
        import json, time, os
        try:
            log_path = "/app/config/debug.log"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a") as f:
                f.write(json.dumps({"hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
        except: pass
    # #endregion
    
    async def _run(self) -> None:
        logger.info("RemoteLLMStream._run() iniziato")
        # #region agent log - H1: entry point
        self._debug_log("H1", "main.py:RemoteLLMStream._run:entry", "Inizio _run()", {"chat_ctx_items": len(list(self._chat_ctx.items))})
        # #endregion
        
        # Estrai l'ultimo messaggio utente dal contesto
        user_message = ""
        # #region agent log - H9: debug chat_ctx
        import json as _json
        try:
            _log_path = "/app/config/debug.log"
            ctx_items = list(self._chat_ctx.items)
            ctx_debug = []
            for msg in ctx_items:
                ctx_debug.append({"role": getattr(msg, 'role', 'unknown'), "content_type": type(getattr(msg, 'content', None)).__name__, "content_preview": str(getattr(msg, 'content', ''))[:100]})
            with open(_log_path, "a") as _f:
                _f.write(_json.dumps({"hypothesisId": "H9", "location": "RemoteLLMStream:chat_ctx", "message": "Contenuto chat_ctx", "data": {"num_items": len(ctx_items), "items": ctx_debug}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
        except: pass
        # #endregion
        
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
        
        # #region agent log - H2: messaggio estratto
        self._debug_log("H2", "main.py:RemoteLLMStream._run:user_msg", "Messaggio utente estratto", {"user_message": user_message[:100], "length": len(user_message)})
        # #endregion
        
        logger.info(f"RemoteLLM: invio messaggio al server remoto: {user_message[:50]}...")
        
        try:
            # Chiama il server remoto
            # #region agent log - H2: pre-call
            self._debug_log("H2", "main.py:RemoteLLMStream._run:pre_call", "Prima della chiamata al server remoto", {"server_url": self._llm._server_url})
            # #endregion
            response = await self._llm._remote_llm.chat(user_message)
            
            # #region agent log - H3: post-call
            self._debug_log("H3", "main.py:RemoteLLMStream._run:post_call", "Risposta ricevuta", {"has_text": bool(response.text), "text_len": len(response.text) if response.text else 0})
            # #endregion
            
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
            
            # ⏱️ LATENCY: Tempo dalla fine domanda all'inizio risposta
            latency_ms = 0
            if _last_stt_end_time:
                latency_ms = (t_tts_end - _last_stt_end_time) * 1000
                logger.info(f"⚡ [LATENCY] Domanda→Risposta: {latency_ms:.0f}ms")
            
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
                    if latency_ms > 0:
                        asyncio.create_task(send_timing_to_server("latency", {
                            "e2e_ms": int(latency_ms),
                            "to_first_audio_ms": int(tts_time_ms)
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
        # #region debug log - synthesize entry
        debug_log("A", "main.py:304", "VibeVoiceLiveKit.synthesize() ENTRY", {"text": text[:50], "text_length": len(text), "language": self.language})
        # #endregion
        # Se auto_language è attivo, usa la lingua rilevata
        current_lang = self.get_current_language()
        if current_lang != self.language:
            logger.info(f"🎤 [VibeVoice] Cambio lingua: {self.language} → {current_lang}")
            self.language = current_lang
        
        stream = VibeVoiceTTSStream(self, text, conn_options)
        # #region debug log - synthesize exit
        debug_log("A", "main.py:311", "VibeVoiceLiveKit.synthesize() EXIT", {"returned_stream": type(stream).__name__})
        # #endregion
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
        
        # #region debug log - TTS stream start
        debug_log("E", "main.py:303", "VibeVoiceTTSStream._run() ENTRY", {"text_length": len(self._text), "has_output_emitter": output_emitter is not None, "text_preview": self._text[:50]})
        # #endregion
        
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
                # #region debug log - check server
                debug_log("B", "main.py:320", "PRIMA _check_server()", {})
                # #endregion
                server_available = await self._tts_instance._check_server()
                # #region debug log - server check result
                debug_log("B", "main.py:320", "DOPO _check_server()", {"server_available": server_available})
                # #endregion
                
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
                        
                        # #region debug log - TTS server request
                        debug_log("B", "main.py:334", "PRIMA POST a TTS server", {"url": f"{self._tts_instance.tts_server_url}/synthesize", "payload_text": payload["text"]})
                        # #endregion
                        async with session.post(
                            f"{self._tts_instance.tts_server_url}/synthesize",
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=60)
                        ) as resp:
                            # #region debug log - TTS server response
                            debug_log("B", "main.py:339", "DOPO POST a TTS server", {"status": resp.status, "headers": dict(resp.headers)})
                            # #endregion
                            if resp.status == 200:
                                pcm_data = await resp.read()
                                engine = resp.headers.get("X-Engine", "unknown")
                                # #region debug log - audio received
                                debug_log("B", "main.py:342", "Audio ricevuto da TTS server", {"pcm_size": len(pcm_data) if pcm_data else 0, "engine": engine})
                                # #endregion
                                logger.info(f"🎤 [VibeVoice] Sintesi via TTS Server (engine={engine})")
                            else:
                                error = await resp.text()
                                # #region debug log - TTS server error
                                debug_log("B", "main.py:345", "ERRORE TTS server", {"status": resp.status, "error": error})
                                # #endregion
                                raise Exception(f"TTS Server error: {error}")
                else:
                    raise Exception("TTS Server non disponibile")
                    
            except Exception as e:
                # #region debug log - fallback
                debug_log("B", "main.py:349", "TTS Server fallback a Edge", {"error": str(e), "type": type(e).__name__})
                # #endregion
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
                # #region debug log - audio ready
                debug_log("C", "main.py:385", "PCM audio pronto per emissione", {"pcm_size": len(pcm_data), "samples": len(pcm_data) // 2})
                # #endregion
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
                    # #region debug log - emit via output_emitter
                    debug_log("C", "main.py:400", "PRIMA output_emitter.initialize()", {"request_id": req_id})
                    # #endregion
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
                    # #region debug log - emit completed
                    debug_log("C", "main.py:410", "DOPO output_emitter.end_input() - audio emesso", {})
                    # #endregion
                else:
                    # #region debug log - emit via event_ch
                    debug_log("C", "main.py:412", "PRIMA self._event_ch.send()", {"request_id": req_id})
                    # #endregion
                    audio_event = tts.SynthesizedAudio(
                        frame=frame,
                        request_id=req_id,
                        is_final=True
                    )
                    await self._event_ch.send(audio_event)
                    # #region debug log - event sent
                    debug_log("C", "main.py:417", "DOPO self._event_ch.send() - audio event inviato", {})
                    # #endregion
            else:
                # #region debug log - no audio
                debug_log("C", "main.py:384", "Nessun PCM audio generato", {})
                # #endregion
                    
        except Exception as e:
            # #region debug log - exception
            debug_log("D", "main.py:420", "ECCEZIONE in VibeVoiceTTSStream._run()", {"error": str(e), "type": type(e).__name__})
            # #endregion
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
        # #region debug log - EdgeTTS synthesize entry
        debug_log("A", "main.py:529", "EdgeTTS.synthesize() ENTRY", {"text": text[:50], "text_length": len(text), "voice": self.voice})
        # #endregion
        # Se auto_language è attivo, usa la lingua rilevata globalmente
        if self.auto_language:
            current_voice = self.get_voice_for_language(_detected_language)
            if current_voice != self.voice:
                logger.info(f"🔊 [TTS] Cambio voce: {self.voice} → {current_voice} (lingua: {_detected_language})")
                self.voice = current_voice
        
        stream = EdgeTTSStream(self, text, conn_options)
        # #region debug log - EdgeTTS synthesize exit
        debug_log("A", "main.py:537", "EdgeTTS.synthesize() EXIT", {"returned_stream": type(stream).__name__})
        # #endregion
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
        
        # #region debug log - EdgeTTS stream start
        debug_log("E", "main.py:548", "EdgeTTSStream._run() ENTRY", {"text_length": len(self._text), "has_output_emitter": output_emitter is not None, "text_preview": self._text[:50]})
        # #endregion
        
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
                    # #region debug log - audio emission
                    debug_log("C", "main.py:603", "PRIMA emissione audio EdgeTTS", {"has_output_emitter": output_emitter is not None, "has_event_ch": hasattr(self, '_event_ch'), "pcm_size": len(pcm_data)})
                    # #endregion
                    if output_emitter is not None:
                        # API 1.3.x
                        # #region debug log - output_emitter path
                        debug_log("C", "main.py:605", "Usando output_emitter path", {"request_id": req_id})
                        # #endregion
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
                        # #region debug log - output_emitter completed
                        debug_log("C", "main.py:615", "output_emitter.end_input() completato", {})
                        # #endregion
                    else:
                        # API 1.0.x
                        # #region debug log - event_ch path
                        debug_log("C", "main.py:617", "Usando _event_ch path", {"request_id": req_id})
                        # #endregion
                        audio_event = tts.SynthesizedAudio(
                            frame=frame,
                            request_id=req_id,
                            is_final=True
                        )
                        await self._event_ch.send(audio_event)
                        # #region debug log - event_ch sent
                        debug_log("C", "main.py:623", "_event_ch.send() completato", {})
                        # #endregion
                    
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
            # #region debug log - EdgeTTS exception
            debug_log("D", "main.py:638", "ECCEZIONE in EdgeTTSStream._run()", {"error": str(e), "type": type(e).__name__})
            # #endregion
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
                        return text
                    else:
                        error_text = await response.text()
                        logger.warning(f"Whisper server error: {response.status} - {error_text[:100]}")
                        return ""
        except Exception as e:
            logger.error(f"Errore transcribe_only: {e}")
            return ""
    
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
        
        # #region agent log
        debug_log("H4-STT", "main.py:_recognize_impl", "STT completato", {"text": text[:100] if text else "", "lang": detected_lang, "time_ms": int(stt_time_ms)})
        # #endregion
        
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
            # #region debug log - hypothesis D
            debug_log("D", "main.py:892", "ChatterboxLiveKit.synthesize chiamato", {"text_preview": text[:50], "tts_type": "ChatterboxLiveKit"})
            # #endregion
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
                stream=False
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
                    ]
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
        # Voice Activation defaults
        "wake_timeout_seconds": "20",
        "vad_energy_threshold": "40",
        "speech_energy_threshold": "100",
        "silence_threshold": "30",
        "tts_cooldown_seconds": "5",
    }
    
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Carica settings
            async with session.get(
                "https://host.docker.internal:8443/api/settings",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    settings.update(data)
                    logger.info(f"📥 Settings caricati da database")
            
            # Carica prompt
            async with session.get(
                "https://host.docker.internal:8443/api/prompt",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    settings["system_prompt"] = data.get("prompt", "")
            
            # Carica context
            async with session.get(
                "https://host.docker.internal:8443/api/context",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    settings["context_injection"] = data.get("context", "")
                    
    except Exception as e:
        logger.warning(f"⚠️ Impossibile caricare settings da DB: {e}")
    
    return settings


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
        await ctx.disconnect()
        return
    
    logger.info(f"Agent connesso alla room: {ctx.room.name}")
    
    # Carica impostazioni dal database
    db_settings = await load_settings_from_server()
    logger.info(f"📥 LLM Provider: {db_settings.get('llm_provider', 'ollama')}")
    
    # Applica Voice Settings dalle impostazioni caricate
    global WAKE_TIMEOUT_SECONDS, VAD_ENERGY_THRESHOLD, SPEECH_ENERGY_THRESHOLD, SILENCE_THRESHOLD, TTS_COOLDOWN_SECONDS
    try:
        WAKE_TIMEOUT_SECONDS = int(db_settings.get('wake_timeout_seconds', '20'))
        VAD_ENERGY_THRESHOLD = int(db_settings.get('vad_energy_threshold', '40'))
        SPEECH_ENERGY_THRESHOLD = int(db_settings.get('speech_energy_threshold', '100'))
        SILENCE_THRESHOLD = int(db_settings.get('silence_threshold', '30'))
        TTS_COOLDOWN_SECONDS = float(db_settings.get('tts_cooldown_seconds', '5'))
        logger.info(f"🎙️ Voice Settings: wake_timeout={WAKE_TIMEOUT_SECONDS}s, vad={VAD_ENERGY_THRESHOLD}, speech={SPEECH_ENERGY_THRESHOLD}, silence={SILENCE_THRESHOLD}, cooldown={TTS_COOLDOWN_SECONDS}s")
    except Exception as e:
        logger.warning(f"⚠️ Errore parsing voice settings: {e}, uso default")
    
    # Inizializza LLM in base al provider configurato
    llm_provider = db_settings.get("llm_provider", "ollama")
    
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
        
        base_llm = openai.LLM(
            model=ollama_model,
            base_url=ollama_base_url,
            api_key="ollama",  # Ollama non richiede API key
        )
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
    
    # Seleziona TTS: priorità al file, poi variabile d'ambiente
    if tts_from_file:
        tts_engine = tts_from_file.get("engine", "edge").lower()
        tts_language = tts_from_file.get("language", "it")
    else:
        tts_engine = config.tts.default_engine.lower()
        tts_language = config.tts.vibevoice_language
    
    # #region debug log - hypothesis D
    debug_log("D", "main.py:1028", "Configurazione TTS letta", {"engine": tts_engine, "language": tts_language, "from_file": bool(tts_from_file), "file_content": tts_from_file})
    # #endregion
    
    logger.info(f"🔊 ======================================")
    logger.info(f"🔊 CONFIGURAZIONE TTS")
    logger.info(f"🔊 Engine selezionato: {tts_engine}")
    logger.info(f"🔊 Lingua: {tts_language}")
    logger.info(f"🔊 Fonte config: {'file' if tts_from_file else 'env'}")
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
            # #region debug log - hypothesis A
            debug_log("A", "main.py:1098", "Chatterbox TTS creato con successo", {"tts_type": type(my_tts).__name__, "model": chatterbox_model, "language": chatterbox_language})
            # #endregion
        except Exception as e:
            logger.error(f"❌ Errore configurazione Chatterbox: {e}")
            import traceback
            traceback.print_exc()
            # #region debug log - hypothesis A
            debug_log("A", "main.py:1100", "Chatterbox fallito, fallback a EdgeTTS", {"error": str(e), "error_type": type(e).__name__})
            # #endregion
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
    else:
        # Default: Edge TTS
        my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
        logger.info(f"🔊 TTS attivo: EdgeTTS")
        logger.info(f"🔊 Voce: {config.tts.edge_voice}")
        logger.info(f"🔊 Auto-language: attivo (cambio voce automatico)")
    
    logger.info(f"🔊 ======================================")
    
    # #region debug log - hypothesis B
    debug_log("B", "main.py:1111", "TTS finale prima di creare Agent", {"tts_type": type(my_tts).__name__, "tts_module": type(my_tts).__module__, "tts_str": str(my_tts)[:100]})
    # #endregion
    
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
    default_prompt = """Sei Sophy, assistente vocale ultra-veloce. PRIORITÀ ASSOLUTA: VELOCITÀ E SINTESI.

ATTIVAZIONE:
IMPORTANTE: Rispondi SOLO quando vieni menzionata esplicitamente con "@sophyai", "@sophy", "Sophy" o varianti simili.
Se il messaggio NON contiene il tuo nome o una menzione diretta a te, NON rispondere affatto.
Quando sei menzionata, rispondi in modo utile e conciso.

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

STILE:
- Rispondi come un amico esperto: diretto, chiaro, utile
- Se non sai qualcosa, dillo in 5 parole
- Preferisci risposte secche e precise

FORMATO TTS:
- NO simboli: * # @ € $ % & / | < > { } [ ] ~ ^ `
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
    
    # #region debug log - hypothesis C
    debug_log("C", "main.py:1151", "Agent creato, verifico TTS passato", {"agent_tts_type": type(agent.tts).__name__ if hasattr(agent, 'tts') else "no_tts_attr"})
    # #endregion
    
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
    logger.info("🎤 [VAD] Monitor inizializzato e avviato")
    
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
            MIN_SPEECH_FRAMES = 10  # ~500ms di speech prima di trascrivere
            # NOTA: SILENCE_THRESHOLD è ora globale e configurabile da database
            
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
                
                # #region agent log - H10/H14: stato TTS flag (SEMPRE se attivo, ogni 50 frame altrimenti)
                should_log = should_discard or (speech_frames % 50 == 0)
                if should_log:
                    import json as _json, os as _os
                    try:
                        with open("/app/config/debug.log", "a") as _f:
                            _f.write(_json.dumps({"hypothesisId": "H10", "location": "process_participant_audio:loop", "message": "Audio frame check", "data": {"tts_active": tts_active, "tts_cooldown": tts_cooldown, "should_discard": should_discard, "energy": round(energy, 1), "buffer_size": len(audio_buffer), "frame": speech_frames, "pid": _os.getpid()}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
                    except: pass
                # #endregion
                
                # ==================== SCARTA AUDIO DURANTE TTS E COOLDOWN ====================
                # Se il TTS è attivo o siamo in cooldown, SCARTA TUTTO l'audio in ingresso
                if should_discard:
                    # #region agent log - H6: TTS attivo
                    import json as _json
                    reason = "TTS_ATTIVO" if tts_active else "COOLDOWN"
                    try:
                        with open("/app/config/debug.log", "a") as _f:
                            _f.write(_json.dumps({"hypothesisId": "H6", "location": "process_participant_audio:discard", "message": f"SCARTO AUDIO - {reason}", "data": {"energy": round(energy, 1), "tts_active": tts_active, "tts_cooldown": tts_cooldown}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
                    except: pass
                    # #endregion
                    
                    # Se c'è voce significativa durante TTS (non cooldown), interrompi
                    if tts_active and energy > 50:  # Soglia abbassata per maggiore sensibilità
                        # #region agent log - H6: tentativo interrupt
                        try:
                            with open("/app/config/debug.log", "a") as _f:
                                _f.write(_json.dumps({"hypothesisId": "H6", "location": "process_participant_audio:interrupt_attempt", "message": "TENTATIVO INTERRUPT", "data": {"energy": round(energy, 1)}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
                        except: pass
                        # #endregion
                        
                        logger.info(f"✋ [BARGE-IN] Voce durante TTS (energia: {energy:.0f}) - INTERRUPT")
                        try:
                            await session.interrupt()
                            set_tts_speaking(False)  # Reset flag dopo interrupt
                            # #region agent log - H6: interrupt success
                            try:
                                with open("/app/config/debug.log", "a") as _f:
                                    _f.write(_json.dumps({"hypothesisId": "H6", "location": "process_participant_audio:interrupt_success", "message": "INTERRUPT RIUSCITO", "data": {}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
                            except: pass
                            # #endregion
                            logger.info(f"✋ [BARGE-IN] TTS interrotto")
                        except Exception as e:
                            # #region agent log - H6: interrupt failed
                            try:
                                with open("/app/config/debug.log", "a") as _f:
                                    _f.write(_json.dumps({"hypothesisId": "H6", "location": "process_participant_audio:interrupt_failed", "message": "INTERRUPT FALLITO", "data": {"error": str(e)}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
                            except: pass
                            # #endregion
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
                        
                        if len(audio_bytes) > 3200:  # Almeno 100ms di audio
                            logger.info(f"🎤 [MULTI-AUDIO] {participant_identity}: {len(audio_bytes)} bytes da trascrivere")
                            
                            # Trascrivi con WhisperSTT (metodo dedicato senza invio automatico)
                            try:
                                text = await my_stt.transcribe_only(audio_bytes, 16000)
                                
                                if text and len(text) > 1:
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
                                    
                                    # Verifica se l'agent deve rispondere (con participant_id per wake sessions)
                                    should_respond, cleaned_text, is_wake = should_agent_respond(text, participant_identity)
                                    
                                    if is_wake:
                                        # Wake word rilevato - pronuncia "Dimmi"
                                        logger.info(f"🎤 [MULTI-AUDIO] Wake word per {participant_identity} - pronuncio Dimmi")
                                        await handle_wake_word_detected(participant_identity)
                                        continue  # Non processare oltre
                                    
                                    if should_respond and cleaned_text:
                                        # Wake session attiva, rispondi
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
                # Per agent usa sempre "SophyAI", per user usa il sender passato
                sender_name = "SophyAI" if role == "assistant" else sender
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

        except Exception as e:
            logger.error(f"Errore parsing messaggio frontend: {e}")

    async def _speak_tts_task(session: AgentSession, text: str):
        """Task helper per eseguire TTS senza bloccare il loop audio"""
        try:
            await session.say(text, allow_interruptions=True)
        finally:
            set_tts_speaking(False)

    async def handle_text_message(session: AgentSession, user_text: str, send_callback, sender_identity: str = None):
        """Gestisce un messaggio testuale - broadcast a tutti, chiama LLM solo se menzionato con @sophyai"""
        # Genera ID univoco per il messaggio utente
        user_message_id = generate_message_id()
        
        # SEMPRE invia il messaggio utente a tutti i partecipanti (broadcast)
        logger.info(f"💬 Broadcast messaggio utente: {user_text[:50]}... (sender={sender_identity})")
        await send_callback(user_text, "user", user_message_id, sender_identity)
        
        # Verifica se l'agent deve rispondere (cerca @sophyai, wake word, o sessione attiva)
        should_respond, cleaned_text, is_wake = should_agent_respond(user_text, sender_identity or "text_user")
        
        if is_wake:
            logger.info(f"🎤 Wake session attivata da messaggio testuale di {sender_identity}")
        
        if not should_respond:
            # L'agent non è stato menzionato e non c'è wake session attiva
            logger.info(f"💬 Messaggio senza menzione/wake, non rispondo: {user_text[:50]}...")
            return
        
        # Genera ID univoco per la risposta dell'agent
        text_response_id = generate_message_id()
        
        try:
            logger.info(f"💬 Elaboro messaggio testuale (menzionato @sophyai): {cleaned_text} (response_id={text_response_id})")
            
            # Nota: l'analisi video è ora gestita via function calling dall'LLM
            # Non serve più pattern matching manuale
            
            # Crea chat context con il messaggio utente (usa il testo pulito senza il trigger)
            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="system", content=agent._instructions)
            chat_ctx.add_message(role="user", content=cleaned_text)
            
            # Chiama LLM
            t_start = time.time()
            response_text = ""
            llm_cancelled = False
            reset_cancel_llm()  # Reset flag prima di iniziare
            stream = my_llm.chat(chat_ctx=chat_ctx)
            
            async for chunk in stream:
                # Check cancellazione ad ogni chunk
                if should_cancel_llm():
                    logger.info("🛑 Risposta LLM ANNULLATA (utente ha interrotto)")
                    llm_cancelled = True
                    break
                    
                if hasattr(chunk, 'delta') and chunk.delta:
                    # chunk.delta è un ChoiceDelta, il testo è in .content
                    content = chunk.delta.content if hasattr(chunk.delta, 'content') else str(chunk.delta)
                    if content:
                        response_text += content
            
            # Se cancellato, non fare TTS
            if llm_cancelled:
                logger.info("🛑 TTS saltato - risposta annullata")
                return
            
            t_llm = time.time()
            logger.info(f"🤖 [LLM] Risposta in {(t_llm - t_start)*1000:.0f}ms: {response_text[:100]}...")
            
            # Invia risposta al frontend con ID
            await send_callback(response_text, "assistant", text_response_id)
            
            # Pronuncia la risposta con tracking TTS in un task separato
            set_tts_speaking(True)
            asyncio.create_task(_speak_tts_task(session, response_text))
            
        except Exception as e:
            set_tts_speaking(False)  # Reset in caso di errore
            logger.error(f"Errore gestione messaggio testuale: {e}")
    
    async def handle_agent_response_only(session: AgentSession, user_text: str, send_callback, sender_identity: str = None):
        """Gestisce solo la risposta dell'agent (messaggio user già inviato da multi-audio)"""
        # Il testo è già pulito dal trigger/wake word, usa direttamente
        cleaned_text = user_text
        
        # Genera ID univoco per la risposta dell'agent
        text_response_id = generate_message_id()
        
        try:
            logger.info(f"🤖 [MULTI-AUDIO] Genero risposta per {sender_identity}: {cleaned_text[:50]}...")
            
            # Crea chat context
            chat_ctx = llm.ChatContext()
            chat_ctx.add_message(role="system", content=agent._instructions)
            chat_ctx.add_message(role="user", content=cleaned_text)
            
            # Chiama LLM
            t_start = time.time()
            response_text = ""
            llm_cancelled = False
            reset_cancel_llm()  # Reset flag prima di iniziare
            stream = my_llm.chat(chat_ctx=chat_ctx)
            
            async for chunk in stream:
                # ==================== CHECK CANCELLAZIONE ====================
                if should_cancel_llm():
                    logger.info("🛑 [MULTI-AUDIO] Risposta LLM ANNULLATA (utente ha interrotto)")
                    llm_cancelled = True
                    break
                
                # Debug log rimosso per fix indentazione
                
                # Prova prima formato choices (RemoteLLMStream)
                if hasattr(chunk, 'choices') and chunk.choices:
                    for choice in chunk.choices:
                        if hasattr(choice, 'content') and choice.content:
                            response_text += choice.content
                # Poi prova formato delta (OpenAI-style) - controlla anche delta.choices
                elif hasattr(chunk, 'delta') and chunk.delta:
                    d = chunk.delta
                    # Prova delta.content direttamente
                    if hasattr(d, 'content') and d.content:
                        response_text += d.content
                    # Prova delta.choices[0].delta.content (formato OpenAI)
                    elif hasattr(d, 'choices') and d.choices:
                        for choice in d.choices:
                            if hasattr(choice, 'delta') and hasattr(choice.delta, 'content') and choice.delta.content:
                                response_text += choice.delta.content
                            elif hasattr(choice, 'content') and choice.content:
                                response_text += choice.content
            
            # Se cancellato, non procedere con TTS
            if llm_cancelled:
                logger.info("🛑 [MULTI-AUDIO] TTS saltato - risposta annullata")
                return
            
            t_llm = time.time()
            # #region agent log - H7: risposta finale
            try:
                _log_path = "/app/config/debug.log"
                with open(_log_path, "a") as _f:
                    _f.write(_json.dumps({"hypothesisId": "H7", "location": "handle_agent_response_only:response", "message": "Risposta LLM accumulata", "data": {"response_len": len(response_text), "response_preview": response_text[:100] if response_text else "EMPTY"}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
            except: pass
            # #endregion
            logger.info(f"🤖 [LLM] Risposta in {(t_llm - t_start)*1000:.0f}ms: {response_text[:100]}...")
            
            # Invia risposta al frontend
            # #region agent log - H8: send_callback
            try:
                _log_path = "/app/config/debug.log"
                with open(_log_path, "a") as _f:
                    _f.write(_json.dumps({"hypothesisId": "H8", "location": "handle_agent_response_only:send_callback", "message": "Invio a frontend", "data": {"text_len": len(response_text), "text_preview": response_text[:100] if response_text else "EMPTY", "msg_id": text_response_id}, "timestamp": int(time.time()*1000), "sessionId": "debug-session"}) + "\n")
            except: pass
            # #endregion
            await send_callback(response_text, "assistant", text_response_id)
            
            # Pronuncia la risposta (pulisci testo per TTS)
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
            
            async def speak_and_reset():
                try:
                    await session.say(tts_text, allow_interruptions=True)
                finally:
                    set_tts_speaking(False)
            
            # Avvia TTS in task separato - NON attendiamo per permettere al loop audio di continuare
            asyncio.create_task(speak_and_reset())
            
        except Exception as e:
            set_tts_speaking(False)  # Reset in caso di errore
            logger.error(f"Errore gestione risposta agent: {e}")
    
    # Messaggio di benvenuto
    # #region debug log - welcome message
    debug_log("A", "main.py:991", "PRIMA di session.say() benvenuto", {"text": "Ciao! Come posso aiutarti?", "session_ready": True})
    try:
        set_tts_speaking(True)
        await session.say("Ciao! Come posso aiutarti?")
        set_tts_speaking(False)
        debug_log("A", "main.py:991", "DOPO session.say() benvenuto - completato senza eccezioni", {})
    except Exception as e:
        set_tts_speaking(False)
        debug_log("A", "main.py:991", "ERRORE in session.say() benvenuto", {"error": str(e), "type": type(e).__name__})
        logger.error(f"❌ Errore nel messaggio di benvenuto: {e}")
        raise
    # #endregion
    
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
        api_key=config.livekit.api_key,
        api_secret=config.livekit.api_secret,
        ws_url=config.livekit.url,
    )
    
    cli.run_app(worker_options)


if __name__ == "__main__":
    main()
