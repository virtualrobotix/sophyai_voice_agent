"""
Main Voice Agent
Agent principale che orchestra STT, LLM e TTS per conversazioni vocali.
"""

import asyncio
import os
import sys
import time
import uuid
from typing import Optional

import json
import aiohttp
from loguru import logger

# #region debug logging
DEBUG_LOG_PATH = "/Users/robertonavoni/Desktop/Lavoro/Progetti-2025/Progetti Software/livekit-test/.cursor/debug.log"
def debug_log(hypothesis_id, location, message, data=None):
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "run1",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data or {},
            "timestamp": int(time.time() * 1000)
        }
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
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
from livekit.plugins import silero, openai
from livekit import rtc

from .config import config

# Callback globale per inviare messaggi al frontend
_send_transcript_callback = None
_sent_messages = set()  # Per evitare duplicati
_last_user_message = ""  # Per evitare duplicati STT
_detected_language = "it"  # Lingua rilevata da Whisper (default italiano)
_last_stt_end_time = None  # Timestamp fine STT per calcolo latenza

def set_transcript_callback(callback):
    global _send_transcript_callback, _sent_messages
    _send_transcript_callback = callback
    _sent_messages.clear()  # Reset quando si connette

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


async def send_transcript(text: str, role: str):
    """Invia trascrizione al frontend (con deduplicazione)"""
    global _sent_messages, _last_user_message
    
    if not text or not text.strip():
        return
        
    # Crea chiave univoca per deduplicazione
    msg_key = f"{role}:{text.strip()}"
    
    logger.info(f"📨 send_transcript chiamato: role={role}, text='{text[:40]}...', key_in_set={msg_key in _sent_messages}, set_size={len(_sent_messages)}")
    
    # Per messaggi utente, controlla anche similarità
    if role == "user":
        if text.strip() == _last_user_message:
            logger.warning(f"⚠️ DUPLICATO USER IGNORATO: {text[:30]}...")
            return
        _last_user_message = text.strip()
    
    # Evita duplicati esatti
    if msg_key in _sent_messages:
        logger.warning(f"⚠️ DUPLICATO IGNORATO: {text[:30]}...")
        return
    
    _sent_messages.add(msg_key)
    logger.info(f"✅ Messaggio aggiunto al set: '{text[:30]}...' (set_size={len(_sent_messages)})")
    
    # Limita dimensione del set (evita memory leak)
    if len(_sent_messages) > 100:
        logger.info("🗑️ Set messaggi troppo grande, reset...")
        _sent_messages.clear()
    
    if _send_transcript_callback:
        try:
            await _send_transcript_callback(text, role)
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
                        if hasattr(c, 'text'):
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
            
            # Invia transcript
            asyncio.create_task(send_transcript(self._text, "assistant"))
            
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
            
            # Invia transcript
            asyncio.create_task(send_transcript(self._text, "assistant"))
            
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
            
            # Invia risposta agent al frontend
            asyncio.create_task(send_transcript(self._text, "assistant"))
            
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
        
        # Invia trascrizione utente al frontend
        if text:
            asyncio.create_task(send_transcript(text, "user"))
        
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


async def entrypoint(ctx: JobContext):
    """Entry point per l'agent LiveKit"""
    await ctx.connect()
    
    logger.info(f"Agent connesso alla room: {ctx.room.name}")
    
    # Inizializza componenti - Usa plugin OpenAI con endpoint Ollama
    # Ollama è compatibile con l'API OpenAI su /v1/
    ollama_base_url = config.ollama.host + "/v1"
    base_llm = openai.LLM(
        model=config.ollama.model,
        base_url=ollama_base_url,
        api_key="ollama",  # Ollama non richiede API key
    )
    
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
            logger.info(f"🤖 [LLM] Inizio richiesta a {config.ollama.model}...")
            stream = self._wrapped.chat(**kwargs)
            return TimedLLMStream(stream, t_start)
    
    my_llm = TimedLLM(base_llm)
    logger.info(f"Usando OpenAI plugin con Ollama: {ollama_base_url}, model={config.ollama.model}")
    
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
    
    my_stt = WhisperSTT(
        model_size=config.whisper.model,
        language=config.whisper.language,
        auto_detect=False  # Forza lingua italiana per evitare rilevamenti errati
    )
    
    # VAD
    vad = silero.VAD.load()
    
    logger.info("Componenti caricati, creo Agent...")
    
    # Crea l'agent con le istruzioni - risposte ULTRA-BREVI e ottimizzate per velocità
    # Include rilevamento automatico della lingua
    agent = Agent(
        instructions="""Sei Sophy, assistente vocale ultra-veloce. PRIORITÀ ASSOLUTA: VELOCITÀ E SINTESI.

REGOLE FONDAMENTALI:
1. RISPOSTE ULTRA-BREVI: massimo 1-2 frasi, mai più di 30 parole
2. VAI DRITTO AL PUNTO: niente preamboli, saluti inutili o ripetizioni
3. LINGUA: rispondi nella stessa lingua dell'utente

STILE:
- Rispondi come un amico esperto: diretto, chiaro, utile
- Se non sai qualcosa, dillo in 5 parole
- Preferisci risposte secche e precise

FORMATO TTS:
- NO simboli: * # @ € $ % & / | < > { } [ ] ~ ^ `
- NO emoji
- Numeri in parole (ventitre, non 23)
- NO elenchi puntati, scrivi discorsivo""",
        vad=vad,
        stt=my_stt,
        llm=my_llm,
        tts=my_tts,
    )
    
    # #region debug log - hypothesis C
    debug_log("C", "main.py:1151", "Agent creato, verifico TTS passato", {"agent_tts_type": type(agent.tts).__name__ if hasattr(agent, 'tts') else "no_tts_attr"})
    # #endregion
    
    logger.info("Agent creato, creo AgentSession...")
    
    # Crea sessione
    session = AgentSession()
    
    logger.info("AgentSession creata, avvio...")
    
    # Avvia la sessione
    await session.start(agent, room=ctx.room)
    
    logger.info("AgentSession avviata!")
    
    # Imposta callback per inviare trascrizioni al frontend
    async def send_to_frontend(text: str, role: str):
        """Invia trascrizione al frontend via data channel"""
        try:
            data = json.dumps({"type": "transcript", "text": text, "role": role})
            await ctx.room.local_participant.publish_data(data.encode(), reliable=True)
            logger.info(f"📤 [FRONTEND] {role}: {text[:50]}...")
        except Exception as e:
            logger.error(f"Errore invio al frontend: {e}")
    
    set_transcript_callback(send_to_frontend)
    
    # Handler per messaggi dal frontend (es. interrupt, text_message)
    @ctx.room.on("data_received")
    def on_data_received(data: rtc.DataPacket):
        try:
            msg = json.loads(data.data.decode())
            msg_type = msg.get("type")
            
            if msg_type == "interrupt":
                logger.info("✋ Richiesta interruzione dal frontend")
                asyncio.create_task(session.interrupt())
            
            elif msg_type == "text_message":
                # Messaggio testuale dall'utente
                text = msg.get("text", "").strip()
                if text:
                    logger.info(f"📝 Messaggio testuale ricevuto: {text}")
                    # Processa come se fosse stato detto vocalmente
                    asyncio.create_task(handle_text_message(session, text, send_to_frontend))
                    
        except Exception as e:
            logger.error(f"Errore parsing messaggio frontend: {e}")
    
    async def handle_text_message(session: AgentSession, user_text: str, send_callback):
        """Gestisce un messaggio testuale - chiama LLM e pronuncia risposta"""
        try:
            logger.info(f"💬 Elaboro messaggio testuale: {user_text}")
            
            # Crea chat context con il messaggio utente
            chat_ctx = llm.ChatContext()
            chat_ctx.append(text=agent._instructions, role="system")
            chat_ctx.append(text=user_text, role="user")
            
            # Chiama LLM
            t_start = time.time()
            response_text = ""
            stream = my_llm.chat(chat_ctx=chat_ctx)
            
            async for chunk in stream:
                if hasattr(chunk, 'delta') and chunk.delta:
                    response_text += chunk.delta
            
            t_llm = time.time()
            logger.info(f"🤖 [LLM] Risposta in {(t_llm - t_start)*1000:.0f}ms: {response_text[:100]}...")
            
            # Invia risposta al frontend
            await send_callback(response_text, "assistant")
            
            # Pronuncia la risposta
            await session.say(response_text, allow_interruptions=True)
            
        except Exception as e:
            logger.error(f"Errore gestione messaggio testuale: {e}")
    
    # Messaggio di benvenuto
    # #region debug log - welcome message
    debug_log("A", "main.py:991", "PRIMA di session.say() benvenuto", {"text": "Ciao! Come posso aiutarti?", "session_ready": True})
    try:
        await session.say("Ciao! Come posso aiutarti?")
        debug_log("A", "main.py:991", "DOPO session.say() benvenuto - completato senza eccezioni", {})
    except Exception as e:
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
