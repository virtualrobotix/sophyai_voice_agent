"""
Main Voice Agent
Agent principale che orchestra STT, LLM e TTS per conversazioni vocali.
"""

import asyncio
import os
import sys
import time
import uuid

import json
import aiohttp
from loguru import logger
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
                "https://host.docker.internal:8080/api/timing",
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
        
        return VibeVoiceTTSStream(self, text, conn_options)


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
        
        return EdgeTTSStream(self, text, conn_options)


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
                    
                    logger.info(f"🔊 [TTS] Tempo totale: {total_tts_time_ms:.0f}ms (API: {download_time_ms:.0f}ms, Convert: {convert_time_ms:.0f}ms) | Audio: {audio_duration_sec:.2f}s | {len(pcm_data)} bytes")
                    
                    # Invia timing stats al server
                    asyncio.create_task(send_timing_to_server("tts", {
                        "time_ms": int(total_tts_time_ms),
                        "audio_sec": round(audio_duration_sec, 2)
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
        t_stt_end = time.time()
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
    
    # Wrapper per timing LLM
    class TimedLLM(llm.LLM):
        def __init__(self, wrapped_llm):
            super().__init__()
            self._wrapped = wrapped_llm
        
        def chat(self, **kwargs) -> llm.LLMStream:
            t_start = time.time()
            logger.info(f"🤖 [LLM] Inizio richiesta a {config.ollama.model}...")
            stream = self._wrapped.chat(**kwargs)
            
            # Wrap dello stream per tracciare la fine
            original_aiter = stream.__aiter__
            first_chunk = True
            
            async def timed_aiter():
                nonlocal first_chunk
                t_first = None
                ttfb = 0
                async for chunk in original_aiter():
                    if first_chunk:
                        t_first = time.time()
                        ttfb = (t_first - t_start) * 1000
                        logger.info(f"🤖 [LLM] Time to first token: {ttfb:.0f}ms")
                        first_chunk = False
                    yield chunk
                t_end = time.time()
                total_time = (t_end - t_start) * 1000
                logger.info(f"🤖 [LLM] Tempo totale: {total_time:.0f}ms")
                
                # Invia timing stats al server
                asyncio.create_task(send_timing_to_server("llm", {
                    "time_ms": int(total_time),
                    "ttft_ms": int(ttfb)
                }))
            
            stream.__aiter__ = timed_aiter
            return stream
    
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
            # Kokoro richiede wrapper LiveKit
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
            logger.info(f"🔊 TTS attivo: EdgeTTS (fallback da Kokoro)")
            logger.warning(f"⚠️ Kokoro non ancora integrato con LiveKit, uso EdgeTTS")
        except Exception as e:
            logger.error(f"❌ Errore configurazione Kokoro: {e}")
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
    elif tts_engine == "piper":
        try:
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
            logger.info(f"🔊 TTS attivo: EdgeTTS (fallback da Piper)")
            logger.warning(f"⚠️ Piper non ancora integrato con LiveKit, uso EdgeTTS")
        except Exception as e:
            logger.error(f"❌ Errore configurazione Piper: {e}")
            my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
    else:
        # Default: Edge TTS
        my_tts = EdgeTTS(voice=config.tts.edge_voice, auto_language=True)
        logger.info(f"🔊 TTS attivo: EdgeTTS")
        logger.info(f"🔊 Voce: {config.tts.edge_voice}")
        logger.info(f"🔊 Auto-language: attivo (cambio voce automatico)")
    
    logger.info(f"🔊 ======================================")
    
    my_stt = WhisperSTT(
        model_size=config.whisper.model,
        language=config.whisper.language
    )
    
    # VAD
    vad = silero.VAD.load()
    
    logger.info("Componenti caricati, creo Agent...")
    
    # Crea l'agent con le istruzioni - risposte BREVI e ottimizzate per TTS
    # Include rilevamento automatico della lingua
    agent = Agent(
        instructions="""Sei un assistente vocale multilingue chiamato Sophy. REGOLE IMPORTANTI:

LINGUA:
- RILEVA automaticamente la lingua in cui l'utente ti parla
- RISPONDI SEMPRE nella STESSA LINGUA dell'utente
- Se l'utente parla in italiano, rispondi in italiano
- Se l'utente parla in inglese, rispondi in inglese
- Adatta il tuo stile alla cultura della lingua

FORMATO RISPOSTA:
- Rispondi SEMPRE in modo BREVISSIMO (massimo 2-3 frasi)
- Vai dritto al punto, niente introduzioni lunghe
- Parla in modo naturale e colloquiale

REGOLE PER IL TTS (Text-to-Speech):
- NON usare MAI caratteri speciali come: * # @ € $ % & / \\ | < > { } [ ] ~ ^ ` 
- NON usare emoji o simboli
- Scrivi i numeri IN PAROLE (es: "ventitre" invece di "23", "twenty-three" in inglese)
- Evita abbreviazioni (es: "chilometri" invece di "km", "kilometers" in inglese)
- Usa la punteggiatura normale: virgole, punti, punti esclamativi e interrogativi
- Evita elenchi puntati, scrivi in modo discorsivo""",
        vad=vad,
        stt=my_stt,
        llm=my_llm,
        tts=my_tts,
    )
    
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
    await session.say("Ciao! Come posso aiutarti?")
    
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
