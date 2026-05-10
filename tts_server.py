#!/usr/bin/env python3.10
"""
TTS Server - Server locale per Text-to-Speech con VibeVoice
Simile a whisper_server.py, gira sul Mac host con accesso a MPS/GPU

Uso:
    python tts_server.py --port 8092

Il Docker agent chiamerà questo server per la sintesi vocale.
"""

import os
import sys
import time
import io
import json
import argparse
import logging
from typing import Optional
import numpy as np

from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="TTS Server", description="Local TTS server with VibeVoice support")

# Stato globale del TTS
_tts_engine = None
_tts_type = None
_device = None
_ALL_ENGINES = ["edge", "coqui", "piper", "kokoro", "vibevoice", "qwen", "chatterbox"]


def _parse_allowed_engines() -> list[str]:
    raw = os.getenv("TTS_ALLOWED_ENGINES", "").strip()
    if not raw:
        return list(_ALL_ENGINES)
    allowed = [x.strip().lower() for x in raw.split(",") if x.strip()]
    # Mantiene ordine definito in _ALL_ENGINES
    return [e for e in _ALL_ENGINES if e in set(allowed)]


def _bootstrap_engine(engine: str) -> bool:
    eng = (engine or "").strip().lower()
    if eng == "vibevoice":
        return load_vibevoice()
    if eng == "edge":
        return load_edge_tts()
    # Altri engine sono on-demand
    return True


class TTSRequest(BaseModel):
    """Request per sintesi TTS"""
    text: str
    language: str = "it"
    speaker: str = "ryan"  # Default compatibile con Qwen TTS
    speed: float = 1.0
    engine: str = "edge"  # edge, qwen, piper, kokoro, vibevoice, chatterbox
    # Parametri Chatterbox
    model: str = None  # "standard" o "multilingual"
    device: str = None  # "auto", "cuda", "cpu", "mps"
    exaggeration: float = None  # 0.0-1.0
    audio_prompt_path: str = None  # Path per voice cloning


class TTSStatus(BaseModel):
    """Stato del TTS server"""
    status: str
    engine: str
    device: str
    model_loaded: bool
    available_engines: list


def load_vibevoice():
    """Carica VibeVoice TTS"""
    global _tts_engine, _tts_type, _device
    
    try:
        import torch
        from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        
        # Determina device
        if torch.cuda.is_available():
            _device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
            logger.warning("⚠️ VibeVoice su CPU sarà lento!")
        
        model_id = "microsoft/VibeVoice-Realtime-0.5B"
        logger.info(f"🎤 Caricamento VibeVoice: {model_id} su {_device}...")
        
        # Carica processor
        processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
        
        # Carica modello (usa float32 per compatibilità con voice presets)
        model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            attn_implementation='sdpa'
        )
        
        if _device != "cpu":
            model.to(_device)
        
        model.eval()
        model.set_ddpm_inference_steps(num_steps=5)
        
        # Carica voice presets
        voices = {}
        voices_dir = os.path.expanduser("~/.cache/vibevoice/voices")
        
        # Prova a trovare i voice presets
        possible_paths = [
            voices_dir,
            "./vibevoice_voices",
            os.path.join(os.path.dirname(__file__), "vibevoice_voices"),
        ]
        
        # Cerca anche nel repo clonato
        try:
            import vibevoice
            vv_path = os.path.dirname(vibevoice.__file__)
            possible_paths.append(os.path.join(os.path.dirname(vv_path), "demo/voices/streaming_model"))
        except:
            pass
        
        # Cerca nel repo locale
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_paths.append(os.path.join(script_dir, "vibevoice_repo/demo/voices/streaming_model"))
        
        for vdir in possible_paths:
            if os.path.exists(vdir):
                import glob
                for pt_file in glob.glob(os.path.join(vdir, "**/*.pt"), recursive=True):
                    name = os.path.splitext(os.path.basename(pt_file))[0].lower()
                    voices[name] = torch.load(pt_file, map_location=_device, weights_only=False)
                    logger.info(f"  Voce caricata: {name}")
                break
        
        if not voices:
            logger.warning("⚠️ Nessun voice preset trovato, sintesi potrebbe fallire")
        
        _tts_engine = {
            "model": model,
            "processor": processor,
            "voices": voices,
            "device": _device
        }
        _tts_type = "vibevoice"
        
        logger.info(f"✅ VibeVoice caricato su {_device} con {len(voices)} voci")
        return True
        
    except ImportError as e:
        logger.error(f"❌ VibeVoice non installato: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Errore caricamento VibeVoice: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_edge_tts():
    """Carica Edge TTS (fallback)"""
    global _tts_engine, _tts_type, _device
    
    try:
        import edge_tts
        _tts_engine = {"type": "edge"}
        _tts_type = "edge"
        _device = "cloud"
        logger.info("✅ Edge TTS disponibile (cloud)")
        return True
    except ImportError:
        logger.error("❌ edge-tts non installato")
        return False


@app.on_event("startup")
async def startup():
    """Carica il TTS engine all'avvio"""
    logger.info("🚀 Avvio TTS Server...")
    allowed = _parse_allowed_engines()
    boot_engine = os.getenv("TTS_BOOT_ENGINE", "").strip().lower()

    if boot_engine and boot_engine not in allowed:
        logger.warning(f"⚠️ TTS_BOOT_ENGINE '{boot_engine}' non in TTS_ALLOWED_ENGINES={allowed}")
        boot_engine = ""

    # Se non specificato, preferisci edge per startup rapido/stabile.
    if not boot_engine:
        boot_engine = "edge" if "edge" in allowed else (allowed[0] if allowed else "")

    if boot_engine:
        ok = _bootstrap_engine(boot_engine)
        if ok:
            logger.info(f"✅ Bootstrap engine caricato: {boot_engine}")
        else:
            logger.warning(f"⚠️ Bootstrap engine non caricabile: {boot_engine}")
    else:
        logger.warning("⚠️ Nessun engine consentito configurato (TTS_ALLOWED_ENGINES vuoto)")


@app.get("/")
async def root():
    return {"status": "ok", "service": "TTS Server"}


@app.get("/health")
async def health():
    return {
        "status": "healthy" if _tts_engine else "no_engine",
        "engine": _tts_type,
        "device": _device
    }


@app.get("/status", response_model=TTSStatus)
async def status():
    """Ritorna lo stato del TTS server"""
    allowed = _parse_allowed_engines()
    return TTSStatus(
        status="ready" if _tts_engine else "not_ready",
        engine=_tts_type or "none",
        device=_device or "none",
        model_loaded=_tts_engine is not None,
        available_engines=allowed
    )


@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """
    Sintetizza testo in audio.
    
    Ritorna audio PCM 16-bit mono a 24kHz.
    """
    global _tts_engine, _tts_type
    
    allowed = _parse_allowed_engines()
    if request.engine.lower() not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Engine '{request.engine}' non consentito su questo server. Allowed: {allowed}",
        )

    # Engine on-demand che non richiedono bootstrap globale.
    on_demand_engines = {"chatterbox", "coqui", "piper", "kokoro", "qwen", "vibevoice"}
    if request.engine not in on_demand_engines and not _tts_engine:
        raise HTTPException(status_code=503, detail="TTS engine non caricato")
    
    t_start = time.time()
    text = request.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Testo vuoto")
    
    # Usa engine dalla request se specificato, altrimenti quello globale
    engine_to_use = request.engine if request.engine else _tts_type
    # #region agent log
    try:
        _lp = os.environ.get("DEBUG_TTS_LOG", "/app/.cursor/debug-fac0c1.log")
        _cuda_n = 0
        _cuda_ok = False
        try:
            import torch as _torch
            _cuda_ok = _torch.cuda.is_available()
            _cuda_n = _torch.cuda.device_count() if _cuda_ok else 0
        except Exception:
            pass
        _payload = {
            "sessionId": "fac0c1",
            "hypothesisId": "H-GPU",
            "location": "tts_server.py:synthesize",
            "message": "synth_entry",
            "data": {
                "engine_requested": request.engine,
                "engine_resolved": engine_to_use,
                "global_tts_type": _tts_type,
                "cuda_available": _cuda_ok,
                "cuda_device_count": _cuda_n,
                "text_len": len(text),
            },
            "timestamp": int(time.time() * 1000),
            "runId": "tts-inproc",
        }
        if os.path.isdir(os.path.dirname(_lp)) or os.path.exists(os.path.dirname(_lp)):
            with open(_lp, "a", encoding="utf-8") as _df:
                _df.write(json.dumps(_payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    # #endregion
    logger.info(f"🎤 Sintesi: '{text[:50]}...' (engine={engine_to_use}, lang={request.language})")
    
    actual_engine_used = engine_to_use  # Tiene traccia dell'engine effettivamente usato
    try:
        if engine_to_use == "chatterbox":
            try:
                pcm_data = await synthesize_chatterbox(
                    text,
                    request.language,
                    request.model,
                    request.device,
                    request.exaggeration,
                    request.audio_prompt_path,
                )
            except Exception as e:
                # Nei test non vogliamo generare audio in fallback.
                raise HTTPException(
                    status_code=503,
                    detail=f"Engine requested '{engine_to_use}' unavailable: {e}",
                )
        elif engine_to_use == "piper":
            try:
                pcm_data = await synthesize_piper(text, request.model)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Engine requested '{engine_to_use}' unavailable: {e}",
                )
        elif engine_to_use == "coqui":
            try:
                pcm_data = await synthesize_coqui(text, request.model)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Engine requested '{engine_to_use}' unavailable: {e}",
                )
        elif engine_to_use == "kokoro":
            try:
                pcm_data = await synthesize_kokoro(text, request.language, request.speed)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Engine requested '{engine_to_use}' unavailable: {e}",
                )
        elif engine_to_use == "vibevoice":
            try:
                # Tenta caricamento on-demand se non è già attivo.
                if _tts_type != "vibevoice":
                    if not load_vibevoice():
                        raise RuntimeError("VibeVoice non installato o non caricabile")
                pcm_data = await synthesize_vibevoice(text, request.speaker, request.speed)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Engine requested '{engine_to_use}' unavailable: {e}",
                )
        elif engine_to_use == "qwen":
            try:
                pcm_data = await synthesize_qwen(text, request.language, request.speaker)
            except Exception as e:
                raise HTTPException(
                    status_code=503,
                    detail=f"Engine requested '{engine_to_use}' unavailable: {e}",
                )
        elif engine_to_use == "edge" or _tts_type == "edge":
            pcm_data = await synthesize_edge(text, request.language)
        else:
            raise HTTPException(status_code=500, detail=f"Engine TTS sconosciuto: {engine_to_use}")
        
        t_end = time.time()
        duration_audio = len(pcm_data) / (24000 * 2)  # 24kHz, 16-bit
        
        logger.info(f"✅ Sintesi completata: {(t_end-t_start)*1000:.0f}ms, audio: {duration_audio:.2f}s")
        # #region agent log
        try:
            _lp = os.environ.get("DEBUG_TTS_LOG", "/app/.cursor/debug-fac0c1.log")
            _payload = {
                "sessionId": "fac0c1",
                "hypothesisId": "H-RESULT",
                "location": "tts_server.py:synthesize",
                "message": "synth_success",
                "data": {
                    "engine_requested": request.engine,
                    "engine_actual": actual_engine_used,
                    "elapsed_ms": int((t_end - t_start) * 1000),
                    "duration_audio_s": round(duration_audio, 3),
                },
                "timestamp": int(time.time() * 1000),
                "runId": "tts-inproc",
            }
            if os.path.isdir(os.path.dirname(_lp)) or os.path.exists(os.path.dirname(_lp)):
                with open(_lp, "a", encoding="utf-8") as _df:
                    _df.write(json.dumps(_payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion
        
        return Response(
            content=pcm_data,
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": "24000",
                "X-Channels": "1",
                "X-Duration": str(duration_audio),
                "X-Engine": actual_engine_used
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Errore sintesi: {e}")
        # #region agent log
        try:
            _lp = os.environ.get("DEBUG_TTS_LOG", "/app/.cursor/debug-fac0c1.log")
            _payload = {
                "sessionId": "fac0c1",
                "hypothesisId": "H-RESULT",
                "location": "tts_server.py:synthesize",
                "message": "synth_error",
                "data": {
                    "engine_requested": request.engine,
                    "engine_resolved": engine_to_use if "engine_to_use" in locals() else None,
                    "error_type": type(e).__name__,
                    "error": str(e)[:400],
                },
                "timestamp": int(time.time() * 1000),
                "runId": "tts-inproc",
            }
            if os.path.isdir(os.path.dirname(_lp)) or os.path.exists(os.path.dirname(_lp)):
                with open(_lp, "a", encoding="utf-8") as _df:
                    _df.write(json.dumps(_payload, ensure_ascii=False) + "\n")
        except Exception:
            pass
        # #endregion
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


async def synthesize_vibevoice(text: str, speaker: str, speed: float) -> bytes:
    """Sintetizza con VibeVoice"""
    import torch
    import copy
    
    model = _tts_engine["model"]
    processor = _tts_engine["processor"]
    voices = _tts_engine["voices"]
    device = _tts_engine["device"]
    
    # Seleziona voce
    speaker_lower = speaker.lower()
    if speaker_lower not in voices:
        # Prova match parziale
        for name in voices:
            if speaker_lower in name or name in speaker_lower:
                speaker_lower = name
                break
        else:
            # Usa prima voce disponibile
            if voices:
                speaker_lower = list(voices.keys())[0]
            else:
                raise Exception("Nessuna voce disponibile")
    
    voice = voices[speaker_lower]
    logger.info(f"  Usando voce: {speaker_lower}")
    
    # Prepara input
    inputs = processor.process_input_with_cached_prompt(
        text=text,
        cached_prompt=voice,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Sposta su device
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
    
    # Calcola max_new_tokens basato sulla lunghezza del testo
    # VibeVoice genera ~7.5 tokens per secondo, ~12 caratteri = 1 secondo
    # Aggiungo margine del 50%
    text_len = len(text)
    estimated_duration = text_len / 12  # secondi stimati
    max_tokens = int(estimated_duration * 7.5 * 1.5)  # tokens con margine
    max_tokens = max(30, min(max_tokens, 500))  # tra 30 e 500 tokens
    
    logger.info(f"  Max tokens calcolati: {max_tokens} (testo: {text_len} chars, durata stimata: {estimated_duration:.1f}s)")
    
    # Genera
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            cfg_scale=1.0,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False},
            verbose=False,
            all_prefilled_outputs=copy.deepcopy(voice),
        )
    
    # Converti output
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio = outputs.speech_outputs[0].cpu().numpy()
        if len(audio.shape) > 1:
            audio = audio.squeeze()
        
        # Normalizza
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        
        # Converti a int16
        pcm_data = (audio * 32767).astype(np.int16).tobytes()
        return pcm_data
    else:
        raise Exception("Nessun output audio generato")


async def synthesize_edge(text: str, language: str) -> bytes:
    """Sintetizza con Edge TTS"""
    import edge_tts
    import subprocess
    
    # Mappa lingua a voce
    voices = {
        "it": "it-IT-DiegoNeural",
        "en": "en-US-GuyNeural",
        "es": "es-ES-AlvaroNeural",
        "fr": "fr-FR-HenriNeural",
        "de": "de-DE-ConradNeural",
        "zh": "zh-CN-YunxiNeural",
    }
    voice = voices.get(language, "it-IT-DiegoNeural")
    
    communicate = edge_tts.Communicate(text, voice)
    
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    
    if not audio_data:
        raise Exception("Nessun audio da Edge TTS")
    
    # Converti MP3 in PCM
    process = subprocess.Popen(
        ['ffmpeg', '-i', 'pipe:0', '-f', 's16le', '-ar', '24000', '-ac', '1', 'pipe:1'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    pcm_data, _ = process.communicate(audio_data)
    
    return pcm_data


async def synthesize_piper(text: str, model: str = None) -> bytes:
    """Sintetizza con Piper TTS"""
    import asyncio
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from agent.tts.piper_tts import PiperTTS
    
    # Usa modello di default se non specificato
    piper_model = model or "it_IT-riccardo-x_low"
    
    logger.info(f"🎤 Piper TTS: model={piper_model}")
    
    # Crea istanza e sintetizza
    piper = PiperTTS(model=piper_model)
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, piper.synthesize, text)
    
    # Converti audio float32 in PCM int16
    audio_float = result.audio_data
    
    # Resample a 24kHz se necessario
    if result.sample_rate != 24000:
        import scipy.signal as signal
        num_samples = int(len(audio_float) * 24000 / result.sample_rate)
        audio_float = signal.resample(audio_float, num_samples)
    
    # Normalizza e converti a int16
    audio_float = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_float * 32767).astype(np.int16)
    
    return audio_int16.tobytes()


async def synthesize_kokoro(text: str, language: str = "it", speed: float = 1.0) -> bytes:
    """Sintetizza con Kokoro TTS"""
    import asyncio
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from agent.tts.kokoro_tts import KokoroTTS
    
    # Alcune build Kokoro non espongono voci italiane dedicate.
    # Proviamo prima la voce italiana, poi un fallback compatibile.
    candidate_voices = ["it_sara", "af_bella"] if language == "it" else ["af_bella"]
    try:
        import torch as _torch
        _use_gpu = bool(_torch.cuda.is_available())
    except Exception:
        _use_gpu = False
    
    last_error = None
    result = None
    for voice in candidate_voices:
        logger.info(f"🎤 Kokoro TTS: voice={voice}, speed={speed}, gpu={_use_gpu}")
        try:
            kokoro = KokoroTTS(voice=voice, speed=speed, gpu=_use_gpu)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, kokoro.synthesize, text)
            break
        except Exception as e:
            last_error = e
            logger.warning(f"⚠️ Kokoro voice '{voice}' fallita: {e}")

    if result is None:
        raise RuntimeError(f"Kokoro synthesis failed for voices {candidate_voices}: {last_error}")
    
    # Converti audio float32 in PCM int16
    audio_float = result.audio_data
    
    # Resample a 24kHz se necessario (Kokoro già usa 24kHz)
    if result.sample_rate != 24000:
        import scipy.signal as signal
        num_samples = int(len(audio_float) * 24000 / result.sample_rate)
        audio_float = signal.resample(audio_float, num_samples)
    
    # Normalizza e converti a int16
    audio_float = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_float * 32767).astype(np.int16)
    
    return audio_int16.tobytes()


async def synthesize_chatterbox(
    text: str, 
    language: str = "it",
    model: str = None,
    device: str = None,
    exaggeration: Optional[float] = None,
    audio_prompt_path: Optional[str] = None
) -> bytes:
    """Sintetizza con Chatterbox TTS"""
    try:
        # Importa Chatterbox (deve essere installato sul Mac host)
        # Aggiungi il path del progetto per importare agent.tts
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from agent.tts.chatterbox_tts import ChatterboxTTS
        
        chatterbox = ChatterboxTTS(
            model=model or "multilingual",
            language=language,
            device=device or "auto",
            exaggeration=exaggeration,
            audio_prompt_path=audio_prompt_path
        )
        
        # Sintetizza
        result = chatterbox.synthesize(text)
        
        # Converti numpy array in PCM 16-bit
        audio_data = result.audio_data
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalizza se necessario
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Converti a int16 PCM
        pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
        
        # Se il sample rate non è 24kHz, ri-campiona (per ora assumiamo 24kHz)
        if result.sample_rate != 24000:
            # TODO: ri-campiona a 24kHz se necessario
            logger.warning(f"Chatterbox sample rate {result.sample_rate} != 24000, potrebbe essere necessario ri-campionare")
        
        return pcm_data
        
    except (ImportError, RuntimeError) as e:
        logger.warning(f"⚠️ Chatterbox TTS non disponibile ({type(e).__name__}), uso EdgeTTS come fallback: {e}")
        # Fallback a EdgeTTS
        return await synthesize_edge(text, language)
    except Exception as e:
        logger.error(f"❌ Errore sintesi Chatterbox: {e}")
        import traceback
        traceback.print_exc()
        # Se è un errore di dipendenze/runtime, prova fallback a EdgeTTS
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ["transformers", "torchvision", "circular import", "partially initialized", "extension", "nms"]):
            logger.warning(f"⚠️ Errore dipendenze Chatterbox ({type(e).__name__}), uso EdgeTTS come fallback")
            return await synthesize_edge(text, language)
        # Per altri errori, solleva eccezione
        raise HTTPException(status_code=500, detail=str(e))


async def synthesize_qwen(text: str, language: str = "it", speaker: str = "Ryan") -> bytes:
    """Sintetizza con Qwen TTS (self-hosted)"""
    import asyncio
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from qwen_tts import Qwen3TTSModel
        import torch
    except ImportError as e:
        logger.error(f"❌ qwen-tts non installato: {e}")
        raise ImportError("qwen-tts non installato. Installa con: pip install qwen-tts")
    
    # Mapping lingua
    LANGUAGE_MAP = {
        "it": "Italian",
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "pt": "Portuguese",
        "ru": "Russian"
    }
    
    qwen_language = LANGUAGE_MAP.get(language, "Italian")
    
    # Speaker validi per Qwen TTS
    QWEN_SPEAKERS = ['aiden', 'dylan', 'eric', 'ono_anna', 'ryan', 'serena', 'sohee', 'uncle_fu', 'vivian']
    # Mapping speaker alternativi
    SPEAKER_MAP = {
        'carter': 'ryan',  # fallback per VibeVoice speaker
        'default': 'ryan',
        'male': 'ryan',
        'female': 'serena',
        'italian': 'ryan',  # Ryan ha buon supporto italiano
    }
    
    # Normalizza speaker
    speaker_lower = speaker.lower()
    if speaker_lower not in QWEN_SPEAKERS:
        mapped = SPEAKER_MAP.get(speaker_lower, 'ryan')
        logger.info(f"🔄 Qwen TTS: mapping speaker '{speaker}' -> '{mapped}'")
        speaker = mapped
    
    logger.info(f"🎤 Qwen TTS: speaker={speaker}, language={qwen_language}")
    
    # Carica modello (singleton pattern - carica solo una volta)
    global _qwen_model
    if '_qwen_model' not in globals() or _qwen_model is None:
        logger.info("📥 Caricamento modello Qwen TTS...")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # sdpa: più stabile nel container; flash_attention_2 può bloccare a lungo in fase di init
        attn_impl = "sdpa"
        
        _qwen_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl
        )
        logger.info(f"✅ Qwen TTS caricato su {device}")
    
    # Sintetizza in thread pool
    def _generate():
        wavs, sr = _qwen_model.generate_custom_voice(
            text=text,
            language=qwen_language,
            speaker=speaker
        )
        return wavs[0], sr
    
    loop = asyncio.get_event_loop()
    audio_data, sample_rate = await loop.run_in_executor(None, _generate)
    
    # Converti a float32 se necessario
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Normalizza
    max_val = np.abs(audio_data).max()
    if max_val > 1.0:
        audio_data = audio_data / max_val
    
    # Resample a 24kHz se necessario
    if sample_rate != 24000:
        import scipy.signal as signal
        num_samples = int(len(audio_data) * 24000 / sample_rate)
        audio_data = signal.resample(audio_data, num_samples)
    
    # Converti a int16 PCM
    audio_data = np.clip(audio_data, -1.0, 1.0)
    pcm_data = (audio_data * 32767).astype(np.int16).tobytes()
    
    return pcm_data


async def synthesize_coqui(text: str, model: str = None) -> bytes:
    """Sintetizza con Coqui TTS (self-hosted)."""
    import asyncio

    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from agent.tts.coqui_tts import CoquiTTS

    coqui_model = model or "tts_models/it/mai_female/vits"
    try:
        import torch as _torch
        _use_gpu = bool(_torch.cuda.is_available())
    except Exception:
        _use_gpu = False

    logger.info(f"🎤 Coqui TTS: model={coqui_model}, gpu={_use_gpu}")
    coqui = CoquiTTS(model=coqui_model, gpu=_use_gpu)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, coqui.synthesize, text)

    audio_float = result.audio_data
    if result.sample_rate != 24000:
        import scipy.signal as signal
        num_samples = int(len(audio_float) * 24000 / result.sample_rate)
        audio_float = signal.resample(audio_float, num_samples)

    audio_float = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_float * 32767).astype(np.int16)
    return audio_int16.tobytes()


# Variabile globale per modello Qwen (singleton)
_qwen_model = None


@app.get("/voices")
async def get_voices():
    """Ritorna le voci disponibili"""
    if _tts_type == "vibevoice" and _tts_engine:
        return {
            "engine": "vibevoice",
            "voices": list(_tts_engine.get("voices", {}).keys())
        }
    elif _tts_type == "edge":
        return {
            "engine": "edge",
            "voices": ["it-IT-DiegoNeural", "en-US-GuyNeural", "es-ES-AlvaroNeural", 
                      "fr-FR-HenriNeural", "de-DE-ConradNeural", "zh-CN-YunxiNeural"]
        }
    return {"engine": "none", "voices": []}


def main():
    parser = argparse.ArgumentParser(description="TTS Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8092, help="Port")
    args = parser.parse_args()
    
    logger.info(f"🎤 TTS Server avviato su http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()







