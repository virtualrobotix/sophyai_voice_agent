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


class TTSRequest(BaseModel):
    """Request per sintesi TTS"""
    text: str
    language: str = "it"
    speaker: str = "carter"
    speed: float = 1.0
    engine: str = "vibevoice"  # vibevoice, edge, chatterbox
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
    
    # Prova prima VibeVoice, poi Edge TTS
    if not load_vibevoice():
        logger.info("Provo Edge TTS come fallback...")
        if not load_edge_tts():
            logger.error("❌ Nessun TTS engine disponibile!")


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
    return TTSStatus(
        status="ready" if _tts_engine else "not_ready",
        engine=_tts_type or "none",
        device=_device or "none",
        model_loaded=_tts_engine is not None,
        available_engines=["vibevoice", "edge", "chatterbox"]
    )


@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """
    Sintetizza testo in audio.
    
    Ritorna audio PCM 16-bit mono a 24kHz.
    """
    global _tts_engine, _tts_type
    
    # Chatterbox non richiede pre-caricamento, può essere usato on-demand
    if request.engine == "chatterbox":
        # Non controlla _tts_engine per Chatterbox, verrà caricato on-demand
        pass
    elif not _tts_engine:
        raise HTTPException(status_code=503, detail="TTS engine non caricato")
    
    t_start = time.time()
    text = request.text.strip()
    
    if not text:
        raise HTTPException(status_code=400, detail="Testo vuoto")
    
    # Usa engine dalla request se specificato, altrimenti quello globale
    engine_to_use = request.engine if request.engine else _tts_type
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
                    request.audio_prompt_path
                )
            except Exception as e:
                # Se Chatterbox fallisce, usa EdgeTTS come fallback
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["transformers", "torchvision", "circular import", "partially initialized", "extension", "nms", "import"]):
                    logger.warning(f"⚠️ Chatterbox non disponibile, uso EdgeTTS come fallback")
                    actual_engine_used = "edge"
                    pcm_data = await synthesize_edge(text, request.language)
                else:
                    raise
        elif engine_to_use == "piper":
            try:
                pcm_data = await synthesize_piper(text, request.model)
            except Exception as e:
                logger.warning(f"⚠️ Piper non disponibile, uso EdgeTTS come fallback: {e}")
                actual_engine_used = "edge"
                pcm_data = await synthesize_edge(text, request.language)
        elif engine_to_use == "kokoro":
            try:
                pcm_data = await synthesize_kokoro(text, request.language, request.speed)
            except Exception as e:
                logger.warning(f"⚠️ Kokoro non disponibile, uso EdgeTTS come fallback: {e}")
                actual_engine_used = "edge"
                pcm_data = await synthesize_edge(text, request.language)
        elif engine_to_use == "vibevoice" or _tts_type == "vibevoice":
            try:
                pcm_data = await synthesize_vibevoice(text, request.speaker, request.speed)
            except Exception as e:
                logger.warning(f"⚠️ VibeVoice non disponibile, uso EdgeTTS come fallback: {e}")
                actual_engine_used = "edge"
                pcm_data = await synthesize_edge(text, request.language)
        elif engine_to_use == "edge" or _tts_type == "edge":
            pcm_data = await synthesize_edge(text, request.language)
        else:
            raise HTTPException(status_code=500, detail=f"Engine TTS sconosciuto: {engine_to_use}")
        
        t_end = time.time()
        duration_audio = len(pcm_data) / (24000 * 2)  # 24kHz, 16-bit
        
        logger.info(f"✅ Sintesi completata: {(t_end-t_start)*1000:.0f}ms, audio: {duration_audio:.2f}s")
        
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
    
    # Seleziona voce in base alla lingua
    voice = "if_sara" if language == "it" else "af_bella"
    
    logger.info(f"🎤 Kokoro TTS: voice={voice}, speed={speed}")
    
    # Crea istanza e sintetizza
    kokoro = KokoroTTS(voice=voice, speed=speed)
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, kokoro.synthesize, text)
    
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
            model=model or "turbo",
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







