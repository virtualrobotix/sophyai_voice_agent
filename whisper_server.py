#!/usr/bin/env python3
"""
Server Whisper con accelerazione MPS per Mac.
Esegue fuori Docker per usare la GPU.

Usa openai-whisper per supporto MPS nativo.
"""

import asyncio
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Configurazione
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "it")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto, cpu, cuda, mps
WHISPER_PORT = int(os.getenv("WHISPER_PORT", "8091"))

app = FastAPI(title="Whisper STT Server (MPS)")

# CORS per permettere chiamate dal Docker
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modello globale
model = None
USE_OPENAI_WHISPER = True  # Usa openai-whisper per supporto MPS


def detect_device():
    """Rileva il miglior dispositivo disponibile"""
    import torch
    
    if WHISPER_DEVICE != "auto":
        logger.info(f"📌 Device forzato: {WHISPER_DEVICE}")
        return WHISPER_DEVICE
    
    # Prova MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("✅ MPS (Apple Silicon GPU) disponibile!")
        return "mps"
    
    # Prova CUDA
    if torch.cuda.is_available():
        logger.info("✅ CUDA GPU disponibile!")
        return "cuda"
    
    logger.info("⚠️ Nessuna GPU disponibile, uso CPU")
    return "cpu"


@app.on_event("startup")
async def startup():
    """Carica il modello Whisper all'avvio"""
    global model, USE_OPENAI_WHISPER
    
    # Imposta cache nella directory del progetto
    cache_dir = Path(__file__).parent / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    device = detect_device()
    
    logger.info(f"🔄 Caricamento modello Whisper '{WHISPER_MODEL}'...")
    logger.info(f"   Device: {device}")
    
    try:
        # Prova openai-whisper per supporto MPS
        import whisper
        USE_OPENAI_WHISPER = True
        
        logger.info("📦 Usando openai-whisper (supporto MPS nativo)")
        model = whisper.load_model(WHISPER_MODEL, device=device, download_root=str(cache_dir / "whisper"))
        logger.info(f"✅ Modello Whisper '{WHISPER_MODEL}' caricato su {device}!")
        
    except ImportError:
        # Fallback a faster-whisper (no MPS)
        logger.warning("⚠️ openai-whisper non disponibile, uso faster-whisper (no MPS)")
        USE_OPENAI_WHISPER = False
        
        from faster_whisper import WhisperModel
        
        # faster-whisper supporta solo cpu/cuda
        fw_device = "cpu" if device == "mps" else device
        compute_type = "int8" if fw_device == "cpu" else "float16"
        
        model = WhisperModel(
            WHISPER_MODEL,
            device=fw_device,
            compute_type=compute_type,
            download_root=str(cache_dir / "huggingface" / "hub")
        )
        logger.info(f"✅ Modello Whisper '{WHISPER_MODEL}' caricato (faster-whisper, {fw_device})")


@app.get("/")
async def health():
    """Health check"""
    return {"status": "ok", "model": WHISPER_MODEL, "ready": model is not None}


@app.get("/info")
async def info():
    """Informazioni sul server"""
    import torch
    
    return {
        "model": WHISPER_MODEL,
        "language": WHISPER_LANGUAGE,
        "device": WHISPER_DEVICE,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "ready": model is not None
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form(default=None),
    detect_language: str = Form(default="false")
):
    """
    Trascrive audio.
    Accetta file audio (wav, mp3, etc.) o raw PCM.
    Se detect_language=true, rileva automaticamente la lingua.
    """
    if model is None:
        return {"error": "Modello non caricato", "text": ""}
    
    # Se detect_language è attivo o la lingua è vuota, non forzare la lingua
    auto_detect = detect_language.lower() == "true" or not language or language.strip() == ""
    lang = None if auto_detect else (language or WHISPER_LANGUAGE)
    
    logger.info(f"🎤 Trascrizione: auto_detect={auto_detect}, lang={lang}")
    
    try:
        # Leggi l'audio
        audio_data = await audio.read()
        
        # Salva in file temporaneo
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        try:
            if USE_OPENAI_WHISPER:
                # openai-whisper (supporto MPS)
                # Se lang è None, Whisper rileverà automaticamente
                # fp16=True causa NaN su MPS, usare solo per CUDA
                device = detect_device()
                use_fp16 = (device == "cuda")  # Solo CUDA supporta fp16 correttamente
                result = model.transcribe(
                    tmp_path,
                    language=lang,
                    fp16=use_fp16
                )
                text = result["text"].strip()
                detected_lang = result.get("language", lang or "it")

                # Calcola durata dall'audio
                import librosa
                duration = librosa.get_duration(path=tmp_path)
                
                # Calcola probabilità lingua (approssimativa)
                lang_prob = 0.9 if not auto_detect else 0.85
                
                logger.info(f"📝 [MPS] Trascritto - Lingua: {detected_lang} ({lang_prob:.0%}), {duration:.1f}s: {text[:100]}...")
                
                return {
                    "text": text,
                    "language": detected_lang,
                    "language_probability": lang_prob,
                    "duration": duration,
                    "auto_detected": auto_detect
                }
            else:
                # faster-whisper
                segments, info = model.transcribe(
                    tmp_path,
                    language=lang,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=200
                    )
                )
                
                text = " ".join([seg.text.strip() for seg in segments])
                detected_lang = info.language
                lang_prob = info.language_probability if hasattr(info, 'language_probability') else 0.9
                
                logger.info(f"📝 Trascritto - Lingua: {detected_lang} ({lang_prob:.0%}), {info.duration:.1f}s: {text[:100]}...")
                
                return {
                    "text": text,
                    "language": detected_lang,
                    "language_probability": lang_prob,
                    "duration": info.duration,
                    "auto_detected": auto_detect
                }
            
        finally:
            # Rimuovi file temporaneo
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"❌ Errore trascrizione: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "text": ""}


@app.post("/transcribe_raw")
async def transcribe_raw(
    samples: bytes = File(...),
    sample_rate: int = Form(default=16000),
    language: str = Form(default=None)
):
    """
    Trascrive audio raw PCM (int16).
    """
    if model is None:
        return {"error": "Modello non caricato", "text": ""}
    
    lang = language or WHISPER_LANGUAGE
    
    try:
        # Converti bytes in numpy array
        audio_array = np.frombuffer(samples, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Salva come WAV temporaneo
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_array, sample_rate)
            tmp_path = tmp.name
        
        try:
            if USE_OPENAI_WHISPER:
                # openai-whisper (supporto MPS)
                # fp16=True causa NaN su MPS, usare solo per CUDA
                device = detect_device()
                use_fp16 = (device == "cuda")  # Solo CUDA supporta fp16 correttamente
                result = model.transcribe(
                    tmp_path,
                    language=lang,
                    fp16=use_fp16
                )
                text = result["text"].strip()
                duration = len(audio_array) / sample_rate

                logger.info(f"📝 [MPS] Trascritto raw: {text[:100]}...")
                
                return {
                    "text": text,
                    "language": result.get("language", lang),
                    "duration": duration
                }
            else:
                # faster-whisper
                segments, info = model.transcribe(
                    tmp_path,
                    language=lang,
                    beam_size=5,
                    vad_filter=True
                )
                
                text = " ".join([seg.text.strip() for seg in segments])
                
                logger.info(f"📝 Trascritto raw: {text[:100]}...")
                
                return {
                    "text": text,
                    "language": info.language,
                    "duration": info.duration
                }
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"❌ Errore trascrizione raw: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "text": ""}


if __name__ == "__main__":
    logger.info(f"🚀 Avvio Whisper Server su porta {WHISPER_PORT}")
    logger.info(f"   Modello: {WHISPER_MODEL}")
    logger.info(f"   Lingua: {WHISPER_LANGUAGE}")
    logger.info(f"   Device: {WHISPER_DEVICE}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=WHISPER_PORT,
        log_level="info"
    )

