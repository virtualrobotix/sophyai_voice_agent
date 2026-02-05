#!/usr/bin/env python3
"""
VibeVoice TTS Server - Container isolato con transformers compatibile
Porta: 8501
"""

import os
import io
import copy
import logging
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="VibeVoice TTS Server", description="Isolated VibeVoice TTS with compatible transformers")

# Stato globale
_model = None
_processor = None
_voices = {}
_device = None


class TTSRequest(BaseModel):
    text: str
    speaker: str = "it-Spk1_man"
    speed: float = 1.0


class TTSStatus(BaseModel):
    status: str
    device: str
    voices: list
    transformers_version: str


def load_model():
    """Carica il modello VibeVoice"""
    global _model, _processor, _voices, _device
    
    try:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        import transformers
        
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Determina device
        if torch.cuda.is_available():
            _device = "cuda"
            logger.info(f"CUDA disponibile: {torch.cuda.get_device_name(0)}")
        else:
            _device = "cpu"
            logger.warning("CUDA non disponibile, uso CPU (sarà lento)")
        
        model_id = "microsoft/VibeVoice-Realtime-0.5B"
        logger.info(f"Caricamento modello: {model_id}")
        
        # Carica processor
        _processor = VibeVoiceStreamingProcessor.from_pretrained(model_id)
        
        # Carica modello
        _model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            attn_implementation='sdpa'
        )
        _model.to(_device)
        _model.eval()
        
        # Carica voice presets e spostali su device
        voices_dir = "/app/voices"
        if os.path.exists(voices_dir):
            for f in os.listdir(voices_dir):
                if f.endswith(".pt"):
                    name = f[:-3]
                    try:
                        voice_data = torch.load(
                            os.path.join(voices_dir, f),
                            map_location=_device,
                            weights_only=False
                        )
                        # Sposta tutti i tensori nel dict su device
                        if isinstance(voice_data, dict):
                            for k, v in voice_data.items():
                                if torch.is_tensor(v):
                                    voice_data[k] = v.to(_device)
                        _voices[name] = voice_data
                        logger.info(f"  Caricata voce: {name}")
                    except Exception as e:
                        logger.warning(f"  Errore caricamento {name}: {e}")
        
        logger.info(f"✅ VibeVoice caricato su {_device} con {len(_voices)} voci")
        return True
        
    except Exception as e:
        logger.error(f"❌ Errore caricamento VibeVoice: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.on_event("startup")
async def startup():
    """Carica il modello all'avvio"""
    logger.info("🚀 Avvio VibeVoice Server...")
    if not load_model():
        logger.error("Impossibile caricare VibeVoice")


@app.get("/health")
async def health():
    return {"status": "healthy" if _model else "no_model", "device": _device}


@app.get("/status")
async def status():
    import transformers
    return TTSStatus(
        status="ready" if _model else "not_ready",
        device=_device or "none",
        voices=list(_voices.keys()),
        transformers_version=transformers.__version__
    )


@app.get("/voices")
async def get_voices():
    return {"voices": list(_voices.keys())}


@app.post("/synthesize")
async def synthesize(request: TTSRequest):
    """Sintetizza audio con VibeVoice"""
    if not _model:
        raise HTTPException(status_code=503, detail="Modello non caricato")
    
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Testo vuoto")
    
    speaker = request.speaker
    
    # Seleziona voce
    if speaker not in _voices:
        # Cerca match parziale
        for name in _voices:
            if speaker.lower() in name.lower() or name.lower() in speaker.lower():
                speaker = name
                break
        else:
            if _voices:
                speaker = list(_voices.keys())[0]
            else:
                raise HTTPException(status_code=400, detail="Nessuna voce disponibile")
    
    voice = _voices[speaker]
    logger.info(f"Sintesi: '{text[:50]}...' con voce {speaker}")
    
    try:
        # Prepara input
        inputs = _processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=voice,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Sposta su device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(_device)
        
        # Calcola max tokens
        text_len = len(text)
        estimated_duration = text_len / 12
        max_tokens = int(estimated_duration * 7.5 * 1.5)
        max_tokens = max(30, min(max_tokens, 500))
        
        # Crea copia dei voice outputs su device
        def deep_copy_to_device(obj, device):
            if torch.is_tensor(obj):
                return obj.clone().to(device)
            elif isinstance(obj, dict):
                return {k: deep_copy_to_device(v, device) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_copy_to_device(item, device) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(deep_copy_to_device(item, device) for item in obj)
            else:
                return copy.deepcopy(obj)
        
        voice_copy = deep_copy_to_device(voice, _device)
        
        # Genera - usa parametri direttamente senza generation_config
        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                cfg_scale=1.0,
                tokenizer=_processor.tokenizer,
                verbose=False,
                all_prefilled_outputs=voice_copy,
            )
        
        # Estrai audio
        if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
            audio = outputs.speech_outputs[0].cpu().numpy()
            if len(audio.shape) > 1:
                audio = audio.squeeze()
            
            # Normalizza
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val
            
            # Converti a int16 PCM
            pcm_data = (audio * 32767).astype(np.int16).tobytes()
            
            logger.info(f"Audio generato: {len(pcm_data)} bytes")
            
            return Response(
                content=pcm_data,
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": "24000",
                    "X-Channels": "1",
                    "X-Bits": "16"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Nessun output audio")
            
    except Exception as e:
        logger.error(f"Errore sintesi: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/synthesize_wav")
async def synthesize_wav(request: TTSRequest):
    """Sintetizza audio e ritorna WAV"""
    import scipy.io.wavfile as wav
    
    # Usa synthesize per ottenere PCM
    response = await synthesize(request)
    pcm_data = response.body
    
    # Converti a numpy
    audio = np.frombuffer(pcm_data, dtype=np.int16)
    
    # Scrivi WAV in memoria
    buffer = io.BytesIO()
    wav.write(buffer, 24000, audio)
    buffer.seek(0)
    
    return Response(
        content=buffer.read(),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=vibevoice_output.wav"}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
