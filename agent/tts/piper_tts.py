"""
Piper TTS Module
TTS veloce e leggero con supporto nativo italiano.
"""

import os
import subprocess
import tempfile
from typing import Optional
import numpy as np
import soundfile as sf
from loguru import logger

from .base import BaseTTS, TTSResult, TTSEngine


class PiperTTS(BaseTTS):
    """
    TTS engine basato su Piper.
    Veloce, leggero e con ottimo supporto italiano.
    """
    
    # Voci italiane disponibili per Piper
    ITALIAN_VOICES = {
        "riccardo": {
            "model": "it_IT-riccardo-x_low",
            "description": "Voce maschile italiana, qualità bassa",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx"
        },
        "paola": {
            "model": "it_IT-paola-medium", 
            "description": "Voce femminile italiana, qualità media",
            "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/paola/medium/it_IT-paola-medium.onnx"
        }
    }
    
    def __init__(
        self,
        model: str = "it_IT-riccardo-x_low",
        models_dir: str = "./models/piper",
        sample_rate: int = 22050,
        speaker: int = 0
    ):
        """
        Inizializza Piper TTS.
        
        Args:
            model: Nome del modello Piper
            models_dir: Directory per i modelli scaricati
            sample_rate: Sample rate output
            speaker: ID dello speaker (per modelli multi-speaker)
        """
        super().__init__(sample_rate=sample_rate, language="it")
        self.model = model
        self.models_dir = models_dir
        self.speaker = speaker
        self.model_path: Optional[str] = None
        
        # Crea directory modelli se non esiste
        os.makedirs(models_dir, exist_ok=True)
        
        logger.info(f"Inizializzazione Piper TTS: model={model}")
    
    @property
    def engine_type(self) -> TTSEngine:
        return TTSEngine.PIPER
    
    @property
    def is_self_hosted(self) -> bool:
        return True
    
    def _ensure_model(self) -> str:
        """Assicura che il modello sia disponibile"""
        model_path = os.path.join(self.models_dir, f"{self.model}.onnx")
        config_path = os.path.join(self.models_dir, f"{self.model}.onnx.json")
        
        if os.path.exists(model_path):
            self.model_path = model_path
            return model_path
        
        # Scarica il modello se non presente
        logger.info(f"Downloading Piper model: {self.model}")
        
        # Trova URL del modello
        for voice_info in self.ITALIAN_VOICES.values():
            if voice_info["model"] == self.model:
                model_url = voice_info["url"]
                config_url = model_url + ".json"
                break
        else:
            raise ValueError(f"Modello {self.model} non trovato nelle voci italiane disponibili")
        
        # Download con curl
        subprocess.run(["curl", "-L", "-o", model_path, model_url], check=True)
        subprocess.run(["curl", "-L", "-o", config_path, config_url], check=True)
        
        self.model_path = model_path
        logger.info(f"Modello scaricato: {model_path}")
        
        return model_path
    
    def synthesize(self, text: str) -> TTSResult:
        """Sintetizza testo in audio con Piper"""
        model_path = self._ensure_model()
        config_path = model_path + ".json"
        
        try:
            # Usa API Python di Piper
            from piper import PiperVoice
            
            voice = PiperVoice.load(model_path, config_path)
            
            # Sintetizza - restituisce generator di AudioChunk
            audio_chunks = []
            for chunk in voice.synthesize(text):
                # Usa audio_float_array per ottenere float32
                audio_chunks.append(chunk.audio_float_array)
            
            # Concatena tutti i chunk
            audio_data = np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=np.float32)
            sr = voice.config.sample_rate
            
            duration = len(audio_data) / sr
            
            logger.debug(f"Piper audio: {len(audio_data)} samples, {duration:.2f}s, sr={sr}")
            
            return TTSResult(
                audio_data=audio_data,
                sample_rate=sr,
                duration_seconds=duration,
                text=text,
                engine=self.engine_type
            )
            
        except ImportError:
            raise ImportError("Piper TTS non installato. Installa con: pip install piper-tts")
    
    async def synthesize_async(self, text: str) -> TTSResult:
        """Versione asincrona della sintesi"""
        import asyncio
        
        # Esegui in thread pool per non bloccare
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)
    
    def get_available_voices(self) -> list[dict]:
        """Ritorna le voci italiane disponibili"""
        return [
            {
                "id": name,
                "model": info["model"],
                "description": info["description"],
                "language": "it"
            }
            for name, info in self.ITALIAN_VOICES.items()
        ]
    
    def set_voice(self, voice_id: str) -> None:
        """Imposta la voce da usare"""
        if voice_id in self.ITALIAN_VOICES:
            self.model = self.ITALIAN_VOICES[voice_id]["model"]
            self.model_path = None  # Force reload
            logger.info(f"Voce impostata: {voice_id}")
        else:
            raise ValueError(f"Voce '{voice_id}' non trovata. Disponibili: {list(self.ITALIAN_VOICES.keys())}")





