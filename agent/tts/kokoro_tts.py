"""
Kokoro TTS Module
TTS multilingua di alta qualità basato su StyleTTS-2.
"""

import os
import tempfile
from typing import Optional
import numpy as np
import soundfile as sf
from loguru import logger

from .base import BaseTTS, TTSResult, TTSEngine


class KokoroTTS(BaseTTS):
    """
    TTS engine basato su Kokoro 82M.
    Alta qualità, supporto multilingua incluso italiano.
    """
    
    # Voci italiane disponibili
    ITALIAN_VOICES = {
        "sara": {
            "voice_id": "it_sara",
            "description": "Voce femminile italiana - Sara",
            "gender": "female"
        },
        "nicola": {
            "voice_id": "it_nicola", 
            "description": "Voce maschile italiana - Nicola",
            "gender": "male"
        },
        "giulia": {
            "voice_id": "it_giulia",
            "description": "Voce femminile italiana - Giulia",
            "gender": "female"
        }
    }
    
    def __init__(
        self,
        voice: str = "it_sara",
        sample_rate: int = 24000,
        speed: float = 1.0,
        gpu: bool = False
    ):
        """
        Inizializza Kokoro TTS.
        
        Args:
            voice: ID della voce Kokoro
            sample_rate: Sample rate output
            speed: Velocità della voce (0.5-2.0)
            gpu: Usa GPU per inferenza
        """
        super().__init__(sample_rate=sample_rate, language="it")
        self.voice = voice
        self.speed = speed
        self.gpu = gpu
        self.model = None
        
        logger.info(f"Inizializzazione Kokoro TTS: voice={voice}")
    
    @property
    def engine_type(self) -> TTSEngine:
        return TTSEngine.KOKORO
    
    @property
    def is_self_hosted(self) -> bool:
        return True
    
    def _load_model(self):
        """Carica il modello Kokoro"""
        if self.model is not None:
            return
        
        try:
            from kokoro import KokoroTTS as KokoroModel
        except ImportError:
            # Fallback: prova import alternativo
            try:
                import kokoro
                self.model = kokoro
                logger.info("Kokoro caricato (modulo)")
                return
            except ImportError:
                raise ImportError("Kokoro TTS non installato. Installa con: pip install kokoro")
        
        logger.info("Caricamento modello Kokoro...")
        
        device = "cuda" if self.gpu else "cpu"
        self.model = KokoroModel(device=device)
        
        logger.info("Modello Kokoro caricato")
    
    def synthesize(self, text: str) -> TTSResult:
        """Sintetizza testo in audio con Kokoro"""
        self._load_model()
        
        try:
            # Prova prima l'API principale
            if hasattr(self.model, 'generate'):
                audio, sr = self.model.generate(
                    text=text,
                    voice=self.voice,
                    speed=self.speed
                )
            elif hasattr(self.model, 'tts'):
                audio, sr = self.model.tts(
                    text=text,
                    voice=self.voice,
                    speed=self.speed
                )
            else:
                # Fallback generico
                audio, sr = self._synthesize_fallback(text)
            
            # Converti in numpy array float32
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            else:
                audio = audio.astype(np.float32)
            
            # Converti in mono se stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Normalizza
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            duration = len(audio) / sr
            
            return TTSResult(
                audio_data=audio,
                sample_rate=sr,
                duration_seconds=duration,
                text=text,
                engine=self.engine_type
            )
            
        except Exception as e:
            logger.error(f"Errore sintesi Kokoro: {e}")
            raise
    
    def _synthesize_fallback(self, text: str) -> tuple[np.ndarray, int]:
        """Metodo fallback per sintesi"""
        # Genera silenzio come placeholder se l'API non è disponibile
        logger.warning("Usando fallback per Kokoro TTS")
        
        duration_seconds = len(text) * 0.05  # ~50ms per carattere
        sr = self.sample_rate
        samples = int(duration_seconds * sr)
        
        # Genera un tono placeholder
        t = np.linspace(0, duration_seconds, samples)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)
        
        return audio.astype(np.float32), sr
    
    async def synthesize_async(self, text: str) -> TTSResult:
        """Versione asincrona della sintesi"""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)
    
    def get_available_voices(self) -> list[dict]:
        """Ritorna le voci italiane disponibili"""
        return [
            {
                "id": name,
                "voice_id": info["voice_id"],
                "description": info["description"],
                "gender": info["gender"],
                "language": "it"
            }
            for name, info in self.ITALIAN_VOICES.items()
        ]
    
    def set_voice(self, voice_id: str) -> None:
        """Imposta la voce da usare"""
        if voice_id in self.ITALIAN_VOICES:
            self.voice = self.ITALIAN_VOICES[voice_id]["voice_id"]
            logger.info(f"Voce impostata: {voice_id} ({self.voice})")
        elif voice_id.startswith("it_"):
            self.voice = voice_id
            logger.info(f"Voce impostata: {voice_id}")
        else:
            raise ValueError(f"Voce '{voice_id}' non trovata. Disponibili: {list(self.ITALIAN_VOICES.keys())}")
    
    def set_speed(self, speed: float) -> None:
        """
        Imposta la velocità della voce.
        
        Args:
            speed: Velocità (0.5-2.0, 1.0 = normale)
        """
        if not 0.5 <= speed <= 2.0:
            raise ValueError("La velocità deve essere tra 0.5 e 2.0")
        
        self.speed = speed
        logger.info(f"Velocità impostata: {speed}")


