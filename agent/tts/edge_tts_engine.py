"""
Edge TTS Module
TTS gratuito di Microsoft Edge con eccellente qualità italiana.
Non completamente self-hosted (usa API Microsoft Edge).
"""

import os
import tempfile
import asyncio
from typing import Optional
import numpy as np
import soundfile as sf
from loguru import logger

from .base import BaseTTS, TTSResult, TTSEngine


class EdgeTTS(BaseTTS):
    """
    TTS engine basato su Microsoft Edge TTS.
    Qualità eccellente, richiede connessione internet.
    """
    
    # Voci italiane disponibili
    ITALIAN_VOICES = {
        "diego": {
            "voice": "it-IT-DiegoNeural",
            "description": "Voce maschile italiana - Diego",
            "gender": "male"
        },
        "elsa": {
            "voice": "it-IT-ElsaNeural", 
            "description": "Voce femminile italiana - Elsa",
            "gender": "female"
        },
        "isabella": {
            "voice": "it-IT-IsabellaNeural",
            "description": "Voce femminile italiana - Isabella",
            "gender": "female"
        },
        "benigno": {
            "voice": "it-IT-BenignoNeural",
            "description": "Voce maschile italiana - Benigno",
            "gender": "male"
        },
        "calimero": {
            "voice": "it-IT-CalimeroNeural",
            "description": "Voce maschile italiana - Calimero",
            "gender": "male"
        },
        "cataldo": {
            "voice": "it-IT-CataldoNeural",
            "description": "Voce maschile italiana - Cataldo",
            "gender": "male"
        },
        "fabiola": {
            "voice": "it-IT-FabiolaNeural",
            "description": "Voce femminile italiana - Fabiola",
            "gender": "female"
        },
        "fiamma": {
            "voice": "it-IT-FiammaNeural",
            "description": "Voce femminile italiana - Fiamma",
            "gender": "female"
        },
        "gianni": {
            "voice": "it-IT-GianniNeural",
            "description": "Voce maschile italiana - Gianni",
            "gender": "male"
        },
        "imelda": {
            "voice": "it-IT-ImeldaNeural",
            "description": "Voce femminile italiana - Imelda",
            "gender": "female"
        },
        "irma": {
            "voice": "it-IT-IrmaNeural",
            "description": "Voce femminile italiana - Irma",
            "gender": "female"
        },
        "lisandro": {
            "voice": "it-IT-LisandroNeural",
            "description": "Voce maschile italiana - Lisandro",
            "gender": "male"
        },
        "palmira": {
            "voice": "it-IT-PalmiraNeural",
            "description": "Voce femminile italiana - Palmira",
            "gender": "female"
        },
        "pierina": {
            "voice": "it-IT-PierinaNeural",
            "description": "Voce femminile italiana - Pierina",
            "gender": "female"
        },
        "rinaldo": {
            "voice": "it-IT-RinaldoNeural",
            "description": "Voce maschile italiana - Rinaldo",
            "gender": "male"
        }
    }
    
    def __init__(
        self,
        voice: str = "it-IT-DiegoNeural",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        sample_rate: int = 24000
    ):
        """
        Inizializza Edge TTS.
        
        Args:
            voice: Nome della voce Edge
            rate: Velocità della voce (+/-0%)
            volume: Volume della voce (+/-0%)
            pitch: Pitch della voce (+/-0Hz)
            sample_rate: Sample rate output
        """
        super().__init__(sample_rate=sample_rate, language="it")
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch
        
        logger.info(f"Inizializzazione Edge TTS: voice={voice}")
    
    @property
    def engine_type(self) -> TTSEngine:
        return TTSEngine.EDGE
    
    @property
    def is_self_hosted(self) -> bool:
        return False  # Richiede connessione a Microsoft
    
    def synthesize(self, text: str) -> TTSResult:
        """Sintetizza testo in audio con Edge TTS (sync wrapper)"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.synthesize_async(text))
        finally:
            loop.close()
    
    async def synthesize_async(self, text: str) -> TTSResult:
        """Sintetizza testo in audio con Edge TTS"""
        try:
            import edge_tts
        except ImportError:
            raise ImportError("edge-tts non installato. Installa con: pip install edge-tts")
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            output_path = tmp.name
        
        try:
            # Genera audio
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume,
                pitch=self.pitch
            )
            
            await communicate.save(output_path)
            
            # Leggi l'audio generato
            audio_data, sr = sf.read(output_path, dtype="float32")
            
            # Converti in mono se stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            duration = len(audio_data) / sr
            
            return TTSResult(
                audio_data=audio_data,
                sample_rate=sr,
                duration_seconds=duration,
                text=text,
                engine=self.engine_type
            )
            
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    async def synthesize_stream(self, text: str, chunk_size: int = 4096):
        """Sintetizza e produce audio in streaming"""
        try:
            import edge_tts
        except ImportError:
            raise ImportError("edge-tts non installato")
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch
        )
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    
    def get_available_voices(self) -> list[dict]:
        """Ritorna le voci italiane disponibili"""
        return [
            {
                "id": name,
                "voice": info["voice"],
                "description": info["description"],
                "gender": info["gender"],
                "language": "it"
            }
            for name, info in self.ITALIAN_VOICES.items()
        ]
    
    def set_voice(self, voice_id: str) -> None:
        """Imposta la voce da usare"""
        if voice_id in self.ITALIAN_VOICES:
            self.voice = self.ITALIAN_VOICES[voice_id]["voice"]
            logger.info(f"Voce impostata: {voice_id} ({self.voice})")
        elif voice_id.startswith("it-IT-"):
            # Accetta anche il nome completo della voce
            self.voice = voice_id
            logger.info(f"Voce impostata: {voice_id}")
        else:
            raise ValueError(f"Voce '{voice_id}' non trovata. Disponibili: {list(self.ITALIAN_VOICES.keys())}")
    
    def set_speech_params(
        self,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
        pitch: Optional[str] = None
    ) -> None:
        """
        Imposta parametri di velocità, volume e pitch.
        
        Args:
            rate: Velocità (es. "+10%", "-20%")
            volume: Volume (es. "+10%", "-20%")
            pitch: Pitch (es. "+50Hz", "-50Hz")
        """
        if rate is not None:
            self.rate = rate
        if volume is not None:
            self.volume = volume
        if pitch is not None:
            self.pitch = pitch
        
        logger.info(f"Parametri voce: rate={self.rate}, volume={self.volume}, pitch={self.pitch}")
    
    @staticmethod
    async def list_all_voices() -> list[dict]:
        """Lista tutte le voci disponibili in Edge TTS"""
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return [
                {
                    "name": v["Name"],
                    "short_name": v["ShortName"],
                    "gender": v["Gender"],
                    "locale": v["Locale"]
                }
                for v in voices
            ]
        except ImportError:
            return []




