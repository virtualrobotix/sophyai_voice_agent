"""
Base TTS Interface
Definisce l'interfaccia comune per tutti i TTS engine.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, AsyncGenerator
import numpy as np


class TTSEngine(str, Enum):
    """Enum dei TTS engine disponibili"""
    PIPER = "piper"
    COQUI = "coqui"
    EDGE = "edge"
    KOKORO = "kokoro"
    VIBEVOICE = "vibevoice"
    CHATTERBOX = "chatterbox"


@dataclass
class TTSResult:
    """Risultato della sintesi vocale"""
    audio_data: np.ndarray
    sample_rate: int
    duration_seconds: float
    text: str
    engine: TTSEngine


class BaseTTS(ABC):
    """
    Classe base astratta per tutti i TTS engine.
    Definisce l'interfaccia comune che tutti i TTS devono implementare.
    """
    
    def __init__(self, sample_rate: int = 22050, language: str = "it"):
        """
        Inizializza il TTS engine.
        
        Args:
            sample_rate: Sample rate dell'audio in uscita
            language: Codice lingua
        """
        self.sample_rate = sample_rate
        self.language = language
    
    @property
    @abstractmethod
    def engine_type(self) -> TTSEngine:
        """Ritorna il tipo di engine TTS"""
        pass
    
    @property
    @abstractmethod
    def is_self_hosted(self) -> bool:
        """Indica se il TTS è completamente self-hosted"""
        pass
    
    @abstractmethod
    def synthesize(self, text: str) -> TTSResult:
        """
        Sintetizza testo in audio in modo sincrono.
        
        Args:
            text: Testo da sintetizzare
            
        Returns:
            TTSResult con i dati audio
        """
        pass
    
    @abstractmethod
    async def synthesize_async(self, text: str) -> TTSResult:
        """
        Sintetizza testo in audio in modo asincrono.
        
        Args:
            text: Testo da sintetizzare
            
        Returns:
            TTSResult con i dati audio
        """
        pass
    
    async def synthesize_stream(self, text: str, chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
        """
        Sintetizza testo e produce audio in streaming.
        Implementazione di default che usa synthesize_async.
        
        Args:
            text: Testo da sintetizzare
            chunk_size: Dimensione dei chunk in byte
            
        Yields:
            Chunk di dati audio
        """
        result = await self.synthesize_async(text)
        audio_bytes = result.audio_data.tobytes()
        
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i:i + chunk_size]
    
    @abstractmethod
    def get_available_voices(self) -> list[dict]:
        """
        Ritorna le voci disponibili per questo engine.
        
        Returns:
            Lista di dizionari con info sulle voci
        """
        pass
    
    @abstractmethod
    def set_voice(self, voice_id: str) -> None:
        """
        Imposta la voce da usare.
        
        Args:
            voice_id: ID della voce
        """
        pass
    
    def get_info(self) -> dict:
        """Ritorna informazioni sul TTS engine"""
        return {
            "engine": self.engine_type.value,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "self_hosted": self.is_self_hosted,
        }









