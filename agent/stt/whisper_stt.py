"""
Whisper STT Module
Utilizza faster-whisper per la trascrizione audio in tempo reale.
"""

import os
import numpy as np
from typing import Optional, AsyncGenerator
from dataclasses import dataclass
from faster_whisper import WhisperModel
from loguru import logger


@dataclass
class TranscriptionResult:
    """Risultato della trascrizione"""
    text: str
    language: str
    confidence: float
    is_final: bool


class WhisperSTT:
    """
    Speech-to-Text engine basato su Whisper.
    Utilizza faster-whisper per performance ottimali.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        language: str = "it",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Inizializza il modello Whisper.
        
        Args:
            model_size: Dimensione del modello (tiny, base, small, medium, large)
            language: Codice lingua (it per italiano)
            device: Device per inferenza (cpu, cuda)
            compute_type: Tipo di computazione (int8, float16, float32)
        """
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None
        
        logger.info(f"Inizializzazione Whisper STT: model={model_size}, lang={language}, device={device}")
    
    def load_model(self) -> None:
        """Carica il modello Whisper"""
        if self.model is not None:
            return
            
        logger.info(f"Caricamento modello Whisper {self.model_size}...")
        
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        
        logger.info("Modello Whisper caricato con successo")
    
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Trascrive audio in testo.
        
        Args:
            audio_data: Array numpy con i dati audio (float32, mono)
            sample_rate: Sample rate dell'audio (default 16000)
            
        Returns:
            TranscriptionResult con il testo trascritto
        """
        if self.model is None:
            self.load_model()
        
        # Assicura che l'audio sia nel formato corretto
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalizza l'audio se necessario
        if np.abs(audio_data).max() > 1.0:
            audio_data = audio_data / np.abs(audio_data).max()
        
        # Trascrizione
        segments, info = self.model.transcribe(
            audio_data,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200
            )
        )
        
        # Combina tutti i segmenti
        full_text = " ".join([segment.text.strip() for segment in segments])
        
        return TranscriptionResult(
            text=full_text,
            language=info.language,
            confidence=info.language_probability,
            is_final=True
        )
    
    async def transcribe_stream(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        chunk_duration_ms: int = 1000
    ) -> AsyncGenerator[TranscriptionResult, None]:
        """
        Trascrive uno stream audio in tempo reale.
        
        Args:
            audio_stream: Generator asincrono di chunk audio
            chunk_duration_ms: Durata di ogni chunk in millisecondi
            
        Yields:
            TranscriptionResult per ogni chunk processato
        """
        if self.model is None:
            self.load_model()
        
        buffer = np.array([], dtype=np.float32)
        sample_rate = 16000
        samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
        
        async for chunk in audio_stream:
            # Aggiungi chunk al buffer
            buffer = np.concatenate([buffer, chunk.astype(np.float32)])
            
            # Processa quando abbiamo abbastanza dati
            if len(buffer) >= samples_per_chunk:
                result = self.transcribe(buffer, sample_rate)
                
                if result.text.strip():
                    yield result
                
                # Reset buffer (con overlap per continuità)
                overlap = int(samples_per_chunk * 0.1)
                buffer = buffer[-overlap:]
    
    def get_supported_languages(self) -> list[str]:
        """Ritorna la lista delle lingue supportate"""
        return [
            "it",  # Italiano
            "en",  # Inglese
            "es",  # Spagnolo
            "fr",  # Francese
            "de",  # Tedesco
            "pt",  # Portoghese
            "nl",  # Olandese
            "pl",  # Polacco
            "ru",  # Russo
            "zh",  # Cinese
            "ja",  # Giapponese
            "ko",  # Coreano
        ]




