"""
Qwen TTS Module
TTS self-hosted basato su Qwen3-TTS di Alibaba Cloud.
Supporta italiano e altre 9 lingue con alta qualità.
"""

import os
import asyncio
from typing import Optional
import numpy as np
from loguru import logger

from .base import BaseTTS, TTSResult, TTSEngine


class QwenTTS(BaseTTS):
    """
    TTS engine basato su Qwen3-TTS.
    Self-hosted con supporto CUDA per alta qualità e bassa latenza.
    """
    
    # Speaker disponibili per CustomVoice model
    AVAILABLE_SPEAKERS = {
        "vivian": {
            "name": "Vivian",
            "description": "Voce femminile giovane, brillante",
            "native_language": "Chinese",
            "gender": "female"
        },
        "serena": {
            "name": "Serena",
            "description": "Voce femminile calda, gentile",
            "native_language": "Chinese",
            "gender": "female"
        },
        "uncle_fu": {
            "name": "Uncle_Fu",
            "description": "Voce maschile matura, profonda",
            "native_language": "Chinese",
            "gender": "male"
        },
        "dylan": {
            "name": "Dylan",
            "description": "Voce maschile giovane (Pechino)",
            "native_language": "Chinese",
            "gender": "male"
        },
        "eric": {
            "name": "Eric",
            "description": "Voce maschile vivace (Sichuan)",
            "native_language": "Chinese",
            "gender": "male"
        },
        "ryan": {
            "name": "Ryan",
            "description": "Voce maschile dinamica",
            "native_language": "English",
            "gender": "male"
        },
        "aiden": {
            "name": "Aiden",
            "description": "Voce maschile solare americana",
            "native_language": "English",
            "gender": "male"
        },
        "ono_anna": {
            "name": "Ono_Anna",
            "description": "Voce femminile giapponese",
            "native_language": "Japanese",
            "gender": "female"
        },
        "sohee": {
            "name": "Sohee",
            "description": "Voce femminile coreana",
            "native_language": "Korean",
            "gender": "female"
        }
    }
    
    # Mapping lingue
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
    
    def __init__(
        self,
        model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        speaker: str = "Ryan",
        language: str = "it",
        sample_rate: int = 24000,
        instruct: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Inizializza Qwen TTS.
        
        Args:
            model: ID modello HuggingFace (1.7B o 0.6B)
            speaker: Nome speaker (Ryan, Aiden, Vivian, etc.)
            language: Codice lingua (it, en, zh, etc.)
            sample_rate: Sample rate output
            instruct: Istruzione opzionale per controllare lo stile
            device: Device per inferenza (cuda, cpu)
        """
        super().__init__(sample_rate=sample_rate, language=language)
        self.model_id = model
        self.speaker = speaker
        self.instruct = instruct or ""
        self.device = device
        self._model = None
        
        logger.info(f"Inizializzazione Qwen TTS: model={model}, speaker={speaker}, language={language}")
    
    @property
    def engine_type(self) -> TTSEngine:
        return TTSEngine.QWEN
    
    @property
    def is_self_hosted(self) -> bool:
        return True
    
    def _load_model(self):
        """Carica il modello Qwen TTS se non già caricato"""
        if self._model is not None:
            return
        
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
            
            logger.info(f"Caricamento modello Qwen TTS: {self.model_id}")
            
            # Determina dtype e attn_implementation
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            attn_impl = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
            
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_id,
                device_map=self.device,
                dtype=dtype,
                attn_implementation=attn_impl
            )
            
            logger.info(f"✅ Qwen TTS caricato su {self.device}")
            
        except ImportError as e:
            logger.error(f"❌ qwen-tts non installato: {e}")
            raise ImportError("qwen-tts non installato. Installa con: pip install qwen-tts")
        except Exception as e:
            logger.error(f"❌ Errore caricamento Qwen TTS: {e}")
            raise
    
    def synthesize(self, text: str) -> TTSResult:
        """Sintetizza testo in audio con Qwen TTS (sync)"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.synthesize_async(text))
        finally:
            loop.close()
    
    async def synthesize_async(self, text: str) -> TTSResult:
        """Sintetizza testo in audio con Qwen TTS (async)"""
        import asyncio
        
        # Esegui in thread pool per non bloccare
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._synthesize_sync, text)
        return result
    
    def _synthesize_sync(self, text: str) -> TTSResult:
        """Sintesi sincrona interna"""
        self._load_model()
        
        # Converti codice lingua in nome completo
        language = self.LANGUAGE_MAP.get(self.language, "Italian")
        
        logger.info(f"🎤 Qwen TTS: text='{text[:50]}...', speaker={self.speaker}, lang={language}")
        
        # Genera audio
        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language=language,
            speaker=self.speaker,
            instruct=self.instruct if self.instruct else None
        )
        
        # Prendi il primo risultato
        audio_data = wavs[0]
        
        # Assicurati che sia float32 normalizzato
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalizza se necessario
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        duration = len(audio_data) / sr
        
        logger.info(f"✅ Qwen TTS completato: duration={duration:.2f}s, sr={sr}")
        
        return TTSResult(
            audio_data=audio_data,
            sample_rate=sr,
            duration_seconds=duration,
            text=text,
            engine=self.engine_type
        )
    
    def get_available_voices(self) -> list[dict]:
        """Ritorna gli speaker disponibili"""
        return [
            {
                "id": key,
                "name": info["name"],
                "description": info["description"],
                "gender": info["gender"],
                "native_language": info["native_language"]
            }
            for key, info in self.AVAILABLE_SPEAKERS.items()
        ]
    
    def set_voice(self, voice_id: str) -> None:
        """Imposta lo speaker da usare"""
        voice_lower = voice_id.lower()
        if voice_lower in self.AVAILABLE_SPEAKERS:
            self.speaker = self.AVAILABLE_SPEAKERS[voice_lower]["name"]
            logger.info(f"Speaker impostato: {self.speaker}")
        else:
            # Prova a usare il nome direttamente
            self.speaker = voice_id
            logger.info(f"Speaker impostato (diretto): {self.speaker}")
    
    def set_instruct(self, instruct: str) -> None:
        """Imposta l'istruzione per controllare lo stile"""
        self.instruct = instruct
        logger.info(f"Instruct impostato: {instruct[:50]}...")
    
    @staticmethod
    def get_supported_languages() -> list[str]:
        """Ritorna le lingue supportate"""
        return list(QwenTTS.LANGUAGE_MAP.keys())
