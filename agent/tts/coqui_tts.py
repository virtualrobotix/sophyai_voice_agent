"""
Coqui TTS Module
TTS di alta qualità con supporto italiano.
"""

import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger

from .base import BaseTTS, TTSResult, TTSEngine


class CoquiTTS(BaseTTS):
    """TTS engine basato su Coqui TTS."""

    ITALIAN_MODELS = {
        "mai_female_glow": {
            "model": "tts_models/it/mai_female/glow-tts",
            "vocoder": "vocoder_models/it/mai_female/vits",
            "description": "Voce femminile italiana - Glow TTS",
        },
        "mai_female_vits": {
            "model": "tts_models/it/mai_female/vits",
            "vocoder": None,
            "description": "Voce femminile italiana - VITS (end-to-end)",
        },
        "mai_male": {
            "model": "tts_models/it/mai_male/glow-tts",
            "vocoder": "vocoder_models/it/mai_male/vits",
            "description": "Voce maschile italiana - Glow TTS",
        },
    }

    def __init__(
        self,
        model: str = "tts_models/it/mai_female/vits",
        vocoder: Optional[str] = None,
        sample_rate: int = 22050,
        gpu: bool = False,
    ):
        super().__init__(sample_rate=sample_rate, language="it")
        self.model_name = model
        self.vocoder_name = vocoder
        self.gpu = gpu
        self.tts = None
        logger.info(f"Inizializzazione Coqui TTS: model={model}")

    @property
    def engine_type(self) -> TTSEngine:
        return TTSEngine.COQUI

    @property
    def is_self_hosted(self) -> bool:
        return True

    def _load_model(self):
        if self.tts is not None:
            return

        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError("Coqui TTS non installato. Installa con: pip install TTS")

        logger.info(f"Caricamento modello Coqui: {self.model_name}")
        kwargs = {
            "model_name": self.model_name,
            "progress_bar": True,
            "gpu": self.gpu,
        }
        if self.vocoder_name:
            try:
                self.tts = TTS(vocoder_name=self.vocoder_name, **kwargs)
            except TypeError:
                logger.warning(
                    "Coqui API senza vocoder_name supportato, continuo senza vocoder esplicito"
                )
                self.tts = TTS(**kwargs)
        else:
            self.tts = TTS(**kwargs)

        logger.info("Modello Coqui caricato")

    def synthesize(self, text: str) -> TTSResult:
        self._load_model()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = tmp.name

        try:
            self.tts.tts_to_file(text=text, file_path=output_path)
            audio_data, sr = sf.read(output_path, dtype="float32")
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            duration = len(audio_data) / sr
            return TTSResult(
                audio_data=audio_data,
                sample_rate=sr,
                duration_seconds=duration,
                text=text,
                engine=self.engine_type,
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    async def synthesize_async(self, text: str) -> TTSResult:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)

    def synthesize_to_array(self, text: str) -> tuple[np.ndarray, int]:
        self._load_model()
        audio = self.tts.tts(text=text)
        audio_array = np.array(audio, dtype=np.float32)
        return audio_array, self.sample_rate

    def get_available_voices(self) -> list[dict]:
        return [
            {
                "id": name,
                "model": info["model"],
                "vocoder": info["vocoder"],
                "description": info["description"],
                "language": "it",
            }
            for name, info in self.ITALIAN_MODELS.items()
        ]

    def set_voice(self, voice_id: str) -> None:
        if voice_id in self.ITALIAN_MODELS:
            model_info = self.ITALIAN_MODELS[voice_id]
            self.model_name = model_info["model"]
            self.vocoder_name = model_info["vocoder"]
            self.tts = None
            logger.info(f"Modello impostato: {voice_id}")
        else:
            raise ValueError(
                f"Modello '{voice_id}' non trovato. Disponibili: {list(self.ITALIAN_MODELS.keys())}"
            )

    @staticmethod
    def list_all_models() -> list[str]:
        try:
            from TTS.api import TTS
            return TTS.list_models()
        except ImportError:
            return []
