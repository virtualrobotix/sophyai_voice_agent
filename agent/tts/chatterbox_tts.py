"""
Chatterbox TTS Module
Resemble AI Chatterbox - TTS state-of-the-art con supporto multilingua, voice cloning e emotion control.
https://github.com/resemble-ai/chatterbox
"""

import os
from typing import Optional
import numpy as np
from loguru import logger

# Monkey patch torch.load per supportare checkpoint salvati su CUDA su sistemi senza CUDA
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    """Forza map_location='cpu' se non specificato, per supportare checkpoint CUDA su CPU/MPS"""
    if 'map_location' not in kwargs:
        kwargs['map_location'] = 'cpu'
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from .base import BaseTTS, TTSResult, TTSEngine


class ChatterboxTTS(BaseTTS):
    """
    TTS engine basato su Resemble AI Chatterbox.
    
    Caratteristiche:
    - Supporto multilingua (23 lingue)
    - Voice cloning zero-shot con reference audio
    - Emotion control (exaggeration, cfg_weight)
    - Alta qualità vocale
    """
    
    # Lingue supportate (per versione multilingua)
    SUPPORTED_LANGUAGES = {
        "ar": "Arabic",
        "de": "German",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese",
        # Altre lingue supportate da Chatterbox Multilingual
    }
    
    def __init__(
        self,
        model: str = "multilingual",  # "standard" o "multilingual"
        language: str = "it",
        sample_rate: int = 24000,
        device: str = "auto",  # "auto", "cuda", "cpu"
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        audio_prompt_path: Optional[str] = None
    ):
        """
        Inizializza Chatterbox TTS.
        
        Args:
            model: Tipo di modello ("standard" o "multilingual")
            language: Codice lingua (it, en, es, fr, de, etc.)
            sample_rate: Sample rate output (default 24000 Hz)
            device: Device da usare ("auto", "cuda", "cpu")
            exaggeration: Controllo emozioni (0.0-1.0, opzionale)
            cfg_weight: Peso per classifier-free guidance (opzionale)
            audio_prompt_path: Path a file audio per voice cloning (opzionale)
        """
        super().__init__(sample_rate=sample_rate, language=language)
        
        self.model_type = model
        self.device = device
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.audio_prompt_path = audio_prompt_path
        self.model = None
        
        logger.info(f"Inizializzazione Chatterbox TTS: model={model}, language={language}, device={device}")
    
    @property
    def engine_type(self) -> TTSEngine:
        return TTSEngine.CHATTERBOX
    
    @property
    def is_self_hosted(self) -> bool:
        return True
    
    def _detect_device(self) -> str:
        """Rileva automaticamente il device migliore"""
        if self.device != "auto":
            return self.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def _load_model(self):
        """Carica il modello Chatterbox"""
        if self.model is not None:
            return
        
        try:
            device = self._detect_device()
            
            if self.model_type == "turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS
                logger.info(f"Caricamento Chatterbox Turbo su {device}...")
                self.model = ChatterboxTurboTTS.from_pretrained(device=device)
            elif self.model_type == "multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                logger.info(f"Caricamento Chatterbox Multilingual su {device}...")
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            else:
                from chatterbox.tts import ChatterboxTTS as ChatterboxStandardTTS
                logger.info(f"Caricamento Chatterbox Standard su {device}...")
                self.model = ChatterboxStandardTTS.from_pretrained(device=device)
            
            logger.info(f"Chatterbox TTS caricato (model={self.model_type}, device={device})")
            
        except ImportError:
            raise ImportError(
                "Chatterbox TTS non installato. Installa con:\n"
                "  pip install chatterbox-tts"
            )
        except Exception as e:
            logger.error(f"Errore caricamento Chatterbox TTS: {e}")
            raise
    
    def synthesize(self, text: str) -> TTSResult:
        """Sintetizza testo in audio con Chatterbox"""
        self._load_model()
        
        try:
            # Prepara parametri per generate()
            generate_kwargs = {}
            
            # Per multilingua, aggiungi language_id
            if self.model_type == "multilingual":
                generate_kwargs["language_id"] = self.language
            
            # Voice cloning
            if self.audio_prompt_path and os.path.exists(self.audio_prompt_path):
                generate_kwargs["audio_prompt_path"] = self.audio_prompt_path
                logger.debug(f"Usando voice cloning con: {self.audio_prompt_path}")
            
            # Emotion control
            if self.exaggeration is not None:
                generate_kwargs["exaggeration"] = self.exaggeration
            if self.cfg_weight is not None:
                generate_kwargs["cfg_weight"] = self.cfg_weight
            
            # Genera audio (ritorna tensor)
            wav_tensor = self.model.generate(text, **generate_kwargs)
            
            # Converti tensor in numpy array
            if hasattr(wav_tensor, 'cpu'):
                # È un tensor PyTorch
                audio_data = wav_tensor.cpu().numpy()
            elif hasattr(wav_tensor, 'numpy'):
                # È un tensor che supporta numpy()
                audio_data = wav_tensor.numpy()
            else:
                # Già un array numpy
                audio_data = np.array(wav_tensor, dtype=np.float32)
            
            # Assicurati che sia float32
            audio_data = audio_data.astype(np.float32)
            
            # Chatterbox restituisce tensor [1, samples] - rimuovi la dimensione batch
            if len(audio_data.shape) == 2 and audio_data.shape[0] == 1:
                audio_data = audio_data[0]  # Prendi il primo canale/batch
            elif len(audio_data.shape) > 1:
                # Converti in mono se stereo
                if audio_data.shape[0] > 1:  # Formato [channels, samples]
                    audio_data = audio_data.mean(axis=0)
                else:  # Formato [samples, channels]
                    audio_data = audio_data.mean(axis=1) if audio_data.shape[1] > 1 else audio_data.squeeze()
            
            # Flatten se necessario per sicurezza
            audio_data = audio_data.flatten()
            
            logger.debug(f"Audio generato: {len(audio_data)} samples, durata={len(audio_data)/24000:.2f}s")
            
            # Normalizza se necessario (Chatterbox dovrebbe già essere normalizzato)
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # Sample rate dal modello
            sr = getattr(self.model, 'sr', self.sample_rate)
            
            # Se il sample rate è diverso, possiamo risampleare (ma Chatterbox usa 24000 di default)
            if sr != self.sample_rate:
                logger.warning(f"Sample rate mismatch: modello={sr}, richiesto={self.sample_rate}")
                # Per ora usiamo il sample rate del modello
                actual_sr = sr
            else:
                actual_sr = self.sample_rate
            
            duration = len(audio_data) / actual_sr
            
            return TTSResult(
                audio_data=audio_data,
                sample_rate=actual_sr,
                duration_seconds=duration,
                text=text,
                engine=self.engine_type
            )
            
        except Exception as e:
            logger.error(f"Errore sintesi Chatterbox: {e}")
            raise
    
    async def synthesize_async(self, text: str) -> TTSResult:
        """Versione asincrona della sintesi"""
        import asyncio
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text)
    
    def get_available_voices(self) -> list[dict]:
        """Ritorna informazioni sulle voci disponibili"""
        voices = []
        
        if self.model_type == "multilingual":
            # Per multilingua, ritorna le lingue supportate
            for lang_code, lang_name in self.SUPPORTED_LANGUAGES.items():
                voices.append({
                    "id": f"multilingual_{lang_code}",
                    "language": lang_code,
                    "language_name": lang_name,
                    "model": "multilingual",
                    "description": f"Chatterbox Multilingual - {lang_name}"
                })
        else:
            # Per standard, ritorna voce predefinita
            voices.append({
                "id": "standard_default",
                "language": "en",
                "model": "standard",
                "description": "Chatterbox Standard - Default voice"
            })
        
        # Se c'è audio_prompt_path, aggiungi come opzione voice cloning
        if self.audio_prompt_path:
            voices.append({
                "id": "voice_cloned",
                "language": self.language,
                "model": "standard" if self.model_type == "standard" else "multilingual",
                "description": f"Voice cloned from {os.path.basename(self.audio_prompt_path)}",
                "audio_prompt_path": self.audio_prompt_path
            })
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """
        Imposta la voce/lingua da usare.
        
        Args:
            voice_id: ID voce (es. "multilingual_it", "standard_default") o codice lingua (es. "it")
        """
        # Se è un codice lingua semplice
        if voice_id in self.SUPPORTED_LANGUAGES:
            self.language = voice_id
            logger.info(f"Lingua impostata: {voice_id} ({self.SUPPORTED_LANGUAGES[voice_id]})")
        # Se è un ID completo
        elif voice_id.startswith("multilingual_"):
            lang_code = voice_id.replace("multilingual_", "")
            if lang_code in self.SUPPORTED_LANGUAGES:
                self.language = lang_code
                self.model_type = "multilingual"
                self.model = None  # Force reload
                logger.info(f"Voce multilingua impostata: {lang_code}")
            else:
                raise ValueError(f"Lingua '{lang_code}' non supportata")
        elif voice_id == "standard_default":
            self.model_type = "standard"
            self.model = None  # Force reload
            logger.info("Voce standard impostata")
        else:
            # Prova a usarlo come codice lingua
            if voice_id in self.SUPPORTED_LANGUAGES:
                self.language = voice_id
                logger.info(f"Lingua impostata: {voice_id}")
            else:
                logger.warning(f"Voice ID '{voice_id}' non riconosciuto, uso come lingua")
                self.language = voice_id
    
    def set_audio_prompt(self, audio_prompt_path: str) -> None:
        """
        Imposta il path per voice cloning.
        
        Args:
            audio_prompt_path: Path al file audio di riferimento
        """
        if not os.path.exists(audio_prompt_path):
            raise FileNotFoundError(f"File audio non trovato: {audio_prompt_path}")
        
        self.audio_prompt_path = audio_prompt_path
        logger.info(f"Audio prompt impostato per voice cloning: {audio_prompt_path}")
    
    def set_emotion_control(self, exaggeration: Optional[float] = None, cfg_weight: Optional[float] = None) -> None:
        """
        Imposta parametri di controllo emozioni.
        
        Args:
            exaggeration: Controllo esagerazione (0.0-1.0)
            cfg_weight: Peso classifier-free guidance
        """
        if exaggeration is not None:
            if not 0.0 <= exaggeration <= 1.0:
                raise ValueError("exaggeration deve essere tra 0.0 e 1.0")
            self.exaggeration = exaggeration
        
        if cfg_weight is not None:
            self.cfg_weight = cfg_weight
        
        logger.info(f"Emotion control: exaggeration={self.exaggeration}, cfg_weight={self.cfg_weight}")
    
    def get_info(self) -> dict:
        """Ritorna informazioni estese sul TTS engine"""
        info = super().get_info()
        info.update({
            "model_type": self.model_type,
            "device": self._detect_device(),
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
            "voice_cloning": self.audio_prompt_path is not None,
            "features": [
                "multilingual" if self.model_type == "multilingual" else "standard",
                "voice cloning" if self.audio_prompt_path else "no voice cloning",
                "emotion control" if (self.exaggeration is not None or self.cfg_weight is not None) else "default emotion"
            ]
        })
        return info






