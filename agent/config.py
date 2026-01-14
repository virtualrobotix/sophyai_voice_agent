"""
Configuration Module
Gestisce la configurazione dell'applicazione.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv


# Carica variabili d'ambiente
load_dotenv()


@dataclass
class LiveKitConfig:
    """Configurazione LiveKit"""
    url: str = field(default_factory=lambda: os.getenv("LIVEKIT_URL", "ws://localhost:7880"))
    api_key: str = field(default_factory=lambda: os.getenv("LIVEKIT_API_KEY", "devkey"))
    api_secret: str = field(default_factory=lambda: os.getenv("LIVEKIT_API_SECRET", "secret"))


@dataclass
class OllamaConfig:
    """Configurazione Ollama"""
    host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "devstral-small-2:latest"))


@dataclass
class WhisperConfig:
    """Configurazione Whisper STT"""
    model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    language: str = field(default_factory=lambda: os.getenv("WHISPER_LANGUAGE", "it"))
    device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))


@dataclass
class TTSConfig:
    """Configurazione TTS"""
    default_engine: str = field(default_factory=lambda: os.getenv("DEFAULT_TTS", "piper"))
    
    # Piper
    piper_model: str = field(default_factory=lambda: os.getenv("PIPER_MODEL", "it_IT-riccardo-x_low"))
    piper_speaker: int = field(default_factory=lambda: int(os.getenv("PIPER_SPEAKER", "0")))
    
    # Edge TTS
    edge_voice: str = field(default_factory=lambda: os.getenv("EDGE_VOICE", "it-IT-DiegoNeural"))
    
    # Coqui TTS
    coqui_model: str = field(default_factory=lambda: os.getenv("COQUI_MODEL", "tts_models/it/mai_female/vits"))
    
    # Kokoro TTS
    kokoro_voice: str = field(default_factory=lambda: os.getenv("KOKORO_VOICE", "it_sara"))
    
    # VibeVoice TTS (Microsoft)
    vibevoice_model: str = field(default_factory=lambda: os.getenv("VIBEVOICE_MODEL", "realtime"))
    vibevoice_language: str = field(default_factory=lambda: os.getenv("VIBEVOICE_LANGUAGE", "it"))
    vibevoice_speaker: str = field(default_factory=lambda: os.getenv("VIBEVOICE_SPEAKER", "speaker_1"))
    vibevoice_speed: float = field(default_factory=lambda: float(os.getenv("VIBEVOICE_SPEED", "1.0")))
    vibevoice_gpu: bool = field(default_factory=lambda: os.getenv("VIBEVOICE_GPU", "true").lower() in ("true", "1", "yes"))
    
    # Chatterbox TTS (Resemble AI)
    chatterbox_model: str = field(default_factory=lambda: os.getenv("CHATTERBOX_MODEL", "multilingual"))
    chatterbox_language: str = field(default_factory=lambda: os.getenv("CHATTERBOX_LANGUAGE", "it"))
    chatterbox_device: str = field(default_factory=lambda: os.getenv("CHATTERBOX_DEVICE", "auto"))
    chatterbox_exaggeration: Optional[float] = field(default_factory=lambda: float(os.getenv("CHATTERBOX_EXAGGERATION")) if os.getenv("CHATTERBOX_EXAGGERATION") else None)
    chatterbox_cfg_weight: Optional[float] = field(default_factory=lambda: float(os.getenv("CHATTERBOX_CFG_WEIGHT")) if os.getenv("CHATTERBOX_CFG_WEIGHT") else None)
    chatterbox_audio_prompt_path: Optional[str] = field(default_factory=lambda: os.getenv("CHATTERBOX_AUDIO_PROMPT_PATH"))


@dataclass
class VisionConfig:
    """Configurazione per modelli vision LLM"""
    # Abilita/disabilita analisi video
    enabled: bool = field(default_factory=lambda: os.getenv("VIDEO_ANALYSIS_ENABLED", "true").lower() in ("true", "1", "yes"))
    
    # Modelli vision per OpenRouter
    openrouter_vision_model: str = field(default_factory=lambda: os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4-vision-preview"))
    
    # Modelli vision per Ollama
    ollama_vision_model: str = field(default_factory=lambda: os.getenv("OLLAMA_VISION_MODEL", "llava"))
    
    # Rate limiting per estrazione frame (frame al secondo)
    max_frame_rate: float = field(default_factory=lambda: float(os.getenv("VIDEO_MAX_FRAME_RATE", "1.0")))


@dataclass
class ServerConfig:
    """Configurazione Server"""
    web_port: int = field(default_factory=lambda: int(os.getenv("WEB_PORT", "8080")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


@dataclass
class AppConfig:
    """Configurazione principale dell'applicazione"""
    livekit: LiveKitConfig = field(default_factory=LiveKitConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


# Istanza globale della configurazione
config = AppConfig()


def reload_config() -> AppConfig:
    """Ricarica la configurazione dall'ambiente"""
    global config
    load_dotenv()
    config = AppConfig()
    return config









