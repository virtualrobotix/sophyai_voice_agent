"""Text-to-Speech module"""
from .base import BaseTTS, TTSResult, TTSEngine
from .edge_tts_engine import EdgeTTS

__all__ = [
    "BaseTTS",
    "TTSResult", 
    "TTSEngine",
    "EdgeTTS",
]


def get_tts_engine(engine_name: str, **kwargs) -> BaseTTS:
    """
    Factory function per ottenere il TTS engine desiderato.
    
    Args:
        engine_name: Nome del TTS engine (edge, piper, coqui, kokoro, vibevoice, chatterbox)
        **kwargs: Argomenti aggiuntivi per il TTS
        
    Returns:
        Istanza del TTS engine
    """
    engine_name = engine_name.lower()
    
    # Edge TTS è sempre disponibile
    if engine_name == "edge":
        return EdgeTTS(**kwargs)
    
    # Piper TTS (opzionale)
    if engine_name == "piper":
        try:
            from .piper_tts import PiperTTS
            return PiperTTS(**kwargs)
        except ImportError:
            print("Piper TTS non disponibile, uso Edge TTS")
            return EdgeTTS(**kwargs)
    
    # Coqui TTS (opzionale)
    if engine_name == "coqui":
        try:
            from .coqui_tts import CoquiTTS
            return CoquiTTS(**kwargs)
        except ImportError:
            print("Coqui TTS non disponibile, uso Edge TTS")
            return EdgeTTS(**kwargs)
    
    # Kokoro TTS (opzionale)
    if engine_name == "kokoro":
        try:
            from .kokoro_tts import KokoroTTS
            return KokoroTTS(**kwargs)
        except ImportError:
            print("Kokoro TTS non disponibile, uso Edge TTS")
            return EdgeTTS(**kwargs)
    
    # VibeVoice TTS (opzionale)
    if engine_name == "vibevoice":
        try:
            from .vibevoice_tts import VibeVoiceTTS
            return VibeVoiceTTS(**kwargs)
        except ImportError:
            print("VibeVoice TTS non disponibile, uso Edge TTS")
            return EdgeTTS(**kwargs)
    
    # Chatterbox TTS (opzionale)
    if engine_name == "chatterbox":
        try:
            from .chatterbox_tts import ChatterboxTTS
            return ChatterboxTTS(**kwargs)
        except ImportError:
            print("Chatterbox TTS non disponibile, uso Edge TTS")
            return EdgeTTS(**kwargs)
    
    # Default a Edge TTS
    print(f"TTS engine '{engine_name}' non riconosciuto, uso Edge TTS")
    return EdgeTTS(**kwargs)
