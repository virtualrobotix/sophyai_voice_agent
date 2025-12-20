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
    
    # Parametri supportati per ogni engine
    EDGE_PARAMS = {'voice', 'language', 'sample_rate', 'speed'}
    PIPER_PARAMS = {'model', 'models_dir', 'sample_rate', 'speaker'}
    KOKORO_PARAMS = {'voice', 'sample_rate', 'speed', 'gpu'}
    VIBEVOICE_PARAMS = {'model', 'language', 'sample_rate', 'speaker', 'speed'}
    CHATTERBOX_PARAMS = {'model', 'language', 'sample_rate', 'device', 'exaggeration', 'cfg_weight', 'audio_prompt_path'}
    COQUI_PARAMS = {'model', 'language', 'sample_rate', 'speaker'}
    
    def filter_kwargs(allowed_params):
        return {k: v for k, v in kwargs.items() if k in allowed_params and v is not None}
    
    # Edge TTS è sempre disponibile
    if engine_name == "edge":
        return EdgeTTS(**filter_kwargs(EDGE_PARAMS))
    
    # Piper TTS (opzionale)
    if engine_name == "piper":
        try:
            from .piper_tts import PiperTTS
            return PiperTTS(**filter_kwargs(PIPER_PARAMS))
        except ImportError:
            print("Piper TTS non disponibile, uso Edge TTS")
            return EdgeTTS(**filter_kwargs(EDGE_PARAMS))
    
    # Coqui TTS (opzionale)
    if engine_name == "coqui":
        try:
            from .coqui_tts import CoquiTTS
            return CoquiTTS(**filter_kwargs(COQUI_PARAMS))
        except ImportError:
            print("Coqui TTS non disponibile, uso Edge TTS")
            return EdgeTTS(**filter_kwargs(EDGE_PARAMS))
    
    # Kokoro TTS (opzionale)
    if engine_name == "kokoro":
        try:
            from .kokoro_tts import KokoroTTS
            return KokoroTTS(**filter_kwargs(KOKORO_PARAMS))
        except ImportError:
            print("Kokoro TTS non disponibile, uso Edge TTS")
            return EdgeTTS(**filter_kwargs(EDGE_PARAMS))
    
    # VibeVoice TTS (opzionale)
    if engine_name == "vibevoice":
        try:
            from .vibevoice_tts import VibeVoiceTTS
            return VibeVoiceTTS(**filter_kwargs(VIBEVOICE_PARAMS))
        except ImportError:
            print("VibeVoice TTS non disponibile, uso Edge TTS")
            return EdgeTTS(**filter_kwargs(EDGE_PARAMS))
    
    # Chatterbox TTS (opzionale)
    if engine_name == "chatterbox":
        try:
            from .chatterbox_tts import ChatterboxTTS
            return ChatterboxTTS(**filter_kwargs(CHATTERBOX_PARAMS))
        except ImportError:
            print("Chatterbox TTS non disponibile, uso Edge TTS")
            return EdgeTTS(**filter_kwargs(EDGE_PARAMS))
    
    # Default a Edge TTS
    print(f"TTS engine '{engine_name}' non riconosciuto, uso Edge TTS")
    return EdgeTTS(**filter_kwargs(EDGE_PARAMS))
