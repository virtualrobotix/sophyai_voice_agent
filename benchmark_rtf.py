#!/usr/bin/env python3
"""TTS Benchmark - Real-Time Factor Analysis"""

import torch
import numpy as np
import scipy.io.wavfile as wav
import time
import os
import tempfile
import subprocess

# Frase standard di Sophy
TEXT = "Ciao sono Sophy il tuo agente turistico personale come posso aiutarti?"

print("=" * 70)
print("TTS BENCHMARK - Real-Time Factor Analysis")
print("=" * 70)
print(f'Frase: "{TEXT}"')
print(f"Lunghezza testo: {len(TEXT)} caratteri")
device_name = "CUDA" if torch.cuda.is_available() else "CPU"
print(f"Device: {device_name}")
print("=" * 70)

results = []

# 1. Edge TTS (Cloud)
print("\n[1] EDGE TTS (Microsoft Cloud)")
try:
    import edge_tts
    import asyncio
    
    async def gen_edge():
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        communicate = edge_tts.Communicate(TEXT, "it-IT-IsabellaNeural")
        await communicate.save(tmp_path)
        return tmp_path
    
    t0 = time.time()
    tmp_path = asyncio.run(gen_edge())
    gen_time = time.time() - t0
    
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
         "-of", "default=noprint_wrappers=1:nokey=1", tmp_path],
        capture_output=True, text=True
    )
    audio_duration = float(result.stdout.strip())
    os.unlink(tmp_path)
    
    rtf = gen_time / audio_duration
    results.append(("Edge TTS", "-", gen_time, audio_duration, rtf, "Cloud"))
    print(f"   Generazione: {gen_time:.3f}s | Durata audio: {audio_duration:.2f}s | RTF: {rtf:.3f}")
except Exception as e:
    print(f"   ERRORE: {e}")

# 2. Qwen TTS
print("\n[2] QWEN TTS (Alibaba - GPU)")
try:
    from qwen_tts import Qwen3TTSModel
    
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="auto",
        dtype=torch.float16
    )
    load_time = time.time() - t0
    
    t0 = time.time()
    wavs, sr = model.generate_custom_voice(text=TEXT, language="Italian", speaker="ryan")
    gen_time = time.time() - t0
    
    audio = wavs[0] if isinstance(wavs, list) else wavs
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    audio_duration = len(audio.flatten()) / sr
    
    rtf = gen_time / audio_duration
    results.append(("Qwen TTS", load_time, gen_time, audio_duration, rtf, "GPU"))
    print(f"   Caricamento: {load_time:.2f}s | Generazione: {gen_time:.3f}s | Durata audio: {audio_duration:.2f}s | RTF: {rtf:.3f}")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ERRORE: {e}")

# 3. Piper TTS
print("\n[3] PIPER TTS (CPU - Veloce)")
try:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    
    t0 = time.time()
    subprocess.run(
        ["piper", "--model", "/app/models/piper/it_IT-riccardo-x_low.onnx", 
         "--output_file", tmp_path],
        input=TEXT.encode(),
        capture_output=True
    )
    gen_time = time.time() - t0
    
    sr, audio = wav.read(tmp_path)
    audio_duration = len(audio) / sr
    os.unlink(tmp_path)
    
    rtf = gen_time / audio_duration
    results.append(("Piper TTS", "-", gen_time, audio_duration, rtf, "CPU"))
    print(f"   Generazione: {gen_time:.3f}s | Durata audio: {audio_duration:.2f}s | RTF: {rtf:.3f}")
except Exception as e:
    print(f"   ERRORE: {e}")

# 4. Kokoro TTS
print("\n[4] KOKORO TTS (GPU)")
try:
    from kokoro import KPipeline
    
    t0 = time.time()
    pipeline = KPipeline(lang_code="it")
    load_time = time.time() - t0
    
    t0 = time.time()
    generator = pipeline(TEXT, voice="if_sara")
    audio_chunks = []
    for _, _, audio_chunk in generator:
        audio_chunks.append(audio_chunk)
    audio_full = np.concatenate(audio_chunks)
    gen_time = time.time() - t0
    
    audio_duration = len(audio_full) / 24000
    
    rtf = gen_time / audio_duration
    results.append(("Kokoro TTS", load_time, gen_time, audio_duration, rtf, "GPU"))
    print(f"   Caricamento: {load_time:.2f}s | Generazione: {gen_time:.3f}s | Durata audio: {audio_duration:.2f}s | RTF: {rtf:.3f}")
    del pipeline
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ERRORE: {e}")

# 5. Chatterbox TTS
print("\n[5] CHATTERBOX TTS (GPU - Voice Cloning)")
try:
    from chatterbox.tts import ChatterboxTTS
    
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained(device="cuda")
    load_time = time.time() - t0
    
    t0 = time.time()
    audio = model.generate(TEXT)
    gen_time = time.time() - t0
    
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    if audio.ndim > 1:
        audio = audio.squeeze()
    audio_duration = len(audio) / 24000
    
    rtf = gen_time / audio_duration
    results.append(("Chatterbox", load_time, gen_time, audio_duration, rtf, "GPU"))
    print(f"   Caricamento: {load_time:.2f}s | Generazione: {gen_time:.3f}s | Durata audio: {audio_duration:.2f}s | RTF: {rtf:.3f}")
    del model
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ERRORE: {e}")

# 6. Coqui TTS
print("\n[6] COQUI TTS (GPU - Italiano nativo)")
try:
    from TTS.api import TTS
    
    t0 = time.time()
    tts = TTS(model_name="tts_models/it/mai_female/glow-tts", progress_bar=False)
    tts = tts.to("cuda")
    load_time = time.time() - t0
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    
    t0 = time.time()
    tts.tts_to_file(text=TEXT, file_path=tmp_path)
    gen_time = time.time() - t0
    
    sr, audio = wav.read(tmp_path)
    audio_duration = len(audio) / sr
    os.unlink(tmp_path)
    
    rtf = gen_time / audio_duration
    results.append(("Coqui TTS", load_time, gen_time, audio_duration, rtf, "GPU"))
    print(f"   Caricamento: {load_time:.2f}s | Generazione: {gen_time:.3f}s | Durata audio: {audio_duration:.2f}s | RTF: {rtf:.3f}")
except Exception as e:
    print(f"   ERRORE: {e}")

# Riepilogo
print("\n" + "=" * 70)
print("RIEPILOGO - Real-Time Factor (RTF < 1.0 = Realtime)")
print("=" * 70)
print(f"{'Engine':<15} {'Load(s)':<10} {'Gen(s)':<10} {'Audio(s)':<10} {'RTF':<8} {'Realtime?':<10} {'Device'}")
print("-" * 70)
for name, load, gen, dur, rtf, device in sorted(results, key=lambda x: x[4]):
    load_str = f"{load:.2f}" if isinstance(load, float) else str(load)
    realtime = "✅ SI" if rtf < 1.0 else "❌ NO"
    print(f"{name:<15} {load_str:<10} {gen:<10.3f} {dur:<10.2f} {rtf:<8.3f} {realtime:<10} {device}")
print("=" * 70)
print("\n📊 NOTA: RTF (Real-Time Factor) = Tempo generazione / Durata audio")
print("   RTF < 1.0 = TTS genera audio più veloce del realtime")
print("   RTF = 0.5 = genera 2x più veloce del realtime")
print("   RTF = 0.1 = genera 10x più veloce del realtime")
