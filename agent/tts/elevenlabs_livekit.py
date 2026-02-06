"""
ElevenLabs TTS wrapper per LiveKit.
Chiama l'API ElevenLabs direttamente e converte l'audio in formato LiveKit-compatibile.
"""
import os
import time
import logging
import asyncio

import aiohttp
from livekit.agents import tts

logger = logging.getLogger(__name__)


class ElevenLabsLiveKit(tts.TTS):
    """
    Wrapper LiveKit-compatibile per ElevenLabs TTS API.
    Supporta text-to-speech con modelli multilingua ElevenLabs.
    """

    SUPPORTED_LANGUAGES = {
        "it": "it-IT", "en": "en-US", "zh": "zh-CN",
        "es": "es-ES", "fr": "fr-FR", "de": "de-DE"
    }

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model: str = "eleven_multilingual_v2",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        language: str = "it",
        auto_language: bool = True,
        sample_rate: int = 24000
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.language = language
        self.auto_language = auto_language
        self._sample_rate = sample_rate

        logger.info(f"ElevenLabsLiveKit init: voice={voice_id}, model={model}, language={language}")

    def synthesize(self, text: str) -> "ElevenLabsSynthesizeStream":
        return ElevenLabsSynthesizeStream(self, text)

    async def _synthesize_audio(self, text: str) -> bytes:
        """Chiama l'API ElevenLabs e ritorna PCM audio."""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }

        payload = {
            "text": text,
            "model_id": self.model,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style,
                "use_speaker_boost": True
            }
        }

        t_start = time.time()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(f"ElevenLabs API error {resp.status}: {error_text}")

                    mp3_data = await resp.read()

            t_api = time.time()

            # Converti MP3 in PCM usando ffmpeg
            pcm_data = await self._mp3_to_pcm(mp3_data)

            t_end = time.time()
            duration_audio = len(pcm_data) / (self._sample_rate * 2)
            logger.info(f"[ElevenLabs] Tempo: {(t_end-t_start)*1000:.0f}ms (API: {(t_api-t_start)*1000:.0f}ms, Convert: {(t_end-t_api)*1000:.0f}ms) | Audio: {duration_audio:.2f}s")

            return pcm_data

        except Exception as e:
            logger.error(f"ElevenLabs synthesis error: {e}")
            raise

    async def _mp3_to_pcm(self, mp3_data: bytes) -> bytes:
        """Converti MP3 in PCM 16-bit mono usando ffmpeg."""
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", "pipe:0",
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", str(self._sample_rate), "-ac", "1",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate(input=mp3_data)

        if process.returncode != 0:
            raise Exception(f"ffmpeg conversion error: {stderr.decode()[:200]}")

        return stdout


class ElevenLabsSynthesizeStream(tts.SynthesizeStream):
    """Stream di sintesi per ElevenLabs."""

    def __init__(self, tts_instance: ElevenLabsLiveKit, text: str):
        super().__init__(tts=tts_instance, input_text=text)
        self._tts = tts_instance
        self._text = text

    async def _run(self):
        """Esegue la sintesi e produce frame audio."""
        try:
            pcm_data = await self._tts._synthesize_audio(self._text)

            if not pcm_data:
                return

            # Invia audio come singolo frame
            from livekit import rtc
            frame = rtc.AudioFrame(
                data=pcm_data,
                sample_rate=self._tts._sample_rate,
                num_channels=1,
                samples_per_channel=len(pcm_data) // 2
            )

            self._event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id="",
                    frame=frame,
                )
            )

        except Exception as e:
            logger.error(f"ElevenLabs stream error: {e}")
            raise
