import io
from typing import Optional

import av
import edge_tts
import numpy as np
import soundfile as sf

from src.config import settings


class EdgeTTSTTS:
    """
    Microsoft Edge TTS backend for Tamil/Kannada.
    Produces WAV bytes (48kHz mono PCM16) for downstream pipelines.
    """

    def __init__(
        self,
        voice_en: Optional[str] = None,
        voice_hi: Optional[str] = None,
        voice_te: Optional[str] = None,
        voice_ta: Optional[str] = None,
        voice_kn: Optional[str] = None,
        rate: Optional[str] = None,
        volume: Optional[str] = None,
        pitch: Optional[str] = None,
    ):
        self.voice_en = voice_en or settings.EDGE_TTS_VOICE_EN
        self.voice_hi = voice_hi or settings.EDGE_TTS_VOICE_HI
        self.voice_te = voice_te or settings.EDGE_TTS_VOICE_TE
        self.voice_ta = voice_ta or settings.EDGE_TTS_VOICE_TA
        self.voice_kn = voice_kn or settings.EDGE_TTS_VOICE_KN
        self.rate = rate or settings.EDGE_TTS_RATE
        self.volume = volume or settings.EDGE_TTS_VOLUME
        self.pitch = pitch or settings.EDGE_TTS_PITCH

    def _voice_for_lang(self, language_code: str) -> str:
        if language_code == "en":
            return self.voice_en
        if language_code == "hi":
            return self.voice_hi
        if language_code == "te":
            return self.voice_te
        if language_code == "kn":
            return self.voice_kn
        return self.voice_ta

    @staticmethod
    def _mp3_to_wav_bytes(mp3_bytes: bytes, target_sr: int = 48000) -> bytes:
        container = av.open(io.BytesIO(mp3_bytes))
        resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)

        pcm = bytearray()
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                # PyAV AudioPlane may not expose to_bytes() in newer versions.
                pcm += bytes(resampled.planes[0])

        if not pcm:
            raise RuntimeError("Edge TTS produced no decodable audio frames.")

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        buf = io.BytesIO()
        sf.write(buf, audio, target_sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    async def synthesize(self, text: str, language_code: str) -> bytes:
        voice = self._voice_for_lang(language_code)
        communicate = edge_tts.Communicate(
            text,
            voice=voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
        )

        audio_bytes = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                audio_bytes += chunk.get("data", b"")

        if not audio_bytes:
            raise RuntimeError("Edge TTS returned no audio data.")

        return self._mp3_to_wav_bytes(bytes(audio_bytes))
