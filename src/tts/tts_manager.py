import asyncio
import os
import shutil
from typing import Optional

import torch

from src.config import settings
from src.tts.edge_tts import EdgeTTSTTS

class TTSManager:
    """
    Manages Text-to-Speech synthesis, routing requests to the appropriate engine
    based on language and latency requirements.
    
    Engines:
    - Piper (Local): English, Hindi, Tamil, Telugu (preferred when models are available)
    - Edge TTS (Cloud): Kannada primary + fallback for any missing Piper voice
    """
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Path to Piper binary (prefer env or PATH)
        self.piper_path = settings.PIPER_PATH or shutil.which("piper") or "piper"

        self.models_dir = os.path.abspath(settings.PIPER_MODELS_DIR)

        self.edge_tts = None
        try:
            self.edge_tts = EdgeTTSTTS()
        except Exception as e:
            print(f"⚠️ Edge TTS init failed: {e}")
        
    def get_engine(self, language_code: str) -> str:
        backend = settings.TTS_BACKEND.lower()
        if backend in {"piper", "edge"}:
            return backend
        # Requirement: keep Kannada on Edge TTS.
        if language_code == "kn":
            return "edge"
        if language_code in ["en", "hi", "ta", "te"]:
            return "piper"
        return "edge"

    def _pick_piper_model(self, language_code: str) -> Optional[str]:
        """
        Return first available Piper model path for a language.
        """
        candidates = {
            "en": [
                "en_US-lessac-medium.onnx",
            ],
            "hi": [
                "hi_IN-rohan-medium.onnx",
            ],
            "ta": [
                "ta_IN-iitm-female-s1-medium.onnx",
                # Common/expected Tamil model names in custom Piper bundles.
                "ta_IN-kani-medium.onnx",
                "ta_IN-ponni-medium.onnx",
                "ta_IN-tamil-medium.onnx",
            ],
            "te": [
                "te_IN-maya-medium.onnx",
            ],
        }

        for model_name in candidates.get(language_code, []):
            model_path = os.path.join(self.models_dir, model_name)
            if os.path.exists(model_path):
                return model_path

        # Non-English fallback to English voice when available.
        fallback_path = os.path.join(self.models_dir, "en_US-lessac-medium.onnx")
        if os.path.exists(fallback_path):
            return fallback_path
        return None

    async def speak(self, text: str, language_code: str) -> Optional[bytes]:
        """
        Synthesize speech from text.
        """
        # engine = self.get_engine(language_code) # DISABLED LOG to reduce spam
        # print(f"🔊 [TTS] Synthesizing with {engine.upper()}: '{text[:20]}...'")
        
        engine = self.get_engine(language_code)

        if engine == "edge":
            audio = None
            if self.edge_tts:
                audio = await self._speak_edge(text, language_code)
            if audio:
                return audio
            # Edge failed: try Piper fallback once (English model first).
            return await self._speak_piper(text, "en")

        audio = await self._speak_piper(text, language_code)
        if audio:
            return audio

        # Piper missing/unavailable: fallback to Edge for non-silent responses.
        if self.edge_tts:
            return await self._speak_edge(text, language_code)
        return None
            
    async def _speak_piper(self, text: str, language_code: str) -> Optional[bytes]:
        """
        Generate audio using Piper TTS (Local).
        """
        model_path = self._pick_piper_model(language_code)
        if not model_path:
            print(f"⚠️ Piper model missing for language={language_code}.")
            return None

        try:
            # Run Piper with WAV output to stdout
            proc = await asyncio.create_subprocess_exec(
                self.piper_path,
                "--model", model_path,
                "--output_file", "-", # stdout
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate(input=text.encode())
             
            if proc.returncode != 0:
                 print(f"❌ Piper Error: {stderr.decode()}")
                 return None
                 
            return stdout
            
        except Exception as e:
            print(f"❌ Piper Exception: {e}")
            return None

    async def _speak_edge(self, text: str, language_code: str) -> Optional[bytes]:
        """
        Generate audio using Edge TTS (Cloud).
        """
        if not self.edge_tts:
            print("❌ Edge TTS not available. Check dependencies or network.")
            return None
        try:
            return await self.edge_tts.synthesize(text, language_code)
        except Exception as e:
            print(f"❌ Edge TTS Exception: {e}")
            return None
