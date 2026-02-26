import asyncio
import io
import av
from aiortc import MediaStreamTrack
import time
from fractions import Fraction

class TTSAudioTrack(MediaStreamTrack):
    """
    A MediaStreamTrack that streams audio from a queue of bytes (WAV/MP3).
    """
    kind = "audio"

    def __init__(self, loop=None):
        super().__init__()
        self.loop = loop
        self.queue = asyncio.Queue()
        self.samplerate = 48000
        self.channels = 2 # Stereo for safety, though mono is fine
        self._start_time = None
        self._timestamp = 0

    def add_audio_bytes(self, audio_bytes: bytes):
        """
        Decode encoded audio bytes (WAV/MP3) and push frames to the queue.
        This must be called from an async context or run via run_coroutine_threadsafe.
        """
        try:
            # Decode using PyAV
            container = av.open(io.BytesIO(audio_bytes))
            
            # Create a resampler to match WebRTC expectations (48kHz, Stereo, s16)
            resampler = av.AudioResampler(
                format="s16",
                layout="stereo",
                rate=self.samplerate,
            )

            for frame in container.decode(audio=0):
                # Resample frame
                for resampled_frame in resampler.resample(frame):
                    if self.loop is not None:
                        self.loop.call_soon_threadsafe(self.queue.put_nowait, resampled_frame)
                    else:
                        self.queue.put_nowait(resampled_frame)
                    
            print(f"🎵 Queue size: {self.queue.qsize()} frames")
            
        except Exception as e:
            print(f"❌ Audio Decode Error: {e}")

    async def recv(self):
        """
        Called by aiortc to get the next frame.
        """
        # Initialize start time on first packet
        if self._start_time is None:
            self._start_time = time.time()

        # Get frame from queue
        frame = await self.queue.get()
        
        # Update frame timing
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, self.samplerate)
        
        # Calculate next timestamp
        # 48000Hz * samples / 48000 = samples
        # typical frame is 960 samples (20ms)
        self._timestamp += frame.samples
        
        return frame
