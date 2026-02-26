import asyncio
import io
import av
import audioop
import base64
from fractions import Fraction
from aiortc import MediaStreamTrack
import time
import numpy as np

class TwilioInputTrack(MediaStreamTrack):
    """
    Receives u-law 8000Hz audio from Twilio and converts it to PCM 48000Hz for the AI pipeline.
    """
    kind = "audio"

    def __init__(self, loop=None):
        super().__init__()
        self.loop = loop
        self.queue = asyncio.Queue()
        self.samplerate = 48000
        self.channels = 1 
        self._start_time = None
        self._timestamp = 0
        self._decode_err_count = 0

    def add_mulaw_chunk(self, mulaw_payload: str):
        """
        Takes a base64 encoded mulaw chunk from Twilio.
        """
        try:
            # 1. Decode Base64 (or accept raw bytes/array)
            if isinstance(mulaw_payload, str):
                mulaw_bytes = base64.b64decode(mulaw_payload)
            elif isinstance(mulaw_payload, (bytes, bytearray)):
                mulaw_bytes = bytes(mulaw_payload)
            elif isinstance(mulaw_payload, list):
                if mulaw_payload and isinstance(mulaw_payload[0], str):
                    mulaw_bytes = base64.b64decode("".join(mulaw_payload))
                else:
                    mulaw_bytes = bytes(mulaw_payload)
            else:
                raise TypeError(f"unsupported payload type: {type(mulaw_payload).__name__}")
            
            # 2. Convert mulaw (8kHz) -> PCM 16-bit (8kHz)
            # audioop.ulaw2lin(fragment, width) -> width=2 for 16-bit
            pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)
            
            # 3. Resample 8kHz -> 48kHz
            # Using audioop.ratecv(fragment, width, nchannels, inrate, outrate, state)
            # state can be None
            pcm_48k, _ = audioop.ratecv(pcm_8k, 2, 1, 8000, 48000, None)
            
            # 4. Create AudioFrame
            frame = av.AudioFrame(format='s16', layout='mono', samples=len(pcm_48k)//2)
            frame.planes[0].update(pcm_48k)
            frame.sample_rate = 48000
            frame.time_base = Fraction(1, self.samplerate)
            
            if self.loop is not None:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, frame)
            else:
                self.queue.put_nowait(frame)
            
        except Exception as e:
            self._decode_err_count += 1
            if self._decode_err_count <= 8 or self._decode_err_count % 200 == 0:
                import traceback
                print(
                    f"❌ Media Decode Error #{self._decode_err_count}: {e} "
                    f"(payload_type={type(mulaw_payload).__name__})"
                )
                traceback.print_exc()

    async def recv(self):
        if self._start_time is None:
            self._start_time = time.time()

        frame = await self.queue.get()
        
        # Timing logic
        frame.pts = self._timestamp
        self._timestamp += frame.samples
        
        return frame
