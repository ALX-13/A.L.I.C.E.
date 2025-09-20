import sounddevice as sd
import numpy as np
import struct
import time
from faster_whisper import WhisperModel
import webrtcvad

# --- Inicialización Whisper ---
"""
WhisperModel:

- Modelos disponibles:
    - tiny
    - base
    - small
    - medium
    - large
    - large-v2
    - large-v3

Cada modelo representa un trade-off entre velocidad y precisión:
- Los más pequeños (tiny, base) son más rápidos pero menos precisos.
- Los medianos (small, medium) equilibran velocidad y exactitud.
- Los grandes (large, large-v2, large-v3) ofrecen la máxima precisión a costa de rendimiento.
"""

model = WhisperModel("small", device="cpu")

vad = webrtcvad.Vad()
vad.set_mode(2)  # 0-3, más alto = más estricto

FRAME_DURATION = 20
SAMPLE_RATE = 16000

def frame_generator(audio, sample_rate=SAMPLE_RATE, frame_duration_ms=FRAME_DURATION):
    n = int(sample_rate * frame_duration_ms / 1000.0)
    for offset in range(0, len(audio) - n + 1, n):
        yield audio[offset:offset+n]

def record_audio_vad(duration):
    print("[ALICE] Escuchando...")
    voiced_frames = []
    start_time = time.time()

    while time.time() - start_time < duration:
        audio_chunk = sd.rec(int(0.5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        audio_chunk = np.squeeze(audio_chunk).astype(np.float32) / 32768.0

        if np.max(np.abs(audio_chunk)) > 0:
            audio_chunk /= np.max(np.abs(audio_chunk))

        for frame in frame_generator(audio_chunk):
            frame_int16 = (frame * 32768).astype(np.int16)
            pcm_data = struct.pack("<" + "h"*len(frame_int16), *frame_int16)
            if vad.is_speech(pcm_data, SAMPLE_RATE):
                voiced_frames.extend(frame)

    if not voiced_frames:
        return np.array([], dtype=np.float32)

    voiced_audio = np.array(voiced_frames, dtype=np.float32)

    if np.max(np.abs(voiced_audio)) > 0:
        voiced_audio /= np.max(np.abs(voiced_audio))
        
    return voiced_audio

def transcribe(audio, language="es"):
    if len(audio) == 0:
        return ""
    segments, info = model.transcribe(audio, beam_size=5, language=language)
    text = " ".join([segment.text for segment in segments])
    return text.strip()
