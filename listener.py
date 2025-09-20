import sounddevice as sd
import numpy as np
import time
from faster_whisper import WhisperModel
from vad_silero import get_speech_timestamps, collect_chunks, VadOptions

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

SAMPLE_RATE = 16000

def record_audio(duration=5):
    """
    Graba audio sin aplicar VAD (se filtra después con Silero VAD).
    """
    print("[ALICE] Escuchando...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    audio = np.squeeze(audio).astype(np.float32) / 32768.0
    return audio

def transcribe(audio, language="es"):
    """
    Aplica VAD (Silero) + transcripción con Whisper.
    """
    if len(audio) == 0:
        return ""

    vad_opts = VadOptions(threshold=0.5, min_silence_duration_ms=500)
    timestamps = get_speech_timestamps(audio, vad_options=vad_opts, sampling_rate=SAMPLE_RATE)

    if not timestamps:
        return ""

    audio_chunks, _ = collect_chunks(audio, timestamps, sampling_rate=SAMPLE_RATE)

    texts = []
    for chunk in audio_chunks:
        if len(chunk) == 0:
            continue
        segments, _ = model.transcribe(chunk, beam_size=5, language=language)
        texts.append(" ".join([seg.text for seg in segments]))

    return " ".join(texts).strip()
