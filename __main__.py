import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from listener import record_audio, transcribe
from tts_engine import speak
from nlp_engine import history, generate_response

while True:
    try:
        audio = record_audio(5)
        user_text = transcribe(audio)
        if not user_text:
            continue
        
        print(f"[Tú] {user_text}")

        response_text = generate_response(user_text)
        print(f"[ALICE] {response_text}")
        speak(response_text)

    except Exception as e:
        print("[ALICE] Error:", e)
