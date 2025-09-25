"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from listener import record_audio, transcribe
from tts_engine import speak
from nlp_engine import generate_response, normalize_text, lemmatize_text, tokenize, get_embedding

print("=== A.L.I.C.E. Audio Test ===")
print("Habla y espera la respuesta. Ctrl+C para salir.\n")

while True:
    try:
        # 1) Recolección de audio
        audio = record_audio(5)

        # 2) Transcripción con Whisper
        user_text = transcribe(audio)
        if not user_text:
            continue

        print(f"[Tú] {user_text}")

        # 3) Preprocesamiento
        text_norm = normalize_text(user_text)
        text_lem = lemmatize_text(text_norm)
        tokens = tokenize(text_lem)
        print(f"[Preprocesado] Normalizado: {text_norm}")
        print(f"[Preprocesado] Lematizado: {text_lem}")
        print(f"[Preprocesado] Tokens: {tokens}")

        # 4) Embedding
        embedding = get_embedding(text_lem)
        print(f"[Embedding] Dimensión: {embedding.shape[0]}")
        print(f"[Embedding] Vector (primeros 5 valores): {embedding[:5]}")

        # 5) Generación de respuesta + contexto
        response_text, tokens, _ = generate_response(user_text)

        print(f"[ALICE] {response_text}")

        # 6) Síntesis de voz
        speak(response_text)

    except KeyboardInterrupt:
        break
    except Exception as e:
        print("[ALICE] Error:", e)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tts_engine import speak
from nlp_engine import generate_response, normalize_text, lemmatize_text, get_embedding

print("=== A.L.I.C.E. Text Mode ===")
print("Escribe un mensaje y espera la respuesta. Escribe 'salir' para terminar.\n")

while True:
    try:
        user_text = input("Tú: ").strip()
        if not user_text:
            continue
        if user_text.lower() in ["salir", "exit", "quit"]:
            print("[ALICE] Hasta pronto 👋")
            break

        text_norm = normalize_text(user_text)
        text_lem = lemmatize_text(text_norm)
        print(f"[Preprocesado] Normalizado: {text_norm}")
        print(f"[Preprocesado] Lematizado: {text_lem}")

        embedding = get_embedding(text_lem)
        print(f"[Embedding] Dimensión: {embedding.shape[0]}")
        print(f"[Embedding] Vector (primeros 5 valores): {embedding[:5]}")

        response_text, _ = generate_response(user_text)
        response_text = response_text.replace("<extra_id_0>", "").strip()
        print(f"[ALICE] {response_text}")

        speak(response_text)

    except KeyboardInterrupt:
        print("\n[ALICE] Interrumpido por el usuario. Saliendo...")
        break
    except Exception as e:
        print("[ALICE] Error:", e)
