from collections import deque
from config import MAX_PROMPTS
import re
import unicodedata

history = deque(maxlen=MAX_PROMPTS)

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import spacy
    nlp = spacy.load("es_core_news_sm")
    USE_SPACY = True
except ImportError:
    USE_SPACY = False
    print("[NLP Engine] spaCy no disponible, se usará solo tokenización básica")

def normalize_text(text: str) -> str:
    """
    Limpia y normaliza el texto:
    - Convierte a minúsculas
    - Quita acentos y caracteres extraños
    - Elimina espacios extra
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r"[^a-zA-Z0-9áéíóúñü¿?¡!,. ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text: str) -> str:
    """
    Convierte las palabras a su lema usando spaCy
    """
    if USE_SPACY:
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])
    return text

def tokenize(text: str) -> list:
    """
    Tokenización subword si tiktoken está disponible,
    si no, tokenización por palabras.
    """
    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        return enc.encode(text)
    else:
        return text.split()

def process_prompt(text: str) -> str:
    """
    Preprocesa, tokeniza y almacena en la memoria
    """
    text = normalize_text(text)
    text = lemmatize_text(text)
    history.append(text)

    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
    else:
        tokens = lemmatize_text.split()

    return text, tokens

def generate_response(prompt: str) -> str:
    """
    Simulación de motor de NLP:
    - Usa historial para mantener contexto
    - Devuelve respuesta fluida y coherente
    """
    text, tokens = process_prompt(prompt)
    
    if len(history) > 1:
        context = " | ".join(list(history)[-3:])
        response = f"Recibido: {text}. Contexto reciente: {context}"
    else:
        response = f"Recibido: {text}"
    
    return response, tokens