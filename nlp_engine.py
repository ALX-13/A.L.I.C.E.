import re
import unicodedata
import numpy as np
import faiss
import emoji
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from config import MAX_CONTEXT
import tiktoken
from unsloth import FastModel, do_gemma_3n_inference

try:
    import spacy
    nlp = spacy.load("es_core_news_lg")
    USE_SPACY = True
except ImportError:
    USE_SPACY = False
    print("[NLP Engine] spaCy no disponible, se usará solo tokenización básica")

try:
    import stanza
    stanza.download("es")
    nlp_stanza = stanza.Pipeline("es", processors="tokenize,pos,lemma", use_gpu=False)
    USE_STANZA = True
except ImportError:
    print("[NLP Engine] Stanza no disponible, fallback a texto limpio.")
    USE_STANZA = False

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedding_dim = embedder.get_sentence_embedding_dimension()

# === Configuración de FAISS ===
faiss_index_cpu = faiss.IndexHNSWFlat(embedding_dim, 32)  # similitud coseno
gpu_index = None
use_gpu = False

try:
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss_index_cpu)
    use_gpu = True
    print("[FAISS] Usando GPU")
except Exception:
    print("[FAISS] GPU no disponible, usando CPU")

embedding_history = []

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it",  # Modelo libre
    dtype = None,
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False
)

def normalize_text(text: str) -> str:
    text = text.lower()
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"[^a-zA-Z0-9áéíóúñü¿?¡!,.(){}\[\]+\-*=<> ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lemmatize_text(text: str) -> str:
    if USE_SPACY:
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])
    return text

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm > 0:
        return embedding / norm
    return embedding

def get_embedding(text: str) -> np.ndarray:
    emb = embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return emb.astype(np.float32)

def process_prompt(prompt_text: str):
    text_norm = normalize_text(prompt_text)
    text_lem = lemmatize_text(text_norm)
    embedding = get_embedding(text_lem)

    faiss_index_cpu.add(np.array([embedding], dtype=np.float32))

    embedding_history.append({
        "text": text_lem,
        "embedding": embedding
    })

    return text_lem, embedding

def generate_response(prompt_text, top_k=MAX_CONTEXT, max_tokens= 64, temperature=0.7):
    # 1) Preprocesamiento
    text_norm = normalize_text(prompt_text)
    text_lem = lemmatize_text(text_norm)
    embedding = get_embedding(text_lem)

    # 2) Recuperar contexto relevante, excluyendo input actual
    context_texts = []
    if faiss_index_cpu.ntotal > 0:
        D, I = faiss_index_cpu.search(np.array([embedding], dtype=np.float32), top_k)
        for idx in I[0]:
            if idx < len(embedding_history):
                context_texts.append(embedding_history[idx]["text"])
        context_texts = context_texts[:top_k]

    context_str = " ".join(context_texts) if context_texts else ""

    # 3) Preparar prompt instructivo para Gemma 3N
    messages = [{
        "role": "user",
        "content": [{
            "type": "text",
            "text": (
                "Eres un asistente virtual que responde de manera natural, clara y útil en español.\n"
                f"Contexto previo: {context_str}\n"
                f"Pregunta del usuario: {text_lem}\n"
                "Respuesta:"
            )
        }]
    }]

    # 4) Inferencia Gemma 3N (CPU)
    response = do_gemma_3n_inference(
        model,
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature
    )
    response_text = response[0]["content"][0]["text"].strip()
    if not response_text:
        response_text = "Lo siento, no entendí tu mensaje."

    # 5) Guardar input y embedding en FAISS después de generar la respuesta
    faiss_index_cpu.add(np.array([embedding], dtype=np.float32))
    embedding_history.append({
        "text": text_lem,
        "embedding": embedding
    })

    return response_text, embedding
