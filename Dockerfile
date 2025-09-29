FROM python:3.12-slim

# Evitar que Python genere pyc y logs bufferizados
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Instalar dependencias del sistema (audio, compilación, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    portaudio19-dev \
    libsndfile1 \
    espeak-ng \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements primero (para aprovechar cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Comando por defecto: ejecutar en modo texto
CMD ["python", "__main__.py"]
