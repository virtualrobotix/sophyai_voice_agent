# Voice Agent Dockerfile
FROM python:3.11-slim

# Installa dipendenze sistema e Docker CLI
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Directory di lavoro
WORKDIR /app

# Copia requirements e installa dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice
COPY . .

# Copia certificati SSL
COPY certs /app/certs

# Crea directory per modelli
RUN mkdir -p /app/models/piper

# Esponi porta web HTTPS
EXPOSE 8080

# Variabili d'ambiente di default
ENV PYTHONUNBUFFERED=1
ENV LIVEKIT_URL=ws://localhost:7880
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV WEB_PORT=8080
ENV LOG_LEVEL=INFO

# Comando di avvio
CMD ["python", "server.py"]
