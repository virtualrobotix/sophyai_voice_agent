#!/bin/bash
# Setup script per server con GPU NVIDIA CUDA
# Uso: ./setup_cuda.sh

set -e

echo "🚀 Setup SophyAI Voice Agent con GPU NVIDIA CUDA"
echo "================================================"

# Verifica NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi non trovato! Assicurati che i driver NVIDIA siano installati."
    echo "   Installazione driver: sudo apt install nvidia-driver-535"
    exit 1
fi

echo "✅ GPU NVIDIA rilevata:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Crea virtual environment
echo ""
echo "📦 Creazione ambiente virtuale..."
python3 -m venv venv
source venv/bin/activate

# Installa PyTorch con CUDA
echo ""
echo "🔥 Installazione PyTorch con CUDA 12.1..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verifica CUDA
echo ""
echo "🔍 Verifica PyTorch + CUDA..."
python -c "import torch; print('PyTorch versione:', torch.__version__); print('CUDA disponibile:', torch.cuda.is_available()); print('CUDA versione:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Installa dipendenze
echo ""
echo "📦 Installazione dipendenze..."
pip install -r requirements-cuda.txt

# Installa VibeVoice
echo ""
echo "🎤 Installazione VibeVoice..."
if [ ! -d "VibeVoice" ]; then
    git clone https://github.com/microsoft/VibeVoice.git
fi
cd VibeVoice && pip install -e . && cd ..

# Copia .env se non esiste
if [ ! -f ".env" ]; then
    echo ""
    echo "📝 Creazione file .env..."
    cp env.example .env
    # Abilita CUDA per Whisper
    sed -i 's/WHISPER_DEVICE=cpu/WHISPER_DEVICE=cuda/' .env
fi

echo ""
echo "✅ Setup completato!"
echo ""
echo "Per avviare:"
echo "  1. Attiva l'ambiente: source venv/bin/activate"
echo "  2. Avvia il TTS server: python tts_server.py"
echo "  3. In un altro terminale: python -m agent.main"
echo ""
echo "Il TTS server userà automaticamente la GPU CUDA."





