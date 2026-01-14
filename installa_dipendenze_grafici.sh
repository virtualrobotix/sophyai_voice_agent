#!/bin/bash
# Script per installare dipendenze per generazione grafici

echo "📦 Installazione dipendenze per generazione grafici..."

# Verifica Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 non trovato!"
    exit 1
fi

# Installa dipendenze
echo "Installing matplotlib, networkx..."
pip3 install matplotlib networkx pillow

echo "✅ Dipendenze installate!"
echo ""
echo "Ora puoi eseguire:"
echo "  python3 genera_grafici.py"
