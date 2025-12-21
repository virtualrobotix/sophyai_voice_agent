#!/usr/bin/env python3
"""
Script di avvio per Voice Agent.
Avvia tutti i servizi necessari.
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

# Colori per output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'


def log(message: str, color: str = Colors.GREEN):
    """Stampa un messaggio colorato"""
    print(f"{color}[Voice Agent]{Colors.END} {message}")


def check_docker():
    """Verifica che Docker sia installato e in esecuzione"""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def check_ollama():
    """Verifica che Ollama sia in esecuzione"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def start_docker_services():
    """Avvia i servizi Docker (LiveKit + Redis)"""
    log("Avvio LiveKit e Redis...", Colors.BLUE)
    
    result = subprocess.run(
        ["docker-compose", "up", "-d"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        log(f"Errore avvio Docker: {result.stderr}", Colors.RED)
        return False
    
    # Attendi che LiveKit sia pronto
    log("Attendo che LiveKit sia pronto...", Colors.YELLOW)
    time.sleep(5)
    
    return True


def start_web_server():
    """Avvia il web server in background"""
    log("Avvio Web Server...", Colors.BLUE)
    
    process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    return process


def start_voice_agent():
    """Avvia il Voice Agent"""
    log("Avvio Voice Agent...", Colors.BLUE)
    
    process = subprocess.Popen(
        [sys.executable, "-m", "agent.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    return process


def main():
    """Funzione principale"""
    log("=" * 50)
    log("🎙️  Voice Agent - Sistema di Avvio")
    log("=" * 50)
    
    processes = []
    
    def cleanup(signum=None, frame=None):
        """Pulisce i processi all'uscita"""
        log("\nArresto servizi...", Colors.YELLOW)
        
        for proc in processes:
            if proc and proc.poll() is None:
                proc.terminate()
                proc.wait()
        
        # Stop Docker services
        subprocess.run(["docker-compose", "down"], capture_output=True)
        
        log("Servizi arrestati.", Colors.GREEN)
        sys.exit(0)
    
    # Gestisci SIGINT e SIGTERM
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # 1. Verifica Docker
    log("Verifica Docker...", Colors.YELLOW)
    if not check_docker():
        log("Docker non trovato o non in esecuzione!", Colors.RED)
        log("Installa Docker e riavvia.", Colors.RED)
        sys.exit(1)
    log("✓ Docker OK", Colors.GREEN)
    
    # 2. Verifica Ollama
    log("Verifica Ollama...", Colors.YELLOW)
    if not check_ollama():
        log("Ollama non in esecuzione!", Colors.YELLOW)
        log("Avvia Ollama con: ollama serve", Colors.YELLOW)
        log("Continuo comunque...", Colors.YELLOW)
    else:
        log("✓ Ollama OK", Colors.GREEN)
    
    # 3. Avvia Docker services
    if not start_docker_services():
        log("Impossibile avviare i servizi Docker!", Colors.RED)
        sys.exit(1)
    log("✓ LiveKit e Redis avviati", Colors.GREEN)
    
    # 4. Avvia Web Server
    web_proc = start_web_server()
    processes.append(web_proc)
    time.sleep(2)
    
    if web_proc.poll() is not None:
        log("Errore avvio Web Server!", Colors.RED)
        cleanup()
    log("✓ Web Server avviato su http://localhost:8080", Colors.GREEN)
    
    # 5. Avvia Voice Agent
    agent_proc = start_voice_agent()
    processes.append(agent_proc)
    time.sleep(2)
    
    if agent_proc.poll() is not None:
        log("Errore avvio Voice Agent!", Colors.RED)
        cleanup()
    log("✓ Voice Agent avviato", Colors.GREEN)
    
    # Pronto!
    log("=" * 50)
    log("🎉 Sistema pronto!", Colors.GREEN)
    log("📱 Apri http://localhost:8080 nel browser")
    log("🛑 Premi Ctrl+C per arrestare")
    log("=" * 50)
    
    # Mantieni in esecuzione
    try:
        while True:
            # Verifica che i processi siano attivi
            for proc in processes:
                if proc.poll() is not None:
                    log("Un processo si è arrestato. Riavvio...", Colors.YELLOW)
                    cleanup()
            
            time.sleep(5)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()







