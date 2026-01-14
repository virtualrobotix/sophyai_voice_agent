#!/usr/bin/env python3
"""
Script per generare tutti i grafici della presentazione.

Requisiti:
    pip install matplotlib networkx graphviz pillow

Uso:
    python genera_grafici.py
"""

import os
from pathlib import Path

# Crea directory per i grafici
output_dir = Path("grafici_presentazione")
output_dir.mkdir(exist_ok=True)

print("🎨 Generazione grafici per la presentazione...")

# 1. Architettura ad Alto Livello
print("📊 1. Creando diagramma architettura...")
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Client Browser
    browser = FancyBboxPatch((1, 7), 2, 1.5, boxstyle="round,pad=0.1", 
                            edgecolor='#2563eb', facecolor='#dbeafe', linewidth=2)
    ax.add_patch(browser)
    ax.text(2, 8, 'Browser\nClient', ha='center', va='center', fontsize=12, weight='bold')
    
    # LiveKit Server
    livekit = FancyBboxPatch((4.5, 7), 2, 1.5, boxstyle="round,pad=0.1",
                             edgecolor='#059669', facecolor='#d1fae5', linewidth=2)
    ax.add_patch(livekit)
    ax.text(5.5, 8, 'LiveKit\nServer', ha='center', va='center', fontsize=12, weight='bold')
    
    # Voice Agent
    agent = FancyBboxPatch((4.5, 4.5), 2, 1.5, boxstyle="round,pad=0.1",
                           edgecolor='#dc2626', facecolor='#fee2e2', linewidth=2)
    ax.add_patch(agent)
    ax.text(5.5, 5.25, 'Voice Agent\nWorker', ha='center', va='center', fontsize=12, weight='bold')
    
    # STT
    stt = FancyBboxPatch((1, 4.5), 1.8, 1.5, boxstyle="round,pad=0.1",
                         edgecolor='#7c3aed', facecolor='#ede9fe', linewidth=2)
    ax.add_patch(stt)
    ax.text(1.9, 5.25, 'Whisper\nSTT', ha='center', va='center', fontsize=11, weight='bold')
    
    # LLM
    llm = FancyBboxPatch((4.5, 2), 2, 1.5, boxstyle="round,pad=0.1",
                         edgecolor='#ea580c', facecolor='#ffedd5', linewidth=2)
    ax.add_patch(llm)
    ax.text(5.5, 2.75, 'LLM\nProvider', ha='center', va='center', fontsize=11, weight='bold')
    
    # TTS
    tts = FancyBboxPatch((7.5, 4.5), 1.8, 1.5, boxstyle="round,pad=0.1",
                         edgecolor='#0891b2', facecolor='#cffafe', linewidth=2)
    ax.add_patch(tts)
    ax.text(8.4, 5.25, 'TTS\nEngines', ha='center', va='center', fontsize=11, weight='bold')
    
    # Database
    db = FancyBboxPatch((4.5, 0.5), 2, 1, boxstyle="round,pad=0.1",
                        edgecolor='#be185d', facecolor='#fce7f3', linewidth=2)
    ax.add_patch(db)
    ax.text(5.5, 1, 'PostgreSQL\nDatabase', ha='center', va='center', fontsize=11, weight='bold')
    
    # Frecce
    # Browser -> LiveKit
    arrow1 = FancyArrowPatch((3, 7.75), (4.5, 7.75), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#2563eb')
    ax.add_patch(arrow1)
    ax.text(3.75, 8.1, 'WebRTC', ha='center', fontsize=9, style='italic')
    
    # LiveKit -> Agent
    arrow2 = FancyArrowPatch((5.5, 7), (5.5, 6), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#059669')
    ax.add_patch(arrow2)
    
    # Agent -> STT
    arrow3 = FancyArrowPatch((4.5, 5.25), (2.8, 5.25), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#7c3aed')
    ax.add_patch(arrow3)
    
    # Agent -> LLM
    arrow4 = FancyArrowPatch((5.5, 4.5), (5.5, 3.5), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#ea580c')
    ax.add_patch(arrow4)
    
    # Agent -> TTS
    arrow5 = FancyArrowPatch((6.5, 5.25), (7.5, 5.25), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#0891b2')
    ax.add_patch(arrow5)
    
    # Agent -> DB
    arrow6 = FancyArrowPatch((5.5, 4.5), (5.5, 1.5), 
                            arrowstyle='->', mutation_scale=20, linewidth=2, color='#be185d')
    ax.add_patch(arrow6)
    
    # LiveKit -> Agent (ritorno)
    arrow7 = FancyArrowPatch((5.5, 6), (5.5, 7), 
                            arrowstyle='->', mutation_scale=20, linewidth=1.5, 
                            color='#059669', linestyle='--', alpha=0.6)
    ax.add_patch(arrow7)
    
    # LiveKit -> Browser (ritorno)
    arrow8 = FancyArrowPatch((4.5, 7.75), (3, 7.75), 
                            arrowstyle='->', mutation_scale=20, linewidth=1.5, 
                            color='#2563eb', linestyle='--', alpha=0.6)
    ax.add_patch(arrow8)
    
    plt.title('Architettura SophyAI Live Server', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / '01_architettura.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Architettura salvata")
except Exception as e:
    print(f"   ⚠️ Errore architettura: {e}")

# 2. Flusso End-to-End
print("📊 2. Creando diagramma flusso end-to-end...")
try:
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Componenti verticali
    components = [
        (1, 'User\nSpeaks', '#fef3c7'),
        (3, 'Browser\n(Mic)', '#dbeafe'),
        (5, 'LiveKit\n(WebRTC)', '#d1fae5'),
        (7, 'Agent\nWorker', '#fee2e2'),
        (9, 'Whisper\nSTT', '#ede9fe'),
        (11, 'LLM\n(Ollama/OR)', '#ffedd5'),
        (13, 'TTS\nEngine', '#cffafe'),
        (15, 'User\nHears', '#fef3c7')
    ]
    
    for x, label, color in components:
        box = FancyBboxPatch((x-0.4, 3), 0.8, 2, boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 4, label, ha='center', va='center', fontsize=9, weight='bold')
    
    # Frecce
    for i in range(len(components)-1):
        x1 = components[i][0]
        x2 = components[i+1][0]
        arrow = FancyArrowPatch((x1+0.4, 4), (x2-0.4, 4),
                               arrowstyle='->', mutation_scale=15, linewidth=2, color='#059669')
        ax.add_patch(arrow)
    
    # Timeline
    ax.plot([0.5, 15.5], [1.5, 1.5], 'k--', linewidth=1, alpha=0.3)
    ax.text(8, 1.2, 'Flusso Audio: Input → Processing → Output', 
           ha='center', fontsize=11, style='italic', weight='bold')
    
    # Latenze
    latencies = [
        (2, '~250ms', 'STT'),
        (6, '~1200ms', 'LLM'),
        (10, '~500ms', 'TTS'),
        (14, '~1950ms', 'Totale E2E')
    ]
    
    for x, time, label in latencies:
        ax.text(x, 2.3, time, ha='center', fontsize=8, color='#dc2626', weight='bold')
        ax.text(x, 2.1, label, ha='center', fontsize=7, style='italic')
    
    plt.title('Flusso End-to-End: Da Input Vocale a Output Vocale', 
             fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / '02_flusso_e2e.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Flusso E2E salvato")
except Exception as e:
    print(f"   ⚠️ Errore flusso E2E: {e}")

# 3. Stack Tecnologico
print("📊 3. Creando diagramma stack tecnologico...")
try:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Layer
    layers = [
        (6, 7, 'Frontend', ['HTML5', 'JavaScript', 'WebRTC API', 'Web Audio API'], '#dbeafe'),
        (6, 5.5, 'Web Server', ['FastAPI', 'Python 3.10+', 'REST API'], '#d1fae5'),
        (6, 4, 'WebRTC', ['LiveKit', 'Agents SDK'], '#fef3c7'),
        (6, 2.5, 'AI/ML', ['Whisper', 'Ollama', 'OpenRouter', 'TTS Engines'], '#fee2e2'),
        (6, 1, 'Infrastructure', ['PostgreSQL', 'Redis', 'Docker'], '#ede9fe')
    ]
    
    for x, y, title, items, color in layers:
        # Box layer
        box = FancyBboxPatch((x-2, y-0.4), 4, 0.8, boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, title, ha='center', va='center', fontsize=11, weight='bold')
        
        # Items
        items_text = ' • '.join(items)
        ax.text(x, y-0.15, items_text, ha='center', va='top', fontsize=8)
    
    plt.title('Stack Tecnologico', fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / '03_stack_tecnologico.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Stack tecnologico salvato")
except Exception as e:
    print(f"   ⚠️ Errore stack: {e}")

# 4. Deployment Docker
print("📊 4. Creando diagramma deployment Docker...")
try:
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Docker Network
    docker_box = FancyBboxPatch((0.5, 5.5), 6, 4, boxstyle="round,pad=0.1",
                               edgecolor='#2563eb', facecolor='#eff6ff', linewidth=3)
    ax.add_patch(docker_box)
    ax.text(3.5, 9.2, 'Docker Network: voiceagent', ha='center', fontsize=12, weight='bold', color='#1e40af')
    
    # Containers
    containers = [
        (1.5, 8, 'PostgreSQL\n:5432', '#be185d'),
        (4, 8, 'Redis\n:6379', '#dc2626'),
        (1.5, 6.5, 'Web Server\n:8080:8443', '#059669'),
        (4, 6.5, 'Agent\nWorker', '#7c3aed')
    ]
    
    for x, y, label, color in containers:
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8, boxstyle="round,pad=0.05",
                            edgecolor=color, facecolor='white', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, weight='bold')
    
    # Host Machine
    host_box = FancyBboxPatch((7.5, 1), 6, 8, boxstyle="round,pad=0.1",
                             edgecolor='#059669', facecolor='#f0fdf4', linewidth=3)
    ax.add_patch(host_box)
    ax.text(10.5, 8.5, 'Host Machine', ha='center', fontsize=12, weight='bold', color='#166534')
    
    # Host Services
    host_services = [
        (9, 7, 'Ollama\n:11434', '#ea580c'),
        (12, 7, 'TTS Server\n:8092', '#0891b2'),
        (10.5, 5, 'LiveKit\n:7880', '#7c3aed')
    ]
    
    for x, y, label, color in host_services:
        box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8, boxstyle="round,pad=0.05",
                            edgecolor=color, facecolor='white', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, weight='bold')
    
    # Frecce connessioni
    arrow1 = FancyArrowPatch((7.5, 6.5), (9, 6.5),
                            arrowstyle='->', mutation_scale=15, linewidth=2, color='#059669')
    ax.add_patch(arrow1)
    ax.text(8.25, 6.8, 'host.docker.internal', ha='center', fontsize=7, style='italic')
    
    arrow2 = FancyArrowPatch((7.5, 6.2), (9, 5.5),
                            arrowstyle='->', mutation_scale=15, linewidth=2, color='#059669')
    ax.add_patch(arrow2)
    
    plt.title('Deployment Architecture - Docker Compose', fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / '04_deployment_docker.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Deployment Docker salvato")
except Exception as e:
    print(f"   ⚠️ Errore deployment: {e}")

# 5. Confronto TTS Engines
print("📊 5. Creando grafico confronto TTS engines...")
try:
    engines = ['Edge', 'Piper', 'Coqui', 'Kokoro', 'VibeVoice', 'Chatterbox']
    self_hosted = [False, True, True, True, True, True]
    quality = [4, 3, 4, 4, 5, 5]  # 1-5 scale
    speed = [5, 5, 3, 3, 4, 3]  # 1-5 scale
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Grafico qualità
    colors = ['#dc2626' if not sh else '#059669' for sh in self_hosted]
    bars1 = ax1.barh(engines, quality, color=colors)
    ax1.set_xlabel('Qualità (1-5)', fontsize=11, weight='bold')
    ax1.set_title('Qualità TTS Engines', fontsize=12, weight='bold')
    ax1.set_xlim(0, 5.5)
    ax1.grid(axis='x', alpha=0.3)
    
    # Aggiungi valori
    for i, (bar, val) in enumerate(zip(bars1, quality)):
        ax1.text(val + 0.1, i, f'{val}/5', va='center', fontsize=10, weight='bold')
    
    # Legenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#dc2626', label='Cloud'),
        Patch(facecolor='#059669', label='Self-Hosted')
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Grafico velocità
    bars2 = ax2.barh(engines, speed, color=colors)
    ax2.set_xlabel('Velocità (1-5)', fontsize=11, weight='bold')
    ax2.set_title('Velocità TTS Engines', fontsize=12, weight='bold')
    ax2.set_xlim(0, 5.5)
    ax2.grid(axis='x', alpha=0.3)
    
    # Aggiungi valori
    for i, (bar, val) in enumerate(zip(bars2, speed)):
        ax2.text(val + 0.1, i, f'{val}/5', va='center', fontsize=10, weight='bold')
    
    plt.suptitle('Confronto TTS Engines', fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / '05_confronto_tts.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Confronto TTS salvato")
except Exception as e:
    print(f"   ⚠️ Errore confronto TTS: {e}")

# 6. Metriche Performance
print("📊 6. Creando grafico metriche performance...")
try:
    categories = ['STT\n(Whisper)', 'LLM\n(Ollama)', 'TTS\n(Edge)', 'Totale\nE2E']
    latencies_cpu = [250, 1200, 500, 1950]  # ms
    latencies_gpu = [150, 800, 300, 1250]  # ms
    
    x = range(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar([i - width/2 for i in x], latencies_cpu, width, 
                   label='CPU (Small Model)', color='#dc2626', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], latencies_gpu, width,
                   label='GPU (Advanced)', color='#059669', alpha=0.8)
    
    ax.set_ylabel('Latenza (ms)', fontsize=11, weight='bold')
    ax.set_title('Metriche Performance - Confronto CPU vs GPU', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Aggiungi valori
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                   f'{int(height)}ms', ha='center', va='bottom', fontsize=9, weight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_metriche_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Metriche performance salvate")
except Exception as e:
    print(f"   ⚠️ Errore metriche: {e}")

# 7. Database Schema
print("📊 7. Creando diagramma database schema...")
try:
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Tabella settings
    settings_box = FancyBboxPatch((1, 5.5), 3, 2, boxstyle="round,pad=0.1",
                                  edgecolor='#7c3aed', facecolor='#f3e8ff', linewidth=2)
    ax.add_patch(settings_box)
    ax.text(2.5, 7, 'settings', ha='center', fontsize=12, weight='bold')
    ax.text(2.5, 6.5, 'key (PK)', ha='center', fontsize=9)
    ax.text(2.5, 6.1, 'value', ha='center', fontsize=9)
    ax.text(2.5, 5.7, 'updated_at', ha='center', fontsize=9)
    
    # Tabella chats
    chats_box = FancyBboxPatch((5.5, 5.5), 3, 2, boxstyle="round,pad=0.1",
                              edgecolor='#059669', facecolor='#d1fae5', linewidth=2)
    ax.add_patch(chats_box)
    ax.text(7, 7, 'chats', ha='center', fontsize=12, weight='bold')
    ax.text(7, 6.5, 'id (PK)', ha='center', fontsize=9)
    ax.text(7, 6.1, 'title', ha='center', fontsize=9)
    ax.text(7, 5.7, 'created_at', ha='center', fontsize=9)
    ax.text(7, 5.3, 'updated_at', ha='center', fontsize=9)
    
    # Tabella messages
    messages_box = FancyBboxPatch((9, 5.5), 3, 2, boxstyle="round,pad=0.1",
                                  edgecolor='#ea580c', facecolor='#ffedd5', linewidth=2)
    ax.add_patch(messages_box)
    ax.text(10.5, 7, 'messages', ha='center', fontsize=12, weight='bold')
    ax.text(10.5, 6.5, 'id (PK)', ha='center', fontsize=9)
    ax.text(10.5, 6.1, 'chat_id (FK)', ha='center', fontsize=9)
    ax.text(10.5, 5.7, 'role', ha='center', fontsize=9)
    ax.text(10.5, 5.3, 'content', ha='center', fontsize=9)
    ax.text(10.5, 4.9, 'created_at', ha='center', fontsize=9)
    
    # Relazione
    arrow = FancyArrowPatch((7, 5.5), (10.5, 5.5),
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#059669')
    ax.add_patch(arrow)
    ax.text(8.75, 5.2, '1:N', ha='center', fontsize=10, weight='bold', color='#059669')
    
    # Descrizione
    ax.text(6, 3.5, 'PostgreSQL Schema', ha='center', fontsize=14, weight='bold')
    ax.text(6, 3, '3 tabelle principali per settings, chats e messages', 
           ha='center', fontsize=10, style='italic')
    
    plt.title('Database Schema', fontsize=14, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / '07_database_schema.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Database schema salvato")
except Exception as e:
    print(f"   ⚠️ Errore database schema: {e}")

# 8. Sequenza Multi-User
print("📊 8. Creando diagramma sequenza multi-user...")
try:
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Attori
    actors = ['User1', 'Browser1', 'LiveKit', 'Agent', 'Browser2', 'User2']
    x_positions = [1, 3, 6, 9, 11, 13]
    
    for i, (actor, x) in enumerate(zip(actors, x_positions)):
        # Linea verticale
        ax.plot([x, x], [0.5, 7.5], 'k-', linewidth=1, alpha=0.3)
        # Box attore
        box = FancyBboxPatch((x-0.4, 7.2), 0.8, 0.3, boxstyle="round,pad=0.02",
                            edgecolor='black', facecolor='#dbeafe', linewidth=1)
        ax.add_patch(box)
        ax.text(x, 7.35, actor, ha='center', va='center', fontsize=8, weight='bold')
    
    # Messaggi
    y = 6.5
    # User1 parla
    arrow1 = FancyArrowPatch((1, y), (3, y), arrowstyle='->', mutation_scale=10, 
                            linewidth=1.5, color='#2563eb')
    ax.add_patch(arrow1)
    ax.text(2, y+0.1, '"Ciao"', ha='center', fontsize=8, style='italic')
    y -= 0.5
    
    # Browser1 -> LiveKit
    arrow2 = FancyArrowPatch((3, y), (6, y), arrowstyle='->', mutation_scale=10,
                            linewidth=1.5, color='#2563eb')
    ax.add_patch(arrow2)
    y -= 0.5
    
    # LiveKit -> Agent
    arrow3 = FancyArrowPatch((6, y), (9, y), arrowstyle='->', mutation_scale=10,
                            linewidth=1.5, color='#059669')
    ax.add_patch(arrow3)
    ax.text(7.5, y+0.1, 'STT: "Ciao"', ha='center', fontsize=8)
    y -= 0.5
    
    # Agent ignora (no trigger)
    ax.text(9, y, '❌ Ignore\n(no @sophyai)', ha='center', fontsize=8, 
           color='#dc2626', weight='bold', bbox=dict(boxstyle='round', facecolor='#fee2e2', alpha=0.7))
    y -= 0.8
    
    # User2 menziona agent
    arrow4 = FancyArrowPatch((13, y), (11, y), arrowstyle='->', mutation_scale=10,
                            linewidth=1.5, color='#ea580c')
    ax.add_patch(arrow4)
    ax.text(12, y+0.1, '"@sophyai ciao"', ha='center', fontsize=8, style='italic')
    y -= 0.5
    
    # Browser2 -> LiveKit
    arrow5 = FancyArrowPatch((11, y), (6, y), arrowstyle='->', mutation_scale=10,
                            linewidth=1.5, color='#ea580c')
    ax.add_patch(arrow5)
    y -= 0.5
    
    # LiveKit -> Agent
    arrow6 = FancyArrowPatch((6, y), (9, y), arrowstyle='->', mutation_scale=10,
                            linewidth=1.5, color='#059669')
    ax.add_patch(arrow6)
    ax.text(7.5, y+0.1, 'STT: "@sophyai ciao"', ha='center', fontsize=8)
    y -= 0.5
    
    # Agent risponde
    ax.text(9, y, '✅ Process & Respond', ha='center', fontsize=8,
           color='#059669', weight='bold', bbox=dict(boxstyle='round', facecolor='#d1fae5', alpha=0.7))
    y -= 0.5
    
    # Agent -> LiveKit -> Browser1 e Browser2
    arrow7 = FancyArrowPatch((9, y), (6, y), arrowstyle='->', mutation_scale=10,
                            linewidth=1.5, color='#7c3aed')
    ax.add_patch(arrow7)
    arrow8a = FancyArrowPatch((6, y-0.2), (3, y-0.2), arrowstyle='->', mutation_scale=10,
                             linewidth=1.5, color='#7c3aed')
    ax.add_patch(arrow8a)
    arrow8b = FancyArrowPatch((6, y-0.4), (11, y-0.4), arrowstyle='->', mutation_scale=10,
                             linewidth=1.5, color='#7c3aed')
    ax.add_patch(arrow8b)
    ax.text(7.5, y+0.1, 'Audio Response', ha='center', fontsize=8, style='italic')
    
    plt.title('Multi-User Support - Mention-based Activation', 
             fontsize=13, weight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(output_dir / '08_sequenza_multiuser.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Sequenza multi-user salvata")
except Exception as e:
    print(f"   ⚠️ Errore sequenza multi-user: {e}")

print(f"\n✅ Tutti i grafici generati in: {output_dir}/")
print(f"📊 File creati:")
for file in sorted(output_dir.glob("*.png")):
    print(f"   - {file.name}")
