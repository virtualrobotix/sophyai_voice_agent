/**
 * Voice Agent - Frontend Application
 * Gestisce la connessione WebRTC e l'interfaccia utente.
 */

// Stato applicazione
const state = {
    room: null,
    localAudioTrack: null,
    connected: false,
    micActive: false,
    selectedTTS: 'piper',
    agentAudioElements: [],  // Track audio elements dell'agent
    isSpeaking: false
};

// Elementi DOM
const elements = {
    connectBtn: document.getElementById('connect-btn'),
    disconnectBtn: document.getElementById('disconnect-btn'),
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    roomName: document.getElementById('room-name'),
    userName: document.getElementById('user-name'),
    visualizer: document.getElementById('visualizer'),
    micStatus: document.getElementById('mic-status'),
    voiceCard: document.getElementById('voice-card'),
    transcriptCard: document.getElementById('transcript-card'),
    transcript: document.getElementById('transcript'),
    ttsGrid: document.getElementById('tts-grid')
};

/**
 * Connetti alla room LiveKit
 */
async function connect() {
    const roomName = elements.roomName.value.trim();
    const userName = elements.userName.value.trim();

    if (!roomName || !userName) {
        alert('Inserisci nome room e nome utente');
        return;
    }

    updateStatus('connecting', 'Connessione in corso...');

    try {
        // Ottieni token dal server
        const tokenResponse = await fetch('/api/token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                room_name: roomName,
                participant_name: userName
            })
        });

        if (!tokenResponse.ok) {
            throw new Error('Errore nel recupero del token');
        }

        const { token, url } = await tokenResponse.json();

        // Crea la room
        state.room = new LivekitClient.Room({
            adaptiveStream: true,
            dynacast: true,
            audioCaptureDefaults: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        // Event listeners
        setupRoomListeners();

        // Connetti
        await state.room.connect(url, token);

        state.connected = true;
        updateStatus('connected', `Connesso a ${roomName}`);
        showVoiceInterface();

        // Invia TTS selezionato
        sendTTSSelection();

    } catch (error) {
        console.error('Errore connessione:', error);
        updateStatus('disconnected', 'Errore: ' + error.message);
        state.connected = false;
    }
}

/**
 * Disconnetti dalla room
 */
async function disconnect() {
    if (state.room) {
        await state.room.disconnect();
        state.room = null;
    }

    if (state.localAudioTrack) {
        state.localAudioTrack.stop();
        state.localAudioTrack = null;
    }

    state.connected = false;
    state.micActive = false;

    updateStatus('disconnected', 'Disconnesso');
    hideVoiceInterface();
}

/**
 * Configura i listener della room
 */
function setupRoomListeners() {
    state.room.on(LivekitClient.RoomEvent.Connected, () => {
        console.log('Connesso alla room');
    });

    state.room.on(LivekitClient.RoomEvent.Disconnected, () => {
        console.log('Disconnesso dalla room');
        disconnect();
    });

    state.room.on(LivekitClient.RoomEvent.TrackSubscribed, (track, publication, participant) => {
        console.log('Track ricevuto:', track.kind, 'da', participant.identity);

        if (track.kind === LivekitClient.Track.Kind.Audio) {
            // Riproduci audio dell'agent
            const audioElement = track.attach();
            audioElement.autoplay = true;
            audioElement.volume = 1.0;
            audioElement.muted = false;
            audioElement.id = 'agent-audio-' + Date.now();
            document.body.appendChild(audioElement);
            
            // Traccia l'elemento audio
            state.agentAudioElements.push(audioElement);
            state.isSpeaking = true;
            
            // Forza la riproduzione (gestisce autoplay policy)
            audioElement.play().then(() => {
                console.log('Audio in riproduzione!');
            }).catch(err => {
                console.error('Errore autoplay:', err);
                // Prova a riprodurre dopo interazione utente
                document.addEventListener('click', () => audioElement.play(), { once: true });
            });
            
            // Quando l'audio finisce
            audioElement.onended = () => {
                state.isSpeaking = false;
                elements.visualizer.classList.remove('speaking');
                elements.micStatus.textContent = 'Clicca per parlare';
            };

            // Mostra che l'assistente sta parlando
            elements.visualizer.classList.add('speaking');
            elements.micStatus.textContent = '🤖 Assistente sta parlando...';
        }
    });

    state.room.on(LivekitClient.RoomEvent.TrackUnsubscribed, (track) => {
        if (track.kind === LivekitClient.Track.Kind.Audio) {
            track.detach().forEach(el => el.remove());
            elements.visualizer.classList.remove('speaking');
            elements.micStatus.textContent = 'Clicca per parlare';
        }
    });

    state.room.on(LivekitClient.RoomEvent.DataReceived, (data, participant) => {
        try {
            const message = JSON.parse(new TextDecoder().decode(data));
            handleDataMessage(message, participant);
        } catch (e) {
            console.error('Errore parsing messaggio:', e);
        }
    });
    
    // Listener per trascrizioni (text streams)
    state.room.on(LivekitClient.RoomEvent.TranscriptionReceived, (segments, participant) => {
        console.log('Trascrizione ricevuta:', segments);
        segments.forEach(segment => {
            if (segment.text && segment.final) {
                // Determina se è l'utente o l'assistente
                const isUser = participant && participant.identity === state.room.localParticipant.identity;
                addToTranscript(segment.text, isUser ? 'user' : 'assistant');
            }
        });
    });
}

/**
 * Toggle microfono
 */
async function toggleMic() {
    if (!state.connected) {
        alert('Connettiti prima!');
        return;
    }

    if (state.micActive) {
        // Disattiva microfono
        if (state.localAudioTrack) {
            await state.room.localParticipant.unpublishTrack(state.localAudioTrack);
            state.localAudioTrack.stop();
            state.localAudioTrack = null;
        }

        state.micActive = false;
        elements.visualizer.classList.remove('active');
        elements.micStatus.textContent = 'Clicca per parlare';

    } else {
        // Attiva microfono
        try {
            state.localAudioTrack = await LivekitClient.createLocalAudioTrack({
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            });

            await state.room.localParticipant.publishTrack(state.localAudioTrack);

            state.micActive = true;
            elements.visualizer.classList.add('active');
            elements.micStatus.textContent = '🎤 Sto ascoltando...';

        } catch (error) {
            console.error('Errore attivazione microfono:', error);
            alert('Errore accesso al microfono: ' + error.message);
        }
    }
}

/**
 * Interrompe l'audio dell'agent e prepara per nuova richiesta
 */
async function interruptAndSpeak() {
    if (!state.connected) {
        alert('Connettiti prima!');
        return;
    }
    
    console.log('Interrompo audio e preparo nuova richiesta...');
    
    // Interrompi tutti gli audio dell'agent
    state.agentAudioElements.forEach(audioEl => {
        audioEl.pause();
        audioEl.currentTime = 0;
        audioEl.remove();
    });
    state.agentAudioElements = [];
    state.isSpeaking = false;
    
    // Rimuovi stato "speaking"
    elements.visualizer.classList.remove('speaking');
    
    // Invia comando di interruzione all'agent
    if (state.room && state.connected) {
        const data = new TextEncoder().encode(JSON.stringify({
            type: 'interrupt'
        }));
        state.room.localParticipant.publishData(data, { reliable: true });
    }
    
    // Attiva il microfono se non è già attivo
    if (!state.micActive) {
        await toggleMic();
    }
    
    elements.micStatus.textContent = '🎤 Parla ora...';
    
    // Aggiungi nota alla trascrizione
    addToTranscript('[Interruzione richiesta]', 'system');
}

/**
 * Seleziona TTS engine
 */
function selectTTS(engine) {
    state.selectedTTS = engine;

    // Aggiorna UI
    document.querySelectorAll('.tts-option').forEach(option => {
        option.classList.remove('selected');
    });
    document.querySelector(`[data-engine="${engine}"]`).classList.add('selected');

    // Invia al server se connesso
    if (state.connected) {
        sendTTSSelection();
    }
}

/**
 * Invia selezione TTS all'agent
 */
function sendTTSSelection() {
    if (state.room && state.connected) {
        const data = new TextEncoder().encode(JSON.stringify({
            type: 'set_tts',
            engine: state.selectedTTS
        }));

        state.room.localParticipant.publishData(data, { reliable: true });
        console.log('TTS selezionato:', state.selectedTTS);
    }
}

/**
 * Gestisce messaggi ricevuti
 */
function handleDataMessage(message, participant) {
    if (message.type === 'transcript') {
        addToTranscript(message.text, message.role);
    } else if (message.type === 'status') {
        elements.micStatus.textContent = message.text;
    }
}

/**
 * Aggiunge testo alla trascrizione con stile conversazione
 */
function addToTranscript(text, role) {
    // Rimuovi placeholder se presente
    const emptyMessage = elements.transcript.querySelector('.transcript-empty');
    if (emptyMessage) {
        emptyMessage.remove();
    }

    const message = document.createElement('div');
    message.className = `message ${role}`;
    
    // Determina avatar e etichetta in base al ruolo
    let avatar = '👤';
    let label = 'Tu';
    
    if (role === 'assistant') {
        avatar = '🤖';
        label = 'Assistente';
    } else if (role === 'system') {
        avatar = 'ℹ️';
        label = 'Sistema';
    }
    
    message.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <div class="message-role">${label}</div>
            <div class="message-text">${text}</div>
        </div>
    `;

    elements.transcript.appendChild(message);
    elements.transcript.scrollTop = elements.transcript.scrollHeight;
}

/**
 * Aggiorna stato connessione UI
 */
function updateStatus(status, text) {
    elements.statusDot.className = 'status-dot ' + status;
    elements.statusText.textContent = text;

    if (status === 'connected') {
        elements.connectBtn.style.display = 'none';
        elements.disconnectBtn.style.display = 'flex';
    } else {
        elements.connectBtn.style.display = 'flex';
        elements.disconnectBtn.style.display = 'none';
    }
}

/**
 * Mostra interfaccia vocale
 */
function showVoiceInterface() {
    elements.voiceCard.style.display = 'block';
    elements.transcriptCard.style.display = 'block';
}

/**
 * Nascondi interfaccia vocale
 */
function hideVoiceInterface() {
    elements.voiceCard.style.display = 'none';
    elements.transcriptCard.style.display = 'none';
}

/**
 * Verifica lo stato dei servizi
 */
async function checkServicesStatus() {
    const statusContainer = document.getElementById('services-status');
    if (!statusContainer) return;
    
    statusContainer.innerHTML = '<div class="status-loading">⏳ Verifica servizi in corso...</div>';
    
    try {
        const response = await fetch('/api/status');
        if (!response.ok) {
            throw new Error('Errore verifica stato');
        }
        
        const status = await response.json();
        console.log('Stato servizi:', status);
        
        let html = '<div class="services-grid">';
        
        // LiveKit
        html += `<div class="service-item ${status.livekit.available ? 'available' : 'unavailable'}">
            <span class="service-icon">${status.livekit.available ? '✅' : '❌'}</span>
            <span class="service-name">LiveKit Server</span>
            <span class="service-message">${status.livekit.message}</span>
        </div>`;
        
        // Ollama
        html += `<div class="service-item ${status.ollama.available ? 'available' : 'unavailable'}">
            <span class="service-icon">${status.ollama.available ? '✅' : '❌'}</span>
            <span class="service-name">Ollama LLM</span>
            <span class="service-message">${status.ollama.message}</span>
        </div>`;
        
        // Agent
        html += `<div class="service-item ${status.agent.available ? 'available' : 'unavailable'}">
            <span class="service-icon">${status.agent.available ? '✅' : '❌'}</span>
            <span class="service-name">Voice Agent</span>
            <span class="service-message">${status.agent.message}</span>
        </div>`;
        
        // Whisper STT
        if (status.whisper) {
            html += `<div class="service-item ${status.whisper.available ? 'available' : 'unavailable'}">
                <span class="service-icon">${status.whisper.available ? '🎤' : '❌'}</span>
                <span class="service-name">Whisper STT</span>
                <span class="service-message">${status.whisper.message}</span>
            </div>`;
        }
        
        html += '</div>';
        
        // Messaggio generale
        if (status.all_ready) {
            html += '<div class="status-ready">🎉 Tutti i servizi sono pronti!</div>';
            elements.connectBtn.disabled = false;
        } else {
            html += '<div class="status-not-ready">⚠️ Alcuni servizi non sono disponibili. Attendi e riprova.</div>';
            html += '<button onclick="checkServicesStatus()" class="btn-refresh">🔄 Ricontrolla</button>';
            elements.connectBtn.disabled = true;
        }
        
        statusContainer.innerHTML = html;
        
        return status.all_ready;
        
    } catch (e) {
        console.error('Errore verifica servizi:', e);
        statusContainer.innerHTML = `
            <div class="status-error">❌ Errore: ${e.message}</div>
            <button onclick="checkServicesStatus()" class="btn-refresh">🔄 Riprova</button>
        `;
        elements.connectBtn.disabled = true;
        return false;
    }
}

/**
 * Inizializzazione
 */
async function init() {
    console.log('Voice Agent inizializzato');

    // Verifica stato servizi
    await checkServicesStatus();
    
    // Aggiorna stato ogni 30 secondi
    setInterval(checkServicesStatus, 30000);

    // Carica configurazione
    try {
        const configResponse = await fetch('/api/config');
        if (configResponse.ok) {
            const config = await configResponse.json();
            console.log('Configurazione:', config);

            // Seleziona TTS default
            if (config.default_tts) {
                selectTTS(config.default_tts);
            }
        }
    } catch (e) {
        console.warn('Impossibile caricare configurazione:', e);
    }
}

// Avvia al caricamento
window.addEventListener('DOMContentLoaded', init);

// Gestisci chiusura pagina
window.addEventListener('beforeunload', () => {
    if (state.connected) {
        disconnect();
    }
});

