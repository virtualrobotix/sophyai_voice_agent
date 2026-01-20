-- Voice Agent Database Schema
-- PostgreSQL 16

-- Settings table: key-value store for configuration
CREATE TABLE IF NOT EXISTS settings (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Chats table: conversation sessions
CREATE TABLE IF NOT EXISTS chats (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL DEFAULT 'Nuova Chat',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Messages table: individual messages in chats
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster message retrieval by chat
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

-- Default settings
INSERT INTO settings (key, value) VALUES 
    ('llm_provider', 'ollama'),
    ('ollama_model', 'gpt-oss'),
    ('openrouter_model', ''),
    ('openrouter_api_key', ''),
    ('whisper_model', 'medium'),
    ('whisper_language', 'it'),
    ('whisper_auto_detect', 'false'),
    ('tts_engine', 'edge'),
    ('tts_language', 'it'),
    ('system_prompt', 'Sei Sophy, assistente vocale ultra-veloce. PRIORITA ASSOLUTA: VELOCITA E SINTESI.

REGOLE FONDAMENTALI:
1. RISPOSTE ULTRA-BREVI: massimo 1-2 frasi, mai piu di 30 parole
2. VAI DRITTO AL PUNTO: niente preamboli, saluti inutili o ripetizioni
3. LINGUA: rispondi nella stessa lingua dell utente

STILE:
- Rispondi come un amico esperto: diretto, chiaro, utile
- Se non sai qualcosa, dillo in 5 parole
- Preferisci risposte secche e precise

FORMATO TTS:
- NO simboli: * # @ $ % & / | < > { } [ ] ~ ^ `
- NO emoji
- Numeri in parole (ventitre, non 23)
- NO elenchi puntati, scrivi discorsivo'),
    ('context_injection', ''),
    ('elevenlabs_api_key', ''),
    ('elevenlabs_model', 'eleven_multilingual_v2'),
    ('elevenlabs_voice', ''),
    ('elevenlabs_stability', '50'),
    ('elevenlabs_similarity', '75'),
    ('elevenlabs_style', '0'),
    ('elevenlabs_boost', 'false'),
    -- Remote LLM Server settings
    ('remote_server_url', ''),
    ('remote_server_token', ''),
    ('remote_server_collection', ''),
    -- Voice Activation settings
    ('wake_timeout_seconds', '20'),
    ('vad_energy_threshold', '40'),
    ('speech_energy_threshold', '100'),
    ('silence_threshold', '30'),
    ('tts_cooldown_seconds', '5')
ON CONFLICT (key) DO NOTHING;

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
DROP TRIGGER IF EXISTS update_settings_updated_at ON settings;
CREATE TRIGGER update_settings_updated_at
    BEFORE UPDATE ON settings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_chats_updated_at ON chats;
CREATE TRIGGER update_chats_updated_at
    BEFORE UPDATE ON chats
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();





