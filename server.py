"""
Web Server
Server FastAPI per il frontend e le API.
Supporta HTTPS per l'accesso al microfono.
"""

import asyncio
import os
import sys
import subprocess
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from loguru import logger
from livekit import api
import httpx

# Aggiungi il path del progetto
sys.path.insert(0, str(Path(__file__).parent))

from agent.config import config

# Configura logging
logger.remove()
logger.add(
    sys.stderr,
    level=config.server.log_level,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"
)

# Crea app FastAPI
app = FastAPI(
    title="Voice Agent API",
    description="API per il Voice Agent WebRTC",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TokenRequest(BaseModel):
    """Richiesta token LiveKit"""
    room_name: str
    participant_name: str


class TokenResponse(BaseModel):
    """Risposta con token LiveKit"""
    token: str
    url: str


class TTSEngineRequest(BaseModel):
    """Richiesta cambio TTS engine"""
    engine: str


# API Endpoints
@app.get("/")
async def root():
    """Serve la pagina principale"""
    return FileResponse("web/index.html")


@app.get("/debug.html")
async def debug_page():
    """Serve la pagina di debug e impostazioni"""
    return FileResponse("web/debug.html")


@app.get("/api/health")
async def health():
    """Health check"""
    return {"status": "ok", "service": "voice-agent", "https": True}


@app.get("/api/status")
async def get_status():
    """Verifica lo stato di tutti i servizi"""
    import aiohttp
    import asyncio
    
    status = {
        "livekit": {"available": False, "message": "Non connesso"},
        "ollama": {"available": False, "message": "Non connesso"},
        "agent": {"available": False, "message": "Non connesso"},
        "sip": {"available": False, "message": "Non configurato"},
        "all_ready": False
    }
    
    # Verifica LiveKit
    try:
        internal_url = os.getenv("LIVEKIT_INTERNAL_URL", "ws://host.docker.internal:7880")
        lk_api = api.LiveKitAPI(
            url=internal_url,
            api_key=config.livekit.api_key,
            api_secret=config.livekit.api_secret
        )
        # Prova a listare le room
        rooms = await lk_api.room.list_rooms(api.ListRoomsRequest())
        status["livekit"] = {"available": True, "message": f"Connesso ({len(rooms.rooms)} room attive)"}
        await lk_api.aclose()
    except Exception as e:
        status["livekit"] = {"available": False, "message": str(e)[:100]}
    
    # Verifica Ollama
    try:
        async with aiohttp.ClientSession() as session:
            # Usa host.docker.internal per connessioni da Docker
            ollama_url = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
            async with session.get(f"{ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    if config.ollama.model in str(models):
                        status["ollama"] = {"available": True, "message": f"Modello {config.ollama.model} disponibile"}
                    else:
                        status["ollama"] = {"available": True, "message": f"Connesso, modelli: {', '.join(models[:3])}"}
                else:
                    status["ollama"] = {"available": False, "message": f"Errore HTTP {resp.status}"}
    except asyncio.TimeoutError:
        status["ollama"] = {"available": False, "message": "Timeout connessione"}
    except Exception as e:
        status["ollama"] = {"available": False, "message": str(e)[:100]}
    
    # Verifica Agent Worker (controlla se il container agent è attivo)
    try:
        async with aiohttp.ClientSession() as session:
            # L'agent LiveKit espone "/" che risponde "OK"
            agent_url = "http://voice-agent-worker:8081/"
            async with session.get(agent_url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    status["agent"] = {"available": True, "message": "Worker attivo e in ascolto"}
                else:
                    status["agent"] = {"available": False, "message": f"Worker risponde con errore {resp.status}"}
    except asyncio.TimeoutError:
        status["agent"] = {"available": False, "message": "Timeout connessione al worker"}
    except Exception as e:
        status["agent"] = {"available": False, "message": str(e)[:100]}
    
    # Verifica Whisper (modello STT)
    whisper_model = os.getenv("WHISPER_MODEL", "tiny")
    hf_home = os.getenv("HF_HOME", "/app/models/huggingface")
    whisper_cache_path = Path(hf_home) / "hub"
    
    # Cerca directory del modello whisper
    model_found = False
    model_dir = None
    if whisper_cache_path.exists():
        for d in whisper_cache_path.iterdir():
            if d.is_dir() and "whisper" in d.name.lower() and whisper_model in d.name.lower():
                model_found = True
                model_dir = d.name
                break
    
    if model_found:
        status["whisper"] = {"available": True, "message": f"Modello '{whisper_model}' scaricato e pronto"}
    else:
        # Il modello verrà scaricato al primo utilizzo
        status["whisper"] = {"available": True, "message": f"Modello '{whisper_model}' (download al primo uso)"}
    
    # Verifica SIP Bridge (opzionale)
    try:
        async with aiohttp.ClientSession() as session:
            sip_url = "http://livekit-sip:8080/health"
            async with session.get(sip_url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status == 200:
                    status["sip"] = {"available": True, "message": "SIP Bridge attivo (porta 5060)"}
                else:
                    status["sip"] = {"available": False, "message": f"SIP Bridge errore {resp.status}"}
    except asyncio.TimeoutError:
        status["sip"] = {"available": False, "message": "SIP Bridge non raggiungibile"}
    except Exception as e:
        # SIP è opzionale, non mostrare errore se non è avviato
        status["sip"] = {"available": False, "message": "SIP Bridge non avviato (opzionale)"}
    
    # Verifica se tutti i servizi sono pronti (SIP è opzionale)
    status["all_ready"] = all([
        status["livekit"]["available"],
        status["ollama"]["available"],
        status["agent"]["available"],
        status["whisper"]["available"]
    ])
    
    return status


# Timing stats storage (in-memory, reset on restart)
_timing_stats = {
    "stt": {"time_ms": 0, "count": 0},
    "llm": {"time_ms": 0, "ttft_ms": 0, "count": 0},
    "tts": {"time_ms": 0, "audio_sec": 0, "count": 0},
    "latency": {"e2e_ms": 0, "to_first_audio_ms": 0, "count": 0}  # Latenza end-to-end
}


@app.get("/api/timing")
async def get_timing():
    """Restituisce le statistiche di timing delle ultime operazioni"""
    return _timing_stats


@app.post("/api/timing")
async def update_timing(data: dict):
    """Aggiorna le statistiche di timing (chiamato dall'agent)"""
    global _timing_stats
    
    if "stt" in data:
        _timing_stats["stt"] = {
            "time_ms": data["stt"].get("time_ms", 0),
            "count": _timing_stats["stt"]["count"] + 1
        }
    
    if "llm" in data:
        _timing_stats["llm"] = {
            "time_ms": data["llm"].get("time_ms", 0),
            "ttft_ms": data["llm"].get("ttft_ms", 0),
            "count": _timing_stats["llm"]["count"] + 1
        }
    
    if "tts" in data:
        _timing_stats["tts"] = {
            "time_ms": data["tts"].get("time_ms", 0),
            "audio_sec": data["tts"].get("audio_sec", 0),
            "count": _timing_stats["tts"]["count"] + 1
        }
    
    if "latency" in data:
        _timing_stats["latency"] = {
            "e2e_ms": data["latency"].get("e2e_ms", 0),
            "to_first_audio_ms": data["latency"].get("to_first_audio_ms", 0),
            "count": _timing_stats["latency"]["count"] + 1
        }
    
    return {"status": "ok"}


@app.post("/api/timing/reset")
async def reset_timing():
    """Resetta tutte le statistiche di timing"""
    global _timing_stats
    _timing_stats = {
        "stt": {"time_ms": 0, "count": 0},
        "llm": {"time_ms": 0, "ttft_ms": 0, "count": 0},
        "tts": {"time_ms": 0, "audio_sec": 0, "count": 0},
        "latency": {"e2e_ms": 0, "to_first_audio_ms": 0, "count": 0}
    }
    return {"status": "ok", "message": "Stats reset"}


# ==================== Database Connection ====================
_db = None

async def get_database():
    """Get database instance, initialize if needed."""
    global _db
    if _db is None:
        try:
            from db.database import get_db
            _db = await get_db()
        except Exception as e:
            logger.warning(f"Database non disponibile: {e}")
            return None
    return _db


@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    try:
        await get_database()
        logger.info("Database connesso")
    except Exception as e:
        logger.warning(f"Database non disponibile all'avvio: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    global _db
    if _db:
        try:
            from db.database import close_db
            await close_db()
        except:
            pass


# ==================== Settings API ====================

class SettingsUpdate(BaseModel):
    """Update settings request."""
    settings: dict


@app.get("/api/settings")
async def get_settings():
    """Get all settings from database."""
    db = await get_database()
    if db is None:
        # Fallback to defaults
        return {
            "llm_provider": "ollama",
            "ollama_model": os.getenv("OLLAMA_MODEL", "gpt-oss"),
            "openrouter_model": "",
            "openrouter_api_key": "",
            "whisper_model": os.getenv("WHISPER_MODEL", "medium"),
            "whisper_language": "it",
            "whisper_auto_detect": "false",
            "tts_engine": os.getenv("DEFAULT_TTS", "edge"),
            "tts_language": "it",
            "system_prompt": "",
            "context_injection": ""
        }
    
    try:
        settings = await db.get_all_settings()
        return settings
    except Exception as e:
        logger.error(f"Errore lettura settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/settings")
async def update_settings(update: SettingsUpdate):
    """Update multiple settings."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        await db.set_multiple_settings(update.settings)
        return {"status": "ok", "updated": list(update.settings.keys())}
    except Exception as e:
        logger.error(f"Errore aggiornamento settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/settings/{key}")
async def get_setting(key: str):
    """Get a single setting."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        value = await db.get_setting(key)
        if value is None:
            raise HTTPException(status_code=404, detail=f"Setting '{key}' non trovato")
        return {"key": key, "value": value}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Errore lettura setting {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SettingValue(BaseModel):
    """Single setting value."""
    value: str


@app.put("/api/settings/{key}")
async def set_setting(key: str, setting: SettingValue):
    """Set a single setting."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        await db.set_setting(key, setting.value)
        return {"status": "ok", "key": key}
    except Exception as e:
        logger.error(f"Errore salvataggio setting {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Ollama API ====================

@app.get("/api/ollama/models")
async def get_ollama_models():
    """Get list of available Ollama models."""
    import aiohttp
    
    ollama_url = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{ollama_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    raise HTTPException(status_code=resp.status, detail="Ollama non raggiungibile")
                
                data = await resp.json()
                models = []
                for m in data.get("models", []):
                    models.append({
                        "id": m["name"],
                        "name": m["name"],
                        "size": m.get("size", 0),
                        "modified_at": m.get("modified_at", ""),
                        "details": m.get("details", {})
                    })
                
                return {"models": models, "host": ollama_url}
    except aiohttp.ClientError as e:
        logger.error(f"Errore connessione Ollama: {e}")
        raise HTTPException(status_code=503, detail=f"Ollama non raggiungibile: {e}")


class OllamaSelectRequest(BaseModel):
    """Select Ollama model request."""
    model: str


@app.post("/api/ollama/select")
async def select_ollama_model(request: OllamaSelectRequest):
    """Select an Ollama model and save to settings."""
    db = await get_database()
    
    if db:
        try:
            await db.set_setting("llm_provider", "ollama")
            await db.set_setting("ollama_model", request.model)
        except Exception as e:
            logger.warning(f"Errore salvataggio in DB: {e}")
    
    # Aggiorna anche la variabile d'ambiente per l'agent
    os.environ["OLLAMA_MODEL"] = request.model
    
    return {"status": "ok", "model": request.model, "provider": "ollama"}


# ==================== OpenRouter API ====================

@app.get("/api/openrouter/models")
async def get_openrouter_models(search: str = None, sort_by: str = "name"):
    """Get list of available OpenRouter models."""
    db = await get_database()
    api_key = None
    
    if db:
        try:
            api_key = await db.get_setting("openrouter_api_key")
        except:
            pass
    
    try:
        async with httpx.AsyncClient() as client:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers=headers,
                timeout=30.0
            )
            
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail="OpenRouter API error")
            
            data = resp.json()
            models = []
            
            for m in data.get("data", []):
                pricing = m.get("pricing", {})
                prompt_cost = float(pricing.get("prompt", 0)) * 1000000  # Per 1M tokens
                completion_cost = float(pricing.get("completion", 0)) * 1000000
                
                model_info = {
                    "id": m["id"],
                    "name": m.get("name", m["id"]),
                    "description": m.get("description", ""),
                    "context_length": m.get("context_length", 0),
                    "prompt_cost": prompt_cost,
                    "completion_cost": completion_cost,
                    "total_cost": prompt_cost + completion_cost,
                    "top_provider": m.get("top_provider", {}).get("max_completion_tokens"),
                }
                
                # Filtro per ricerca
                if search:
                    search_lower = search.lower()
                    if search_lower not in model_info["id"].lower() and search_lower not in model_info["name"].lower():
                        continue
                
                models.append(model_info)
            
            # Filtro gratuiti o ordinamento
            if sort_by == "free":
                models = [m for m in models if m["total_cost"] == 0]
                models.sort(key=lambda x: x["name"].lower())
            elif sort_by == "cost":
                models.sort(key=lambda x: x["total_cost"])
            elif sort_by == "cost_desc":
                models.sort(key=lambda x: x["total_cost"], reverse=True)
            elif sort_by == "context":
                models.sort(key=lambda x: x["context_length"], reverse=True)
            else:
                models.sort(key=lambda x: x["name"].lower())
            
            return {"models": models, "count": len(models)}
    
    except httpx.HTTPError as e:
        logger.error(f"Errore OpenRouter API: {e}")
        raise HTTPException(status_code=503, detail=f"OpenRouter non raggiungibile: {e}")


class OpenRouterKeyRequest(BaseModel):
    """Save OpenRouter API key request."""
    api_key: str


@app.post("/api/openrouter/key")
async def save_openrouter_key(request: OpenRouterKeyRequest):
    """Save OpenRouter API key to database."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        await db.set_setting("openrouter_api_key", request.api_key)
        return {"status": "ok", "message": "API key salvata"}
    except Exception as e:
        logger.error(f"Errore salvataggio API key: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class OpenRouterSelectRequest(BaseModel):
    """Select OpenRouter model request."""
    model: str


@app.post("/api/openrouter/select")
async def select_openrouter_model(request: OpenRouterSelectRequest):
    """Select an OpenRouter model and save to settings."""
    db = await get_database()
    
    if db:
        try:
            await db.set_setting("llm_provider", "openrouter")
            await db.set_setting("openrouter_model", request.model)
        except Exception as e:
            logger.warning(f"Errore salvataggio in DB: {e}")
    
    return {"status": "ok", "model": request.model, "provider": "openrouter"}


# ==================== ElevenLabs API ====================

@app.get("/api/elevenlabs/voices")
async def get_elevenlabs_voices():
    """Get available ElevenLabs voices."""
    db = await get_database()
    api_key = None
    
    if db:
        try:
            api_key = await db.get_setting("elevenlabs_api_key")
        except Exception as e:
            logger.warning(f"Errore lettura API key: {e}")
    
    if not api_key:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
    
    if not api_key:
        return {"error": "API Key ElevenLabs non configurata", "voices": []}
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": api_key}
            )
            
            if resp.status_code == 401:
                return {"error": "API Key non valida", "voices": []}
            
            resp.raise_for_status()
            data = resp.json()
            
            voices = []
            for v in data.get("voices", []):
                voices.append({
                    "voice_id": v.get("voice_id"),
                    "name": v.get("name"),
                    "category": v.get("category"),
                    "labels": v.get("labels", {}),
                    "preview_url": v.get("preview_url"),
                    "description": v.get("description")
                })
            
            return {"voices": voices}
            
    except httpx.HTTPError as e:
        logger.error(f"Errore ElevenLabs API: {e}")
        return {"error": f"Errore API: {str(e)}", "voices": []}


@app.get("/api/elevenlabs/models")
async def get_elevenlabs_models():
    """Get available ElevenLabs models."""
    models = [
        {"id": "eleven_multilingual_v2", "name": "Multilingual v2", "description": "Migliore qualità, multilingua"},
        {"id": "eleven_turbo_v2_5", "name": "Turbo v2.5", "description": "Veloce, bassa latenza"},
        {"id": "eleven_turbo_v2", "name": "Turbo v2", "description": "Veloce, bassa latenza"},
        {"id": "eleven_monolingual_v1", "name": "Monolingual v1", "description": "Solo inglese"},
        {"id": "eleven_flash_v2_5", "name": "Flash v2.5", "description": "Ultra veloce, streaming"},
        {"id": "eleven_flash_v2", "name": "Flash v2", "description": "Ultra veloce, streaming"}
    ]
    return {"models": models}


# ==================== Chat API ====================

class ChatCreateRequest(BaseModel):
    """Create chat request."""
    title: str = "Nuova Chat"


class MessageRequest(BaseModel):
    """Add message request."""
    role: str
    content: str


@app.get("/api/chats")
async def get_chats():
    """Get all chats."""
    db = await get_database()
    if db is None:
        return {"chats": []}
    
    try:
        chats = await db.get_chats()
        return {"chats": chats}
    except Exception as e:
        logger.error(f"Errore lettura chats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chats")
async def create_chat(request: ChatCreateRequest):
    """Create a new chat."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        chat_id = await db.create_chat(request.title)
        return {"status": "ok", "id": chat_id, "title": request.title}
    except Exception as e:
        logger.error(f"Errore creazione chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: int):
    """Get a chat with its messages."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        chat = await db.get_chat(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat non trovata")
        
        messages = await db.get_messages(chat_id)
        chat["messages"] = messages
        return chat
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Errore lettura chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: int):
    """Delete a chat and all its messages."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        deleted = await db.delete_chat(chat_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Chat non trovata")
        return {"status": "ok", "deleted": chat_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Errore eliminazione chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chats/{chat_id}/messages")
async def add_message(chat_id: int, request: MessageRequest):
    """Add a message to a chat."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        message_id = await db.add_message(chat_id, request.role, request.content)
        return {"status": "ok", "id": message_id, "chat_id": chat_id}
    except Exception as e:
        logger.error(f"Errore aggiunta messaggio a chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Prompt/Context API ====================

class PromptUpdate(BaseModel):
    """Update system prompt."""
    prompt: str


@app.get("/api/prompt")
async def get_prompt():
    """Get current system prompt."""
    db = await get_database()
    
    default_prompt = """Sei Sophy, assistente vocale ultra-veloce. PRIORITA ASSOLUTA: VELOCITA E SINTESI.

REGOLE FONDAMENTALI:
1. RISPOSTE ULTRA-BREVI: massimo 1-2 frasi, mai piu di 30 parole
2. VAI DRITTO AL PUNTO: niente preamboli, saluti inutili o ripetizioni
3. LINGUA: rispondi nella stessa lingua dell utente"""
    
    if db is None:
        return {"prompt": default_prompt}
    
    try:
        prompt = await db.get_setting("system_prompt")
        return {"prompt": prompt or default_prompt}
    except Exception as e:
        logger.error(f"Errore lettura prompt: {e}")
        return {"prompt": default_prompt}


@app.post("/api/prompt")
async def update_prompt(request: PromptUpdate):
    """Update system prompt."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        await db.set_setting("system_prompt", request.prompt)
        return {"status": "ok", "message": "Prompt aggiornato"}
    except Exception as e:
        logger.error(f"Errore salvataggio prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class ContextUpdate(BaseModel):
    """Update context injection."""
    context: str


@app.get("/api/context")
async def get_context():
    """Get current context injection."""
    db = await get_database()
    
    if db is None:
        return {"context": ""}
    
    try:
        context = await db.get_setting("context_injection")
        return {"context": context or ""}
    except Exception as e:
        logger.error(f"Errore lettura context: {e}")
        return {"context": ""}


@app.post("/api/context")
async def update_context(request: ContextUpdate):
    """Update context injection."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        await db.set_setting("context_injection", request.context)
        return {"status": "ok", "message": "Context aggiornato"}
    except Exception as e:
        logger.error(f"Errore salvataggio context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/token", response_model=TokenResponse)
async def get_token(request: TokenRequest):
    """
    Genera un token LiveKit per un partecipante e dispatcha l'agent.
    """
    try:
        # Crea token
        token = api.AccessToken(
            config.livekit.api_key,
            config.livekit.api_secret
        )
        
        token.with_identity(request.participant_name)
        token.with_name(request.participant_name)
        
        # Grants - aggiungo room_create per creare la room automaticamente
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=request.room_name,
            room_create=True,  # Permette di creare la room
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True
        ))
        
        jwt_token = token.to_jwt()
        
        # URL WebSocket per il client
        ws_url = config.livekit.url
        
        # Dispatcha l'agent nella room
        # Usa host.docker.internal per le chiamate API interne da Docker
        import os
        internal_url = os.getenv("LIVEKIT_INTERNAL_URL", "ws://host.docker.internal:7880")
        try:
            lk_api = api.LiveKitAPI(
                url=internal_url,
                api_key=config.livekit.api_key,
                api_secret=config.livekit.api_secret
            )
            
            # Crea la room se non esiste
            await lk_api.room.create_room(
                api.CreateRoomRequest(name=request.room_name)
            )
            
            # Dispatcha l'agent
            await lk_api.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    room=request.room_name,
                    agent_name=""  # Agent di default
                )
            )
            logger.info(f"Agent dispatchato per room {request.room_name}")
            
            await lk_api.aclose()
        except Exception as dispatch_err:
            logger.warning(f"Agent dispatch fallito (potrebbe essere già attivo): {dispatch_err}")
        
        logger.info(f"Token generato per {request.participant_name} in room {request.room_name}")
        
        return TokenResponse(token=jwt_token, url=ws_url)
        
    except Exception as e:
        logger.error(f"Errore generazione token: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tts/engines")
async def get_tts_engines():
    """Ritorna la lista dei TTS engine disponibili"""
    return {
        "engines": [
            {
                "id": "piper",
                "name": "Piper TTS",
                "self_hosted": True,
                "description": "Veloce e leggero, ottimo supporto italiano"
            },
            {
                "id": "coqui",
                "name": "Coqui TTS",
                "self_hosted": True,
                "description": "Alta qualità, richiede più risorse"
            },
            {
                "id": "edge",
                "name": "Edge TTS (Microsoft)",
                "self_hosted": False,
                "description": "Qualità eccellente, richiede internet"
            },
            {
                "id": "kokoro",
                "name": "Kokoro 82M",
                "self_hosted": True,
                "description": "Multilingua, alta qualità"
            },
            {
                "id": "vibevoice",
                "name": "VibeVoice (Microsoft)",
                "self_hosted": True,
                "description": "Espressivo, multi-speaker, real-time streaming"
            }
        ],
        "default": config.tts.default_engine
    }


@app.get("/api/tts/{engine}/voices")
async def get_tts_voices(engine: str):
    """Ritorna le voci disponibili per un TTS engine"""
    try:
        from agent.tts import get_tts_engine
        tts = get_tts_engine(engine)
        voices = tts.get_available_voices()
        return {"engine": engine, "voices": voices}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Errore recupero voci: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Stato TTS corrente (può essere cambiato dinamicamente)
_current_tts_config = {
    "engine": None,  # Sarà impostato dal default o dalla selezione utente
    "language": "it",
    "voice": None,
    "last_updated": None
}

@app.get("/api/config")
async def get_config():
    """Ritorna la configurazione pubblica"""
    return {
        "livekit_url": config.livekit.url,
        "default_tts": config.tts.default_engine,
        "whisper_model": config.whisper.model,
        "ollama_model": config.ollama.model
    }


@app.get("/api/tts/current")
async def get_current_tts():
    """Ritorna il TTS attualmente in uso (legge dal file se esiste)"""
    import json
    
    # Prima prova a leggere dal file di configurazione
    config_path = "/app/config/tts_config.json"
    file_config = None
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                file_config = json.load(f)
        except:
            pass
    
    if file_config:
        return {
            "engine": file_config.get("engine", "edge"),
            "language": file_config.get("language", "it"),
            "voice": file_config.get("voice"),
            "default_engine": config.tts.default_engine,
            "last_updated": file_config.get("updated_at"),
            "is_default": False,
            "source": "file"
        }
    
    # Fallback alla configurazione in memoria o default
    engine = _current_tts_config["engine"] or config.tts.default_engine
    
    return {
        "engine": engine,
        "language": _current_tts_config["language"],
        "voice": _current_tts_config["voice"],
        "default_engine": config.tts.default_engine,
        "last_updated": _current_tts_config["last_updated"],
        "is_default": _current_tts_config["engine"] is None,
        "source": "memory" if _current_tts_config["engine"] else "default"
    }


class TTSConfigUpdate(BaseModel):
    """Aggiornamento configurazione TTS"""
    engine: str
    language: str = "it"
    voice: str = None
    speaker: str = None
    speed: float = 1.0
    # Parametri Chatterbox
    model: str = None  # "standard" o "multilingual"
    device: str = None  # "auto", "cuda", "cpu", "mps"
    exaggeration: float = None  # 0.0-1.0
    audio_prompt_path: str = None  # Path per voice cloning


@app.post("/api/tts/current")
async def set_current_tts(update: TTSConfigUpdate):
    """Imposta il TTS da usare e salva su file condiviso"""
    import datetime
    import json
    
    _current_tts_config["engine"] = update.engine
    _current_tts_config["language"] = update.language
    _current_tts_config["voice"] = update.voice
    _current_tts_config["last_updated"] = datetime.datetime.now().isoformat()
    
    # Salva configurazione su file condiviso per l'agent
    config_path = "/app/config/tts_config.json"
    os.makedirs("/app/config", exist_ok=True)
    
    tts_file_config = {
        "engine": update.engine,
        "language": update.language,
        "voice": update.voice,
        "speaker": update.speaker,
        "speed": update.speed,
        "updated_at": _current_tts_config["last_updated"]
    }
    
    # Aggiungi parametri Chatterbox se presenti
    if update.model is not None:
        tts_file_config["model"] = update.model
    if update.device is not None:
        tts_file_config["device"] = update.device
    if update.exaggeration is not None:
        tts_file_config["exaggeration"] = update.exaggeration
    if update.audio_prompt_path is not None:
        tts_file_config["audio_prompt_path"] = update.audio_prompt_path
    
    with open(config_path, "w") as f:
        json.dump(tts_file_config, f, indent=2)
    
    logger.info(f"🔊 TTS aggiornato e salvato: engine={update.engine}, language={update.language}")
    logger.info(f"📁 Config salvata in: {config_path}")
    
    return {
        "status": "ok",
        "engine": update.engine,
        "language": update.language,
        "saved_to_file": True,
        "message": f"TTS {update.engine} configurato. Riavvia l'agent per applicare."
    }


class TTSTestRequest(BaseModel):
    """Richiesta per test TTS"""
    engine: str
    text: str
    language: str = "it"
    voice: str = None
    speaker: str = None
    speed: float = 1.0
    # Parametri specifici
    model: str = None  # Per Chatterbox/VibeVoice
    device: str = None  # Per Chatterbox
    exaggeration: float = None  # Per Chatterbox
    audio_prompt_path: str = None  # Per Chatterbox voice cloning


async def test_tts_via_external_server(request: TTSTestRequest, tts_server_url: str):
    """Chiama il server TTS esterno (Mac host) per sintetizzare"""
    import aiohttp
    import numpy as np
    import soundfile as sf
    import io
    
    try:
        # Prepara payload per il server esterno
        payload = {
            "text": request.text,
            "language": request.language,
            "engine": request.engine
        }
        
        if request.engine == "chatterbox":
            if request.model:
                payload["model"] = request.model
            if request.device:
                payload["device"] = request.device
            if request.exaggeration is not None:
                payload["exaggeration"] = request.exaggeration
            if request.audio_prompt_path:
                payload["audio_prompt_path"] = request.audio_prompt_path
        elif request.engine == "vibevoice":
            if request.model:
                payload["model"] = request.model
            if request.speaker:
                payload["speaker"] = request.speaker
            if request.speed:
                payload["speed"] = request.speed
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{tts_server_url}/synthesize",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=error)
                
                # Leggi PCM data
                pcm_data = await resp.read()
                sample_rate = int(resp.headers.get("X-Sample-Rate", "24000"))
                duration = float(resp.headers.get("X-Duration", "0"))
                
                # Converti PCM in numpy array e poi in WAV
                audio_array = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32767.0
                
                # Crea buffer WAV in memoria
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, audio_array, sample_rate, format='WAV', subtype='PCM_16')
                wav_data = wav_buffer.getvalue()
                
                logger.info(f"✅ Test TTS via server esterno completato: engine={request.engine}, duration={duration:.2f}s")
                
                return Response(
                    content=wav_data,
                    media_type="audio/wav",
                    headers={
                        "X-Sample-Rate": str(sample_rate),
                        "X-Duration": str(duration),
                        "X-Engine": request.engine
                    }
                )
    except aiohttp.ClientError as e:
        logger.error(f"❌ Errore connessione server TTS esterno: {e}")
        raise HTTPException(status_code=503, detail=f"Server TTS esterno non disponibile: {e}")
    except Exception as e:
        logger.error(f"❌ Errore test TTS via server esterno: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts/test")
async def test_tts(request: TTSTestRequest):
    """Testa un TTS engine con un testo specifico, ritorna audio WAV"""
    import numpy as np
    import soundfile as sf
    import io
    
    try:
        # Importa il factory function per creare TTS engines
        sys.path.insert(0, str(Path(__file__).parent))
        from agent.tts import get_tts_engine
        
        # Prepara parametri in base all'engine
        tts_params = {}
        
        # TTS che vanno chiamati via server esterno (venv locale con MPS/GPU)
        if request.engine in ["chatterbox", "piper", "kokoro", "vibevoice"]:
            tts_server_url = os.getenv("TTS_SERVER_URL", "http://host.docker.internal:8092")
            logger.info(f"🔗 Routing {request.engine} a server esterno: {tts_server_url}")
            return await test_tts_via_external_server(request, tts_server_url)
        
        # TTS che girano nel container Docker
        if request.engine == "edge":
            # EdgeTTS non accetta 'language' come parametro, ma possiamo usarlo per selezionare la voce
            if request.voice:
                tts_params["voice"] = request.voice
            else:
                # Usa una voce di default basata sulla lingua
                voice_map = {
                    "it": "it-IT-DiegoNeural",
                    "en": "en-US-GuyNeural",
                    "es": "es-ES-AlvaroNeural",
                    "fr": "fr-FR-HenriNeural",
                    "de": "de-DE-ConradNeural",
                    "zh": "zh-CN-YunxiNeural"
                }
                tts_params["voice"] = voice_map.get(request.language, "it-IT-DiegoNeural")
        else:
            # Per altri engine, aggiungi language se supportato
            tts_params["language"] = request.language
        
        # Crea istanza TTS (get_tts_engine gestisce automaticamente i fallback se un engine non è disponibile)
        try:
            tts_engine = get_tts_engine(request.engine, **tts_params)
            # Verifica se il fallback è stato usato (controlla il tipo effettivo)
            actual_engine_type = tts_engine.engine_type.value if hasattr(tts_engine, 'engine_type') else None
            if request.engine.lower() != actual_engine_type and actual_engine_type == "edge":
                logger.info(f"ℹ️ {request.engine} non disponibile nel container, uso EdgeTTS come fallback per il test")
        except Exception as e:
            # Se l'engine richiesto fallisce, prova con EdgeTTS come fallback
            logger.warning(f"⚠️ Errore creazione {request.engine} TTS: {e}, uso EdgeTTS come fallback")
            from agent.tts.edge_tts_engine import EdgeTTS
            voice_map = {
                "it": "it-IT-DiegoNeural",
                "en": "en-US-GuyNeural",
                "es": "es-ES-AlvaroNeural",
                "fr": "fr-FR-HenriNeural",
                "de": "de-DE-ConradNeural",
                "zh": "zh-CN-YunxiNeural"
            }
            tts_engine = EdgeTTS(voice=voice_map.get(request.language, "it-IT-DiegoNeural"))
        
        # Sintetizza
        import asyncio
        result = await tts_engine.synthesize_async(request.text)
        
        # Converti audio_data (float32 numpy array) in WAV
        # Normalizza se necessario
        audio_data = result.audio_data
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Normalizza se fuori range [-1, 1]
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Crea buffer WAV in memoria
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, result.sample_rate, format='WAV', subtype='PCM_16')
        wav_data = wav_buffer.getvalue()
        
        logger.info(f"✅ Test TTS completato: engine={request.engine}, text_len={len(request.text)}, audio_duration={result.duration_seconds:.2f}s")
        
        return Response(
            content=wav_data,
            media_type="audio/wav",
            headers={
                "X-Sample-Rate": str(result.sample_rate),
                "X-Duration": str(result.duration_seconds),
                "X-Engine": request.engine
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Errore test TTS: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# VibeVoice Model Management
_vibevoice_download_status = {
    "downloading": False,
    "percent": 0,
    "complete": False,
    "error": None,
    "model": None,
    "downloaded_bytes": 0,
    "total_bytes": 0,
    "speed_bps": 0,
    "eta_seconds": 0,
    "current_file": "",
    "files_done": 0,
    "files_total": 0
}

VIBEVOICE_MODELS = {
    "realtime": {
        "model_id": "microsoft/VibeVoice-Realtime-0.5B",
        "name": "VibeVoice Realtime 0.5B",
        "download_size": "~1.5GB",
        "download_bytes": 1500000000,
        "description": "Streaming TTS con bassa latenza (~300ms)"
    },
    "longform": {
        "model_id": "microsoft/VibeVoice-1.6B",
        "name": "VibeVoice Long-form 1.6B",
        "download_size": "~4GB",
        "download_bytes": 4000000000,
        "description": "Multi-speaker, alta qualità per contenuti lunghi"
    }
}


@app.get("/api/tts/vibevoice/status")
async def get_vibevoice_status(model: str = "realtime"):
    """Verifica se il modello VibeVoice è installato"""
    if model not in VIBEVOICE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modello non valido: {model}")
    
    model_info = VIBEVOICE_MODELS[model]
    
    # Controlla se il modello è installato
    # Il modello sarà in ~/.cache/huggingface/hub o nella directory configurata
    import os
    from pathlib import Path
    
    hf_cache = Path(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))) / "hub"
    model_id = model_info["model_id"].replace("/", "--")
    model_dir = hf_cache / f"models--{model_id}"
    
    # Controlla anche directory alternativa per modelli locali
    local_model_dir = Path("/app/models/vibevoice") / model
    
    installed = model_dir.exists() or local_model_dir.exists()
    
    # Stima dimensione se installato
    size = None
    if installed:
        try:
            if model_dir.exists():
                size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                size = f"{size / (1024**3):.1f}GB"
        except:
            size = "N/A"
    
    return {
        "installed": installed,
        "model": model,
        "model_name": model_info["name"],
        "model_id": model_info["model_id"],
        "download_size": model_info["download_size"],
        "size": size,
        "description": model_info["description"]
    }


class VibeVoiceDownloadRequest(BaseModel):
    """Richiesta download modello VibeVoice"""
    model: str = "realtime"


@app.post("/api/tts/vibevoice/download")
async def download_vibevoice_model(request: VibeVoiceDownloadRequest):
    """Avvia il download del modello VibeVoice"""
    global _vibevoice_download_status
    
    if request.model not in VIBEVOICE_MODELS:
        raise HTTPException(status_code=400, detail=f"Modello non valido: {request.model}")
    
    if _vibevoice_download_status["downloading"]:
        return {"status": "already_downloading", "model": _vibevoice_download_status["model"]}
    
    model_info = VIBEVOICE_MODELS[request.model]
    
    # Reset status
    _vibevoice_download_status = {
        "downloading": True,
        "percent": 0,
        "complete": False,
        "error": None,
        "model": request.model
    }
    
    # Avvia download in background
    import asyncio
    asyncio.create_task(_download_vibevoice_model_task(request.model, model_info["model_id"]))
    
    logger.info(f"Avvio download modello VibeVoice: {request.model}")
    
    return {
        "status": "started",
        "model": request.model,
        "model_id": model_info["model_id"]
    }


async def _download_vibevoice_model_task(model: str, model_id: str):
    """Task asincrono per download modello con tracking dettagliato"""
    global _vibevoice_download_status
    
    import time
    
    # Directory per i modelli (volume Docker persistente)
    models_dir = Path(os.getenv("HF_HOME", "/app/models/huggingface"))
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Download VibeVoice in: {models_dir}")
    
    try:
        from huggingface_hub import snapshot_download, HfApi
        from huggingface_hub.utils import tqdm as hf_tqdm
        
        _vibevoice_download_status["percent"] = 5
        _vibevoice_download_status["current_file"] = "Inizializzazione..."
        
        # Ottieni info sul repository
        try:
            api = HfApi()
            repo_info = api.repo_info(repo_id=model_id)
            
            # Calcola dimensione totale
            total_size = 0
            files_list = []
            for sibling in repo_info.siblings:
                if sibling.size:
                    total_size += sibling.size
                    files_list.append(sibling.rfilename)
            
            _vibevoice_download_status["total_bytes"] = total_size
            _vibevoice_download_status["files_total"] = len(files_list)
            
            logger.info(f"VibeVoice: {len(files_list)} files, {total_size / (1024**3):.2f} GB")
            
        except Exception as e:
            logger.warning(f"Impossibile ottenere info repo: {e}")
            _vibevoice_download_status["total_bytes"] = VIBEVOICE_MODELS.get(model, {}).get("download_bytes", 1500000000)
        
        _vibevoice_download_status["percent"] = 10
        start_time = time.time()
        last_update = start_time
        last_bytes = 0
        
        def progress_callback(progress):
            """Callback per tracciare il progresso del download"""
            nonlocal last_update, last_bytes
            
            current_time = time.time()
            
            if hasattr(progress, 'n') and hasattr(progress, 'total'):
                downloaded = progress.n
                total = progress.total or _vibevoice_download_status["total_bytes"]
                
                _vibevoice_download_status["downloaded_bytes"] = downloaded
                
                if total > 0:
                    percent = min(95, 10 + int((downloaded / total) * 85))
                    _vibevoice_download_status["percent"] = percent
                
                # Calcola velocità ogni secondo
                if current_time - last_update >= 1.0:
                    elapsed = current_time - last_update
                    bytes_diff = downloaded - last_bytes
                    speed = bytes_diff / elapsed if elapsed > 0 else 0
                    
                    _vibevoice_download_status["speed_bps"] = int(speed)
                    
                    # ETA
                    remaining = total - downloaded
                    if speed > 0:
                        eta = remaining / speed
                        _vibevoice_download_status["eta_seconds"] = int(eta)
                    
                    last_update = current_time
                    last_bytes = downloaded
        
        # Download con resume automatico
        _vibevoice_download_status["current_file"] = "Download in corso..."
        
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=model_id,
                cache_dir=str(models_dir),
                resume_download=True,
                local_files_only=False
            )
        )
        
        _vibevoice_download_status["percent"] = 100
        _vibevoice_download_status["complete"] = True
        _vibevoice_download_status["downloading"] = False
        _vibevoice_download_status["current_file"] = "Completato!"
        _vibevoice_download_status["eta_seconds"] = 0
        
        logger.info(f"Download modello VibeVoice completato: {model}")
        
    except ImportError as e:
        logger.error(f"huggingface_hub non disponibile: {e}")
        _vibevoice_download_status["error"] = "huggingface_hub non installato"
        _vibevoice_download_status["downloading"] = False
        _vibevoice_download_status["complete"] = False
        
    except Exception as e:
        logger.error(f"Errore download modello VibeVoice: {e}")
        _vibevoice_download_status["error"] = str(e)
        _vibevoice_download_status["downloading"] = False
        _vibevoice_download_status["complete"] = False
        _vibevoice_download_status["current_file"] = f"Errore: {str(e)[:50]}"


@app.get("/api/tts/vibevoice/download/progress")
async def get_vibevoice_download_progress():
    """Ritorna lo stato del download in corso"""
    return _vibevoice_download_status


# Serve file statici
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


def main():
    """Avvia il server con HTTPS"""
    port = config.server.web_port
    
    # Percorsi certificati (controlla sia locale che Docker)
    cert_dir = Path(__file__).parent / "certs"
    docker_cert_dir = Path("/app/certs")
    
    # Usa certificati Docker se esistono, altrimenti locali
    if docker_cert_dir.exists():
        ssl_keyfile = docker_cert_dir / "key.pem"
        ssl_certfile = docker_cert_dir / "cert.pem"
    else:
        ssl_keyfile = cert_dir / "key.pem"
        ssl_certfile = cert_dir / "cert.pem"
    
    # Verifica se i certificati esistono
    use_ssl = ssl_keyfile.exists() and ssl_certfile.exists()
    
    if use_ssl:
        logger.info(f"🔒 Avvio server HTTPS su porta {port}...")
        logger.info(f"📱 Collegati a: https://localhost:{port}")
        
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=port,
            ssl_keyfile=str(ssl_keyfile),
            ssl_certfile=str(ssl_certfile),
            reload=False,  # Reload non funziona bene con SSL
            log_level=config.server.log_level.lower()
        )
    else:
        logger.warning("⚠️ Certificati SSL non trovati, avvio in HTTP")
        logger.info(f"Avvio server HTTP su porta {port}...")
        
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level=config.server.log_level.lower()
        )


if __name__ == "__main__":
    main()
