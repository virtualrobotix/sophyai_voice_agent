"""
Web Server
Server FastAPI per il frontend e le API.
Supporta HTTPS per l'accesso al microfono.
"""

import asyncio
import json
import os
import sys
import subprocess
import time
import yaml
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import socket
from loguru import logger
from livekit import api
import httpx

# Aggiungi il path del progetto
sys.path.insert(0, str(Path(__file__).parent))

from agent.config import config


def get_server_ip() -> str:
    """Rileva l'IP del server nella rete locale"""
    # Prima controlla se è impostato esplicitamente
    server_ip = os.getenv("SERVER_IP", "").strip()
    if server_ip and server_ip not in ("host.docker.internal", ""):
        return server_ip
    
    # Prova a rilevare automaticamente l'IP
    try:
        # Crea un socket UDP e connettiti a un IP esterno per rilevare l'IP locale
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        # Se è un IP interno Docker (172.x.x.x), ritorna localhost
        if ip.startswith("172."):
            return "localhost"
        return ip
    except Exception:
        return "localhost"


def get_livekit_url_for_client(request: Request) -> str:
    """Costruisce l'URL LiveKit appropriato per il client"""
    # #region agent log
    try:
        log_data = {"location": "server.py:get_livekit_url_for_client", "message": "Entry", "data": {"configured_url": config.livekit.url, "request_scheme": str(request.url.scheme), "x_forwarded_proto": request.headers.get("x-forwarded-proto"), "origin": request.headers.get("origin"), "host": request.headers.get("host")}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "hypothesisId": "A"}; log_path = Path(__file__).parent / ".cursor" / "debug.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.open("a").write(json.dumps(log_data) + "\n")
    except Exception: pass
    # #endregion
    # Se LIVEKIT_URL è configurato con un IP specifico (non localhost/0.0.0.0), usalo
    configured_url = config.livekit.url
    
    # Estrai host e porta dall'URL configurato
    # Formato: ws://host:port o wss://host:port
    if "://" in configured_url:
        proto, rest = configured_url.split("://", 1)
        host_port = rest.split("/")[0]
        if ":" in host_port:
            host, port = host_port.rsplit(":", 1)
        else:
            host = host_port
            port = "7880"
    else:
        proto = "ws"
        host = "localhost"
        port = "7880"
    
    # Se l'host è localhost, 0.0.0.0 o 127.0.0.1, usa l'host dalla richiesta
    if host in ("localhost", "0.0.0.0", "127.0.0.1"):
        # Prova a usare l'host dalla richiesta (es. 192.168.1.100 o localhost)
        request_host = request.headers.get("host", "").split(":")[0]
        
        # IMPORTANTE: Se il client accede da localhost, mantieni localhost
        # (il certificato è valido solo per localhost)
        if request_host in ("localhost", "127.0.0.1"):
            host = "localhost"
        elif request_host:
            # Se il client sta accedendo via IP, usa quell'IP
            host = request_host
        else:
            # Altrimenti rileva l'IP del server
            host = get_server_ip()
    
    # IMPORTANTE: Se la richiesta arriva via HTTPS, usa WSS per evitare Mixed Content
    # Controlla l'header X-Forwarded-Proto o lo schema della richiesta
    is_https = (
        request.headers.get("x-forwarded-proto") == "https" or
        request.url.scheme == "https" or
        request.headers.get("origin", "").startswith("https://")
    )
    
    # #region agent log
    try:
        log_data = {"location": "server.py:get_livekit_url_for_client", "message": "Before HTTPS check", "data": {"is_https": is_https, "proto_before": proto, "host": host, "port_before": port}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "hypothesisId": "A"}; log_path = Path(__file__).parent / ".cursor" / "debug.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.open("a").write(json.dumps(log_data) + "\n")
    except Exception: pass
    # #endregion
    
    if is_https:
        proto = "wss"
        port = "7443"  # Porta del proxy TLS per LiveKit
    
    final_url = f"{proto}://{host}:{port}"
    # #region agent log
    try:
        log_data = {"location": "server.py:get_livekit_url_for_client", "message": "Exit", "data": {"final_url": final_url, "proto": proto, "host": host, "port": port}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "hypothesisId": "A"}; log_path = Path(__file__).parent / ".cursor" / "debug.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.open("a").write(json.dumps(log_data) + "\n")
    except Exception: pass
    # #endregion
    return final_url


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

# Lock per evitare race condition nel dispatch degli agent
_room_dispatch_locks: dict[str, asyncio.Lock] = {}

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


@app.get("/admin.html")
@app.get("/admin")
async def admin_page():
    """Serve la pagina admin per i log delle chiamate"""
    return FileResponse("web/admin.html")


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


@app.post("/api/agent/restart")
async def restart_agent():
    """Riavvia il container dell'agent per applicare nuovi settings"""
    import asyncio
    
    try:
        # Esegui docker restart in background
        process = await asyncio.create_subprocess_exec(
            "docker", "restart", "voice-agent-worker",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
        
        if process.returncode == 0:
            logger.info("🔄 Agent riavviato con successo")
            return {"status": "ok", "message": "Agent riavviato. I nuovi settings saranno applicati."}
        else:
            error_msg = stderr.decode() if stderr else "Errore sconosciuto"
            logger.error(f"Errore restart agent: {error_msg}")
            return {"status": "error", "message": f"Errore: {error_msg}"}
    except asyncio.TimeoutError:
        return {"status": "error", "message": "Timeout durante il riavvio"}
    except Exception as e:
        logger.error(f"Errore restart agent: {e}")
        return {"status": "error", "message": str(e)}


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
        db = await get_database()
        logger.info("Database connesso")
        
        # Assicura che i nuovi voice settings esistano con valori di default
        if db:
            voice_defaults = {
                "wake_timeout_seconds": "20",
                "vad_energy_threshold": "40",
                "speech_energy_threshold": "100",
                "silence_threshold": "30",
                "tts_cooldown_seconds": "5"
            }
            for key, default_value in voice_defaults.items():
                existing = await db.get_setting(key)
                if existing is None:
                    await db.set_setting(key, default_value)
                    logger.info(f"📝 Aggiunto setting default: {key}={default_value}")
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
            "ollama_model": os.getenv("OLLAMA_MODEL", "gpt-oss:20b"),
            "openrouter_model": "",
            "openrouter_api_key": "",
            "whisper_model": os.getenv("WHISPER_MODEL", "medium"),
            "whisper_language": "it",
            "whisper_auto_detect": "false",
            "tts_engine": os.getenv("DEFAULT_TTS", "edge"),
            "tts_language": "it",
            "system_prompt": "",
            "context_injection": "",
            "remote_server_url": "",
            "remote_server_token": "",
            "remote_server_collection": "",
            # Voice Activation defaults
            "wake_timeout_seconds": "20",
            "vad_energy_threshold": "40",
            "speech_energy_threshold": "100",
            "silence_threshold": "30",
            "tts_cooldown_seconds": "5"
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
                
                # Estrai info architettura e capabilities
                architecture = m.get("architecture", {})
                modality = architecture.get("modality", "text->text")
                input_modalities = architecture.get("input_modalities", ["text"])
                output_modalities = architecture.get("output_modalities", ["text"])
                tokenizer = architecture.get("tokenizer", "unknown")
                
                # Determina se supporta vision
                supports_vision = "image" in input_modalities or "image" in modality.lower()
                
                # Parametri supportati
                supported_params = m.get("supported_parameters", [])
                supports_tools = "tools" in supported_params
                supports_json_mode = "response_format" in supported_params
                
                model_info = {
                    "id": m["id"],
                    "name": m.get("name", m["id"]),
                    "description": m.get("description", ""),
                    "context_length": m.get("context_length", 0),
                    "prompt_cost": prompt_cost,
                    "completion_cost": completion_cost,
                    "total_cost": prompt_cost + completion_cost,
                    "top_provider": m.get("top_provider", {}).get("max_completion_tokens"),
                    # Nuovi campi
                    "supports_vision": supports_vision,
                    "supports_tools": supports_tools,
                    "supports_json_mode": supports_json_mode,
                    "modality": modality,
                    "input_modalities": input_modalities,
                    "output_modalities": output_modalities,
                    "tokenizer": tokenizer,
                    "supported_parameters": supported_params,
                    "created": m.get("created"),
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
            elif sort_by == "vision":
                # Solo modelli con supporto vision
                models = [m for m in models if m["supports_vision"]]
                models.sort(key=lambda x: x["name"].lower())
            elif sort_by == "tools":
                # Solo modelli con supporto function calling
                models = [m for m in models if m["supports_tools"]]
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
    # Dettagli opzionali del modello
    name: str = None
    context_length: int = None
    supports_vision: bool = None
    supports_tools: bool = None
    supports_json_mode: bool = None
    modality: str = None
    prompt_cost: float = None
    completion_cost: float = None


@app.post("/api/openrouter/select")
async def select_openrouter_model(request: OpenRouterSelectRequest):
    """Select an OpenRouter model and save to settings with full details."""
    db = await get_database()
    
    if db:
        try:
            await db.set_setting("llm_provider", "openrouter")
            await db.set_setting("openrouter_model", request.model)
            
            # Salva dettagli del modello se forniti
            if request.name:
                await db.set_setting("openrouter_model_name", request.name)
            if request.context_length is not None:
                await db.set_setting("openrouter_context_length", str(request.context_length))
            if request.supports_vision is not None:
                await db.set_setting("openrouter_supports_vision", str(request.supports_vision).lower())
            if request.supports_tools is not None:
                await db.set_setting("openrouter_supports_tools", str(request.supports_tools).lower())
            if request.supports_json_mode is not None:
                await db.set_setting("openrouter_supports_json_mode", str(request.supports_json_mode).lower())
            if request.modality:
                await db.set_setting("openrouter_modality", request.modality)
            if request.prompt_cost is not None:
                await db.set_setting("openrouter_prompt_cost", str(request.prompt_cost))
            if request.completion_cost is not None:
                await db.set_setting("openrouter_completion_cost", str(request.completion_cost))
                
        except Exception as e:
            logger.warning(f"Errore salvataggio in DB: {e}")
    
    return {
        "status": "ok", 
        "model": request.model, 
        "provider": "openrouter",
        "details": {
            "name": request.name,
            "context_length": request.context_length,
            "supports_vision": request.supports_vision,
            "supports_tools": request.supports_tools,
            "modality": request.modality
        }
    }


@app.get("/api/openrouter/selected")
async def get_selected_openrouter_model():
    """Get details of the currently selected OpenRouter model."""
    db = await get_database()
    if db is None:
        return {"model": None, "details": None}
    
    try:
        model_id = await db.get_setting("openrouter_model")
        if not model_id:
            return {"model": None, "details": None}
        
        # Recupera tutti i dettagli salvati
        details = {
            "id": model_id,
            "name": await db.get_setting("openrouter_model_name") or model_id,
            "context_length": int(await db.get_setting("openrouter_context_length") or 0),
            "supports_vision": (await db.get_setting("openrouter_supports_vision") or "false") == "true",
            "supports_tools": (await db.get_setting("openrouter_supports_tools") or "false") == "true",
            "supports_json_mode": (await db.get_setting("openrouter_supports_json_mode") or "false") == "true",
            "modality": await db.get_setting("openrouter_modality") or "text->text",
            "prompt_cost": float(await db.get_setting("openrouter_prompt_cost") or 0),
            "completion_cost": float(await db.get_setting("openrouter_completion_cost") or 0),
        }
        
        return {"model": model_id, "details": details}
        
    except Exception as e:
        logger.error(f"Errore recupero modello selezionato: {e}")
        return {"model": None, "details": None, "error": str(e)}


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
    
    # Costruisci prompt di default con nome assistente dalla configurazione
    triggers_str = ", ".join([f'"{t}"' for t in config.branding.assistant_triggers[:3]])
    default_prompt = f"""Sei {config.branding.assistant_name}, assistente vocale ultra-veloce. PRIORITA ASSOLUTA: VELOCITA E SINTESI.

ATTIVAZIONE:
Rispondi SOLO quando vieni menzionato con {triggers_str} o varianti simili.

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


@app.get("/api/rooms")
async def get_rooms():
    """
    Ottiene la lista delle room attive in LiveKit.
    """
    import os
    internal_url = os.getenv("LIVEKIT_INTERNAL_URL", "ws://host.docker.internal:7880")
    
    try:
        lk_api = api.LiveKitAPI(
            url=internal_url,
            api_key=config.livekit.api_key,
            api_secret=config.livekit.api_secret
        )
        
        rooms_response = await lk_api.room.list_rooms(api.ListRoomsRequest())
        await lk_api.aclose()
        
        rooms = []
        for room in rooms_response.rooms:
            rooms.append({
                "name": room.name,
                "num_participants": room.num_participants,
                "creation_time": room.creation_time,
                "active_recording": room.active_recording
            })
        
        return {"rooms": rooms}
        
    except Exception as e:
        logger.error(f"Errore ottenimento rooms: {e}")
        return {"rooms": [], "error": str(e)}


@app.post("/api/livekit/webhook")
async def livekit_webhook(request: Request):
    """
    Webhook per eventi LiveKit.
    Gestisce apertura/chiusura chiamate SIP e dispatch automatico dell'agent.
    """
    import os
    import re
    internal_url = os.getenv("LIVEKIT_INTERNAL_URL", "ws://host.docker.internal:7880")
    
    try:
        body = await request.body()
        auth_header = request.headers.get("Authorization", "")
        
        # Verifica il webhook token (API LiveKit aggiornata)
        from livekit.api import WebhookReceiver, TokenVerifier
        token_verifier = TokenVerifier(
            api_key=config.livekit.api_key,
            api_secret=config.livekit.api_secret
        )
        webhook_receiver = WebhookReceiver(token_verifier)
        
        event = webhook_receiver.receive(body.decode(), auth_header)
        
        logger.info(f"📞 Webhook ricevuto: {event.event}")
        
        # Ottieni connessione database
        db = await get_database()
        
        # ==================== PARTECIPANTE SIP ENTRATO ====================
        if event.event == "participant_joined":
            participant = event.participant
            room = event.room
            
            # Verifica se è un partecipante SIP
            if participant and participant.identity.startswith("sip_"):
                logger.info(f"📞 CHIAMATA IN ARRIVO: {participant.identity} in room {room.name}")
                
                # Estrai numero dal participant identity (formato: sip_+XXXXXXXXXXX)
                caller_number = participant.identity.replace("sip_", "")
                caller_name = participant.name or f"Phone {caller_number}"
                
                # Genera un call_id univoco
                call_id = f"call_{room.name}_{int(datetime.now().timestamp())}"
                
                # Crea il log della chiamata nel database
                try:
                    call_log_id = await db.create_call_log(
                        call_id=call_id,
                        room_name=room.name,
                        caller_number=caller_number,
                        caller_name=caller_name,
                        metadata={
                            "participant_sid": participant.sid,
                            "room_sid": room.sid if hasattr(room, 'sid') else None
                        }
                    )
                    logger.info(f"📝 Log chiamata creato: ID={call_log_id}, call_id={call_id}")
                except Exception as e:
                    logger.error(f"Errore creazione log chiamata: {e}")
                
                # Dispatcha l'agent
                lk_api = api.LiveKitAPI(
                    url=internal_url,
                    api_key=config.livekit.api_key,
                    api_secret=config.livekit.api_secret
                )
                
                try:
                    # Verifica se c'è già un agent
                    participants_resp = await lk_api.room.list_participants(
                        api.ListParticipantsRequest(room=room.name)
                    )
                    
                    agent_exists = False
                    for p in participants_resp.participants:
                        if p.identity.startswith("agent-"):
                            agent_exists = True
                            break
                    
                    if not agent_exists:
                        await lk_api.agent_dispatch.create_dispatch(
                            api.CreateAgentDispatchRequest(
                                room=room.name,
                                agent_name=""
                            )
                        )
                        logger.info(f"✅ Agent dispatchato per chiamata SIP in room {room.name}")
                    else:
                        logger.info(f"ℹ️ Agent già presente in room {room.name}")
                        
                except Exception as e:
                    logger.error(f"Errore dispatch agent: {e}")
                finally:
                    await lk_api.aclose()
        
        # ==================== PARTECIPANTE SIP USCITO ====================
        elif event.event == "participant_left":
            participant = event.participant
            room = event.room
            
            if participant and participant.identity.startswith("sip_"):
                logger.info(f"📞 CHIAMATA TERMINATA: {participant.identity} da room {room.name}")
                
                # Trova e chiudi il log della chiamata
                try:
                    call_log = await db.get_call_log_by_room(room.name, status="active")
                    if call_log:
                        await db.end_call_log(call_log['call_id'], status="completed")
                        logger.info(f"📝 Log chiamata chiuso: {call_log['call_id']}")
                except Exception as e:
                    logger.error(f"Errore chiusura log chiamata: {e}")
        
        # ==================== ROOM CHIUSA ====================
        elif event.event == "room_finished":
            room = event.room
            logger.info(f"🚪 Room chiusa: {room.name}")
            
            # Chiudi eventuali chiamate ancora attive in questa room
            try:
                call_log = await db.get_call_log_by_room(room.name, status="active")
                if call_log:
                    await db.end_call_log(call_log['call_id'], status="completed")
                    logger.info(f"📝 Log chiamata chiuso (room finished): {call_log['call_id']}")
            except Exception as e:
                logger.error(f"Errore chiusura log per room finished: {e}")
        
        return {"status": "ok"}
        
    except Exception as e:
        logger.error(f"Errore webhook: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ==================== CALL LOGS API ====================

@app.get("/api/calls")
async def get_calls(
    limit: int = 50,
    offset: int = 0,
    status: str = None
):
    """Ottiene la lista delle chiamate con paginazione."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        calls = await db.get_call_logs(limit=limit, offset=offset, status=status)
        stats = await db.get_call_stats()
        return {
            "calls": calls,
            "stats": stats,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": len(calls) == limit
            }
        }
    except Exception as e:
        logger.error(f"Errore ottenimento chiamate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/calls/{call_id}")
async def get_call_detail(call_id: str):
    """Ottiene i dettagli di una singola chiamata."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        call_log = await db.get_call_log_by_call_id(call_id)
        if not call_log:
            raise HTTPException(status_code=404, detail="Chiamata non trovata")
        
        # Ottieni i messaggi
        messages = await db.get_call_messages(call_log['id'])
        
        # Formatta datetime
        if call_log.get('start_time'):
            call_log['start_time'] = call_log['start_time'].isoformat()
        if call_log.get('end_time'):
            call_log['end_time'] = call_log['end_time'].isoformat()
        if call_log.get('created_at'):
            call_log['created_at'] = call_log['created_at'].isoformat()
        if call_log.get('updated_at'):
            call_log['updated_at'] = call_log['updated_at'].isoformat()
        
        return {
            "call": call_log,
            "messages": messages
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Errore ottenimento dettaglio chiamata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/calls/stats/summary")
async def get_calls_stats():
    """Ottiene statistiche sulle chiamate."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        stats = await db.get_call_stats()
        return stats
    except Exception as e:
        logger.error(f"Errore ottenimento statistiche: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/calls/{call_id}/message")
async def add_call_message(call_id: str, role: str, content: str):
    """Aggiunge un messaggio al log di una chiamata (usato dall'agent)."""
    db = await get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database non disponibile")
    
    try:
        call_log = await db.get_call_log_by_call_id(call_id)
        if not call_log:
            # Prova a cercare per room name
            call_log = await db.get_call_log_by_room(call_id, status="active")
        
        if not call_log:
            raise HTTPException(status_code=404, detail="Chiamata non trovata")
        
        msg_id = await db.add_call_message(call_log['id'], role, content)
        return {"status": "ok", "message_id": msg_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Errore aggiunta messaggio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/token", response_model=TokenResponse)
async def get_token(request: TokenRequest, http_request: Request):
    """
    Genera un token LiveKit per un partecipante e dispatcha l'agent.
    Verifica che il nome utente sia univoco nella room.
    """
    import os
    internal_url = os.getenv("LIVEKIT_INTERNAL_URL", "ws://host.docker.internal:7880")
    
    try:
        # Crea API client per verifiche e operazioni
        lk_api = api.LiveKitAPI(
            url=internal_url,
            api_key=config.livekit.api_key,
            api_secret=config.livekit.api_secret
        )
        
        # Verifica se esiste già un utente con lo stesso nome nella room
        try:
            participants = await lk_api.room.list_participants(
                api.ListParticipantsRequest(room=request.room_name)
            )
            for p in participants.participants:
                if p.identity == request.participant_name:
                    await lk_api.aclose()
                    logger.warning(f"Nome utente duplicato: {request.participant_name} già presente in {request.room_name}")
                    raise HTTPException(
                        status_code=409, 
                        detail=f"Il nome '{request.participant_name}' è già in uso nella room. Scegli un nome diverso."
                    )
        except HTTPException:
            raise  # Rilancia l'errore 409
        except Exception as e:
            # La room potrebbe non esistere ancora, continua
            logger.debug(f"Room {request.room_name} non esiste ancora o errore verifica: {e}")
        
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
        
        # #region agent log
        try:
            log_data = {"location": "server.py:get_token", "message": "Token generated", "data": {"token_length": len(jwt_token), "token_preview": jwt_token[:50] + "..." if len(jwt_token) > 50 else jwt_token, "room_name": request.room_name, "participant_name": request.participant_name}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "hypothesisId": "B"}; log_path = Path(__file__).parent / ".cursor" / "debug.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.open("a").write(json.dumps(log_data) + "\n")
        except Exception: pass
        # #endregion
        
        # URL WebSocket per il client - costruito dinamicamente
        ws_url = get_livekit_url_for_client(http_request)
        
        # #region agent log
        try:
            log_data = {"location": "server.py:get_token", "message": "URL generated", "data": {"ws_url": ws_url, "room_name": request.room_name}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "hypothesisId": "A"}; log_path = Path(__file__).parent / ".cursor" / "debug.log"; log_path.parent.mkdir(parents=True, exist_ok=True); log_path.open("a").write(json.dumps(log_data) + "\n")
        except Exception: pass
        # #endregion
        
        # Crea la room se non esiste e dispatcha l'agent SOLO se non ce n'è già uno
        # Usa un lock per evitare race condition quando più utenti si connettono simultaneamente
        if request.room_name not in _room_dispatch_locks:
            _room_dispatch_locks[request.room_name] = asyncio.Lock()
        
        async with _room_dispatch_locks[request.room_name]:
            try:
                # Crea la room se non esiste
                await lk_api.room.create_room(
                    api.CreateRoomRequest(name=request.room_name)
                )
                
                # Verifica se c'è già un agent nella room
                participants = await lk_api.room.list_participants(
                    api.ListParticipantsRequest(room=request.room_name)
                )
                
                agent_exists = False
                for p in participants.participants:
                    # Gli agent hanno identity che inizia con "agent-"
                    if p.identity.startswith("agent-"):
                        agent_exists = True
                        logger.info(f"Agent già presente nella room {request.room_name}: {p.identity}")
                        break
                
                # Dispatcha l'agent SOLO se non ce n'è già uno
                if not agent_exists:
                    await lk_api.agent_dispatch.create_dispatch(
                        api.CreateAgentDispatchRequest(
                            room=request.room_name,
                            agent_name=""  # Agent di default
                        )
                    )
                    logger.info(f"Nuovo agent dispatchato per room {request.room_name}")
                else:
                    logger.info(f"Agent già attivo in room {request.room_name}, skip dispatch")

            except Exception as dispatch_err:
                logger.warning(f"Agent dispatch fallito (potrebbe essere già attivo): {dispatch_err}")
        
        await lk_api.aclose()
        
        logger.info(f"Token generato per {request.participant_name} in room {request.room_name}")
        logger.info(f"🌐 LiveKit URL per client: {ws_url}")
        
        return TokenResponse(token=jwt_token, url=ws_url)
        
    except HTTPException:
        raise  # Rilancia errori HTTP (es. 409)
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
async def get_config(request: Request):
    """Ritorna la configurazione pubblica"""
    # Costruisci URL LiveKit dinamicamente in base alla richiesta
    livekit_url = get_livekit_url_for_client(request)
    server_ip = get_server_ip()
    
    return {
        "livekit_url": livekit_url,
        "livekit_url_configured": config.livekit.url,
        "server_ip": server_ip,
        "default_tts": config.tts.default_engine,
        "whisper_model": config.whisper.model,
        "ollama_model": config.ollama.model
    }


@app.get("/api/branding")
async def get_branding():
    """Ritorna la configurazione di branding dell'applicativo"""
    return {
        "app_name": config.branding.app_name,
        "assistant_name": config.branding.assistant_name,
        "assistant_triggers": config.branding.assistant_triggers
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


# ==================== Remote LLM Server API ====================

class RemoteServerTestRequest(BaseModel):
    """Test remote server connection request."""
    server_url: str
    token: str = ""
    collection: str = ""


@app.post("/api/remote/test")
async def test_remote_server(request: RemoteServerTestRequest):
    """
    Testa la connessione a un server LLM remoto.
    
    Invia un messaggio di test e verifica la risposta.
    """
    try:
        # Test diretto senza usare RemoteLLM per evitare dipendenze LiveKit
        headers = {"Content-Type": "application/json"}
        if request.token:
            headers["Authorization"] = f"Bearer {request.token}"
        
        test_payload = {
            "message": "test connection",
            "collection": request.collection
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Prova health check
            try:
                health_resp = await client.get(
                    f"{request.server_url}/health",
                    headers=headers
                )
                if health_resp.status_code == 200:
                    return {
                        "status": "ok",
                        "message": "Server raggiungibile",
                        "endpoint": "/health"
                    }
            except:
                pass
            
            # Prova chat endpoint
            resp = await client.post(
                f"{request.server_url}/chat",
                headers=headers,
                json=test_payload
            )
            
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": "ok",
                    "message": "Connessione riuscita",
                    "endpoint": "/chat",
                    "response_preview": data.get("response", "")[:100]
                }
            elif resp.status_code == 401:
                return {
                    "status": "error",
                    "message": "Token non valido o mancante",
                    "code": 401
                }
            else:
                return {
                    "status": "error",
                    "message": f"Errore HTTP {resp.status_code}",
                    "code": resp.status_code
                }
                
    except httpx.TimeoutException:
        return {"status": "error", "message": "Timeout: server non risponde"}
    except httpx.ConnectError:
        return {"status": "error", "message": f"Impossibile connettersi a {request.server_url}"}
    except Exception as e:
        logger.error(f"Errore test server remoto: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/remote/collections")
async def get_remote_collections(server_url: str, token: str = ""):
    """
    Recupera la lista delle collection disponibili dal server remoto.
    """
    try:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{server_url}/collections",
                headers=headers
            )
            
            if resp.status_code == 200:
                data = resp.json()
                # Supporta vari formati di risposta
                if isinstance(data, list):
                    return {"collections": data, "count": len(data)}
                elif isinstance(data, dict):
                    collections = data.get("collections", data.get("data", []))
                    return {"collections": collections, "count": len(collections)}
                return {"collections": [], "count": 0}
            else:
                return {"collections": [], "error": f"HTTP {resp.status_code}"}
                
    except Exception as e:
        logger.error(f"Errore recupero collections: {e}")
        return {"collections": [], "error": str(e)}


class RemoteServerSelectRequest(BaseModel):
    """Select remote server request."""
    server_url: str
    token: str = ""
    collection: str = ""


@app.post("/api/remote/select")
async def select_remote_server(request: RemoteServerSelectRequest):
    """
    Seleziona un server LLM remoto e salva la configurazione.
    """
    db = await get_database()
    
    if db:
        try:
            await db.set_setting("llm_provider", "remote")
            await db.set_setting("remote_server_url", request.server_url)
            await db.set_setting("remote_server_token", request.token)
            await db.set_setting("remote_server_collection", request.collection)
        except Exception as e:
            logger.warning(f"Errore salvataggio in DB: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    logger.info(f"🖥️ Server remoto selezionato: {request.server_url}, collection={request.collection}")
    
    return {
        "status": "ok",
        "provider": "remote",
        "server_url": request.server_url,
        "collection": request.collection
    }


@app.get("/api/remote/config")
async def get_remote_config():
    """
    Recupera la configurazione del server remoto salvata.
    """
    db = await get_database()
    
    if db is None:
        return {
            "server_url": "",
            "token": "",
            "collection": "",
            "configured": False
        }
    
    try:
        server_url = await db.get_setting("remote_server_url") or ""
        token = await db.get_setting("remote_server_token") or ""
        collection = await db.get_setting("remote_server_collection") or ""
        
        return {
            "server_url": server_url,
            "token": token,
            "collection": collection,
            "configured": bool(server_url)
        }
        
    except Exception as e:
        logger.error(f"Errore lettura config remote: {e}")
        return {
            "server_url": "",
            "token": "",
            "collection": "",
            "configured": False,
            "error": str(e)
        }


# ==================== SIP Configuration API ====================

class SIPConfig(BaseModel):
    """SIP configuration model."""
    sip_port: int = 5060
    sip_port_tls: int = 5061
    rtp_port_start: int = 10000
    rtp_port_end: int = 10100
    # Trunk configuration
    trunk_name: str = ""
    trunk_host: str = ""
    trunk_port: int = 5060
    trunk_username: str = ""
    trunk_password: str = ""
    trunk_numbers: str = ""  # Comma-separated list of phone numbers
    # Dispatch rules
    room_prefix: str = "sip-call-"
    enable_recording: bool = False
    # Audio codecs (comma-separated)
    audio_codecs: str = "opus,pcmu,pcma"


@app.get("/api/sip/status")
async def get_sip_status():
    """Verifica lo stato dettagliato del servizio SIP di LiveKit."""
    import socket
    
    status = {
        "available": False,
        "message": "Non configurato",
        "details": {
            "service_running": False,
            "port_5060": False,
            "port_5061": False,
            "trunks_configured": 0,
            "dispatch_rules": 0
        },
        "config": None
    }
    
    # Leggi configurazione SIP attuale
    sip_config_path = Path(__file__).parent / "sip-config.yaml"
    if sip_config_path.exists():
        try:
            with open(sip_config_path, "r") as f:
                sip_config = yaml.safe_load(f)
                status["config"] = sip_config
                
                # Conta trunk e regole
                trunks = sip_config.get("trunks", [])
                status["details"]["trunks_configured"] = len(trunks) if trunks else 0
                
                dispatch_rules = sip_config.get("dispatch_rules", [])
                status["details"]["dispatch_rules"] = len(dispatch_rules) if dispatch_rules else 0
        except Exception as e:
            logger.warning(f"Errore lettura sip-config.yaml: {e}")
    
    # Verifica se il servizio SIP è in esecuzione controllando le porte
    # LiveKit SIP non ha un endpoint HTTP health, verifichiamo direttamente le porte
    
    # Test porta 5060 TCP (SIP signaling)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("livekit-sip", 5060))
        if result == 0:
            status["details"]["port_5060"] = True
            status["details"]["service_running"] = True
        sock.close()
    except Exception as e:
        logger.debug(f"Test porta 5060 fallito: {e}")
    
    # Test porta 5061 TLS (se configurato)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(("livekit-sip", 5061))
        if result == 0:
            status["details"]["port_5061"] = True
        sock.close()
    except Exception as e:
        logger.debug(f"Test porta 5061 fallito: {e}")
    
    # Aggiorna stato finale
    if status["details"]["service_running"]:
        status["available"] = True
        status["message"] = "SIP Bridge attivo (porta 5060)"
        if status["details"]["port_5061"]:
            status["message"] += " e TLS (porta 5061)"
    else:
        status["message"] = "SIP Bridge non raggiungibile"
    
    return status


@app.get("/api/sip/config")
async def get_sip_config():
    """Recupera la configurazione SIP corrente."""
    
    sip_config_path = Path(__file__).parent / "sip-config.yaml"
    
    # Valori di default
    default_config = {
        "sip_port": 5060,
        "sip_port_tls": 5061,
        "rtp_port_start": 10000,
        "rtp_port_end": 10100,
        "trunk_name": "",
        "trunk_host": "",
        "trunk_port": 5060,
        "trunk_username": "",
        "trunk_password": "",
        "trunk_numbers": "",
        "room_prefix": "sip-call-",
        "enable_recording": False,
        "audio_codecs": "opus,pcmu,pcma"
    }
    
    # Prova a leggere dal database
    db = await get_database()
    if db:
        try:
            settings = await db.get_all_settings()
            for key in default_config.keys():
                sip_key = f"sip_{key}"
                if sip_key in settings:
                    value = settings[sip_key]
                    # Converti i tipi appropriati
                    if key in ["sip_port", "sip_port_tls", "rtp_port_start", "rtp_port_end", "trunk_port"]:
                        default_config[key] = int(value)
                    elif key == "enable_recording":
                        default_config[key] = value.lower() == "true"
                    else:
                        default_config[key] = value
        except Exception as e:
            logger.warning(f"Errore lettura SIP settings dal DB: {e}")
    
    # Leggi anche dal file YAML per completezza
    if sip_config_path.exists():
        try:
            with open(sip_config_path, "r") as f:
                yaml_config = yaml.safe_load(f)
                
                # Mappa valori YAML ai campi del form
                if yaml_config:
                    sip = yaml_config.get("sip", {})
                    if sip:
                        default_config["sip_port"] = sip.get("port", default_config["sip_port"])
                        default_config["sip_port_tls"] = sip.get("port_tls", default_config["sip_port_tls"])
                    
                    rtp = yaml_config.get("rtp", {})
                    if rtp:
                        default_config["rtp_port_start"] = rtp.get("port_range_start", default_config["rtp_port_start"])
                        default_config["rtp_port_end"] = rtp.get("port_range_end", default_config["rtp_port_end"])
                    
                    trunks = yaml_config.get("trunks", [])
                    if trunks and len(trunks) > 0:
                        trunk = trunks[0]
                        default_config["trunk_name"] = trunk.get("name", "")
                        default_config["trunk_host"] = trunk.get("host", "")
                        default_config["trunk_port"] = trunk.get("port", 5060)
                        default_config["trunk_username"] = trunk.get("username", "")
                        default_config["trunk_password"] = trunk.get("password", "")
                        numbers = trunk.get("numbers", [])
                        default_config["trunk_numbers"] = ",".join(numbers) if numbers else ""
                    
                    dispatch_rules = yaml_config.get("dispatch_rules", [])
                    if dispatch_rules and len(dispatch_rules) > 0:
                        rule = dispatch_rules[0]
                        default_config["room_prefix"] = rule.get("room_prefix", "sip-call-")
                        default_config["enable_recording"] = rule.get("enable_recording", False)
                    
                    audio = yaml_config.get("audio", {})
                    if audio:
                        codecs = audio.get("codecs", [])
                        default_config["audio_codecs"] = ",".join(codecs) if codecs else "opus,pcmu,pcma"
        except Exception as e:
            logger.warning(f"Errore lettura sip-config.yaml: {e}")
    
    return default_config


@app.post("/api/sip/config")
async def save_sip_config(sip_config: SIPConfig):
    """Salva la configurazione SIP."""
    
    # Salva nel database
    db = await get_database()
    if db:
        try:
            settings = {
                "sip_sip_port": str(sip_config.sip_port),
                "sip_sip_port_tls": str(sip_config.sip_port_tls),
                "sip_rtp_port_start": str(sip_config.rtp_port_start),
                "sip_rtp_port_end": str(sip_config.rtp_port_end),
                "sip_trunk_name": sip_config.trunk_name,
                "sip_trunk_host": sip_config.trunk_host,
                "sip_trunk_port": str(sip_config.trunk_port),
                "sip_trunk_username": sip_config.trunk_username,
                "sip_trunk_password": sip_config.trunk_password,
                "sip_trunk_numbers": sip_config.trunk_numbers,
                "sip_room_prefix": sip_config.room_prefix,
                "sip_enable_recording": str(sip_config.enable_recording).lower(),
                "sip_audio_codecs": sip_config.audio_codecs
            }
            await db.set_multiple_settings(settings)
        except Exception as e:
            logger.error(f"Errore salvataggio SIP settings nel DB: {e}")
    
    # Genera e salva il file sip-config.yaml
    sip_config_path = Path(__file__).parent / "sip-config.yaml"
    
    # Costruisci la configurazione YAML
    yaml_config = {
        "sip": {
            "port": sip_config.sip_port,
            "port_tls": sip_config.sip_port_tls
        },
        "rtp": {
            "port_range_start": sip_config.rtp_port_start,
            "port_range_end": sip_config.rtp_port_end
        },
        "livekit": {
            "url": os.getenv("LIVEKIT_INTERNAL_URL", "ws://host.docker.internal:7880"),
            "api_key": os.getenv("LIVEKIT_API_KEY", "devkey"),
            "api_secret": os.getenv("LIVEKIT_API_SECRET", "secret_dev_key_change_in_production")
        },
        "dispatch_rules": [
            {
                "name": "default-inbound",
                "room_prefix": sip_config.room_prefix,
                "metadata": {
                    "source": "sip",
                    "type": "phone-call"
                },
                "enable_recording": sip_config.enable_recording
            }
        ],
        "audio": {
            "codecs": [c.strip() for c in sip_config.audio_codecs.split(",") if c.strip()],
            "sample_rate": 48000
        },
        "logging": {
            "level": "info"
        }
    }
    
    # Aggiungi trunk se configurato
    if sip_config.trunk_host:
        trunk = {
            "name": sip_config.trunk_name or "trunk-principale",
            "host": sip_config.trunk_host,
            "port": sip_config.trunk_port
        }
        if sip_config.trunk_username:
            trunk["username"] = sip_config.trunk_username
        if sip_config.trunk_password:
            trunk["password"] = sip_config.trunk_password
        if sip_config.trunk_numbers:
            trunk["numbers"] = [n.strip() for n in sip_config.trunk_numbers.split(",") if n.strip()]
        yaml_config["trunks"] = [trunk]
    
    # Scrivi il file YAML
    try:
        with open(sip_config_path, "w") as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"📞 Configurazione SIP salvata in {sip_config_path}")
    except Exception as e:
        logger.error(f"Errore scrittura sip-config.yaml: {e}")
        raise HTTPException(status_code=500, detail=f"Errore salvataggio file: {e}")
    
    return {
        "status": "ok",
        "message": "Configurazione SIP salvata. Riavvia il servizio SIP per applicare.",
        "config_file": str(sip_config_path)
    }


@app.post("/api/sip/test")
async def test_sip_connection():
    """Testa la connessione SIP."""
    import aiohttp
    
    result = {
        "status": "unknown",
        "tests": []
    }
    
    # Test 1: Verifica container SIP
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://livekit-sip:8080/health", timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    result["tests"].append({"name": "SIP Service Health", "passed": True, "message": "Servizio attivo"})
                else:
                    result["tests"].append({"name": "SIP Service Health", "passed": False, "message": f"HTTP {resp.status}"})
    except Exception as e:
        result["tests"].append({"name": "SIP Service Health", "passed": False, "message": str(e)[:50]})
    
    # Test 2: Verifica LiveKit API per SIP
    try:
        internal_url = os.getenv("LIVEKIT_INTERNAL_URL", "ws://host.docker.internal:7880")
        lk_api = api.LiveKitAPI(
            url=internal_url,
            api_key=config.livekit.api_key,
            api_secret=config.livekit.api_secret
        )
        # Verifica connessione LiveKit
        await lk_api.room.list_rooms(api.ListRoomsRequest())
        result["tests"].append({"name": "LiveKit Connection", "passed": True, "message": "Connesso"})
        await lk_api.aclose()
    except Exception as e:
        result["tests"].append({"name": "LiveKit Connection", "passed": False, "message": str(e)[:50]})
    
    # Calcola risultato finale
    all_passed = all(t["passed"] for t in result["tests"])
    result["status"] = "ok" if all_passed else "partial" if any(t["passed"] for t in result["tests"]) else "failed"
    
    return result


# Serve file statici
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


def main():
    """Avvia il server con HTTPS e HTTP"""
    import threading
    
    https_port = config.server.web_port  # 8443
    http_port = 8080  # Porta HTTP per app che non supportano certificati self-signed
    
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
    
    def run_http_server():
        """Avvia server HTTP in un thread separato"""
        import uvicorn
        logger.info(f"📱 Server HTTP su porta {http_port} (per app mobile)")
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=http_port,
            reload=False,
            log_level="warning"  # Meno verbose per HTTP
        )
    
    if use_ssl:
        # Avvia HTTP in background per le app mobile
        http_thread = threading.Thread(target=run_http_server, daemon=True)
        http_thread.start()
        
        logger.info(f"🔒 Avvio server HTTPS su porta {https_port}...")
        logger.info(f"📱 Collegati a: https://localhost:{https_port}")
        logger.info(f"📱 HTTP disponibile su: http://localhost:{http_port}")
        
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=https_port,
            ssl_keyfile=str(ssl_keyfile),
            ssl_certfile=str(ssl_certfile),
            reload=False,
            log_level=config.server.log_level.lower()
        )
    else:
        logger.warning("⚠️ Certificati SSL non trovati, avvio solo in HTTP")
        logger.info(f"Avvio server HTTP su porta {http_port}...")
        
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=http_port,
            reload=True,
            log_level=config.server.log_level.lower()
        )


if __name__ == "__main__":
    main()
