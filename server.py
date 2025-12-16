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
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
from livekit import api

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
    "tts": {"time_ms": 0, "audio_sec": 0, "count": 0}
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
    
    return {"status": "ok"}


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
