#!/usr/bin/env python3
"""
TTS Proxy Server

Instrada la sintesi TTS al container specifico per engine.
Blocca i fallback: se X-Engine != engine richiesto ritorna errore.
"""

from __future__ import annotations

import os
from typing import Any

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel


class TTSRequest(BaseModel):
    text: str
    language: str = "it"
    speaker: str = "ryan"
    speed: float = 1.0
    engine: str = "edge"
    model: str | None = None
    device: str | None = None
    exaggeration: float | None = None
    audio_prompt_path: str | None = None


app = FastAPI(title="TTS Proxy", description="Engine-aware routing proxy")


def _engine_routes() -> dict[str, str]:
    # Core container gestisce edge/piper/kokoro/qwen.
    core = os.getenv("TTS_CORE_URL", "http://tts-core:8092")
    return {
        "edge": os.getenv("TTS_EDGE_URL", core),
        "piper": os.getenv("TTS_PIPER_URL", core),
        "kokoro": os.getenv("TTS_KOKORO_URL", core),
        "qwen": os.getenv("TTS_QWEN_URL", core),
        "coqui": os.getenv("TTS_COQUI_URL", "http://tts-coqui:8092"),
        "chatterbox": os.getenv("TTS_CHATTERBOX_URL", "http://tts-chatterbox:8092"),
        "vibevoice": os.getenv("TTS_VIBEVOICE_URL", "http://tts-vibevoice:8092"),
    }


def _build_payload(req: TTSRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": req.text,
        "language": req.language,
        "speaker": req.speaker,
        "speed": req.speed,
        "engine": req.engine,
    }
    if req.model is not None:
        payload["model"] = req.model
    if req.device is not None:
        payload["device"] = req.device
    if req.exaggeration is not None:
        payload["exaggeration"] = req.exaggeration
    if req.audio_prompt_path is not None:
        payload["audio_prompt_path"] = req.audio_prompt_path
    return payload


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "tts-proxy"}


@app.get("/status")
async def status() -> dict[str, Any]:
    routes = _engine_routes()
    checks: dict[str, dict[str, Any]] = {}
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for engine, base in routes.items():
            try:
                async with session.get(f"{base}/health") as resp:
                    body = await resp.text()
                    checks[engine] = {"http_status": resp.status, "url": base, "body": body[:200]}
            except Exception as e:
                checks[engine] = {"http_status": -1, "url": base, "error": str(e)}
    return {"status": "ready", "routes": checks}


@app.post("/synthesize")
async def synthesize(req: TTSRequest) -> Response:
    engine = req.engine.lower()
    routes = _engine_routes()
    if engine not in routes:
        raise HTTPException(status_code=400, detail=f"Engine non supportato: {engine}")

    target = routes[engine]
    payload = _build_payload(req)
    timeout = aiohttp.ClientTimeout(total=600)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{target}/synthesize", json=payload) as resp:
                body = await resp.read()
                if resp.status != 200:
                    raise HTTPException(
                        status_code=resp.status,
                        detail=body.decode(errors="replace")[:600],
                    )

                actual = (resp.headers.get("X-Engine") or "").lower()
                if actual != engine:
                    raise HTTPException(
                        status_code=503,
                        detail=f"Fallback rilevato: richiesto '{engine}', usato '{actual or 'unknown'}'",
                    )

                return Response(
                    content=body,
                    media_type=resp.headers.get("content-type", "audio/pcm"),
                    headers={
                        "X-Sample-Rate": resp.headers.get("X-Sample-Rate", "24000"),
                        "X-Channels": resp.headers.get("X-Channels", "1"),
                        "X-Duration": resp.headers.get("X-Duration", "0"),
                        "X-Engine": actual,
                        "X-Upstream": target,
                    },
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Routing error ({engine} -> {target}): {e}")


@app.get("/routes")
async def routes() -> JSONResponse:
    return JSONResponse(_engine_routes())
