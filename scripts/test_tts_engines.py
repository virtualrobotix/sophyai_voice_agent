#!/usr/bin/env python3
"""
Test runtime di tutti gli engine TTS esposti da tts_server (porta 8092).
Scrive NDJSON in .cursor/debug-fac0c1.log per analisi debug.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / ".cursor" / "debug-fac0c1.log"
SESSION_ID = "fac0c1"
BASE = "http://127.0.0.1:8092"
TEXT_IT = "Ciao, questa è una prova di sintesi vocale in italiano."
TIMEOUT_SEC = 900  # Qwen: primo download+caricamento modello può superare i 7 minuti

ENGINES = [
    ("edge", {"text": TEXT_IT, "language": "it", "engine": "edge"}, "H2-cloud"),
    ("piper", {"text": TEXT_IT, "language": "it", "engine": "piper"}, "H4-piper"),
    ("kokoro", {"text": TEXT_IT, "language": "it", "engine": "kokoro", "speed": 1.0}, "H4-kokoro"),
    ("qwen", {"text": TEXT_IT, "language": "it", "engine": "qwen", "speaker": "ryan"}, "H3-qwen"),
    (
        "chatterbox",
        {
            "text": TEXT_IT,
            "language": "it",
            "engine": "chatterbox",
            "model": "multilingual",
            "device": "cuda",
        },
        "H2-chatterbox",
    ),
    (
        "vibevoice",
        {"text": TEXT_IT, "language": "it", "engine": "vibevoice", "speaker": "ryan", "speed": 1.0},
        "H1-vibevoice",
    ),
]


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = {
        "sessionId": SESSION_ID,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
        "runId": "tts-batch",
    }
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def _get_json(path: str) -> tuple[int, dict | None]:
    req = urllib.request.Request(f"{BASE}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode()
            return r.status, json.loads(body) if body else None
    except Exception as e:
        return -1, {"error": str(e)}


def _post_synth(payload: dict) -> tuple[int, dict, bytes]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}/synthesize",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as r:
            raw = r.read()
            hdrs = {k.lower(): v for k, v in r.headers.items()}
            return (
                r.status,
                {
                    "x_engine": hdrs.get("x-engine"),
                    "x_sample_rate": hdrs.get("x-sample-rate"),
                    "content_length": len(raw),
                },
                raw,
            )
    except urllib.error.HTTPError as e:
        err_body = e.read().decode(errors="replace")[:500]
        return e.code, {"error_body": err_body}, b""
    except Exception as e:
        return -1, {"error": str(e)}, b""


def main() -> None:
    code, st = _get_json("/status")
    _log(
        "H5-status",
        "scripts/test_tts_engines.py:main",
        "tts_status",
        {"http_status": code, "status_json": st},
    )

    global_engine = (st or {}).get("engine") if st else None
    _log(
        "H1-startup",
        "scripts/test_tts_engines.py:main",
        "global_tts_engine_at_startup",
        {"global_engine": global_engine, "device": (st or {}).get("device")},
    )

    for name, payload, hid in ENGINES:
        if name == "vibevoice" and global_engine != "vibevoice":
            _log(
                hid,
                "scripts/test_tts_engines.py:skip",
                "skip_vibevoice_not_loaded",
                {"global_engine": global_engine},
            )
            continue

        t0 = time.time()
        sc, meta, pcm = _post_synth(payload)
        elapsed_ms = int((time.time() - t0) * 1000)
        _log(
            hid,
            "scripts/test_tts_engines.py:synthesize",
            f"engine_{name}",
            {
                "engine_requested": name,
                "http_status": sc,
                "elapsed_ms": elapsed_ms,
                "pcm_bytes": len(pcm),
                "response_meta": meta,
            },
        )
        print(f"{name:12} HTTP {sc:3}  {elapsed_ms:6d} ms  pcm={len(pcm)}  X-Engine={meta.get('x_engine') if isinstance(meta, dict) else ''}")


if __name__ == "__main__":
    main()
