#!/usr/bin/env python3
"""
Benchmark combinato TTS + STT per SophyAI Voice Agent.

Output:
- benchmark/system_benchmark_<timestamp>/results.json
- benchmark/system_benchmark_<timestamp>/REPORT_BENCHMARK_TTS_STT.md
"""
from __future__ import annotations

import argparse
import io
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_TEXTS = [
    "Ciao, questa è una prova di sintesi vocale per misurare prestazioni e qualità dei motori disponibili.",
    "SophyAI deve rispondere rapidamente mantenendo una voce naturale e comprensibile durante una conversazione.",
    "Questo benchmark confronta latenza, stabilità e accuratezza della pipeline speech to text e text to speech.",
]


def _http_get_json(url: str, timeout_s: int = 30) -> tuple[int, dict[str, Any] | None, str | None]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(payload) if payload else {}, None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, None, body[:500]
    except Exception as e:  # pragma: no cover - runtime/network dependent
        return -1, None, str(e)


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: int = 120) -> tuple[int, bytes, dict[str, str], str | None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return resp.status, raw, headers, None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, b"", {}, body[:500]
    except Exception as e:  # pragma: no cover - runtime/network dependent
        return -1, b"", {}, str(e)


def _http_post_multipart(
    url: str,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
    timeout_s: int = 180,
) -> tuple[int, dict[str, Any] | None, str | None]:
    boundary = f"----sophyai-{uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )

    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_bytes)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(raw) if raw else {}, None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, None, body[:500]
    except Exception as e:  # pragma: no cover - runtime/network dependent
        return -1, None, str(e)


def _http_post_raw(
    url: str,
    raw_pcm_bytes: bytes,
    sample_rate: int,
    language: str,
    timeout_s: int = 180,
) -> tuple[int, dict[str, Any] | None, str | None]:
    boundary = f"----sophyai-{uuid4().hex}"
    chunks: list[bytes] = []

    def _add_field(name: str, value: str) -> None:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )

    _add_field("sample_rate", str(sample_rate))
    _add_field("language", language)

    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        (
            'Content-Disposition: form-data; name="samples"; filename="audio.pcm"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(raw_pcm_bytes)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            return resp.status, json.loads(payload) if payload else {}, None
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return e.code, None, body[:500]
    except Exception as e:  # pragma: no cover - runtime/network dependent
        return -1, None, str(e)


def _pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    with io.BytesIO() as bio:
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return bio.getvalue()


def _normalize_text(text: str) -> list[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [tok for tok in cleaned.split() if tok]


def _levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            sub_cost = previous[j - 1] + (0 if ca == cb else 1)
            current.append(min(insert_cost, delete_cost, sub_cost))
        previous = current
    return previous[-1]


def _wer(reference_text: str, hypothesis_text: str) -> float:
    ref_tokens = _normalize_text(reference_text)
    hyp_tokens = _normalize_text(hypothesis_text)
    if not ref_tokens:
        return 0.0
    distance = _levenshtein(ref_tokens, hyp_tokens)
    return distance / len(ref_tokens)


@dataclass
class TTSRun:
    engine: str
    text_idx: int
    run_idx: int
    ok: bool
    status: int
    elapsed_ms: float
    pcm_bytes: int
    audio_s: float
    rtf: float | None
    x_engine: str | None
    error: str | None


@dataclass
class STTRun:
    mode: str
    text_idx: int
    run_idx: int
    ok: bool
    status: int
    elapsed_ms: float
    recognized_text: str
    wer: float | None
    language: str | None
    error: str | None


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_p95(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, math.ceil(0.95 * len(sorted_vals)) - 1)
    return sorted_vals[idx]


def _format_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f} ms"


def _format_float(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _discover_tts_engines(tts_status: dict[str, Any] | None, fallback: list[str] | None = None) -> list[str]:
    if not isinstance(tts_status, dict):
        return list(fallback or ["edge"])

    engines = tts_status.get("available_engines")
    if isinstance(engines, list) and engines:
        return [str(x).strip().lower() for x in engines if str(x).strip()]

    routes = tts_status.get("routes")
    if isinstance(routes, dict) and routes:
        discovered: list[str] = []
        for key, route_info in routes.items():
            name = str(key).strip().lower()
            if not name:
                continue
            # Considera candidati solo se l'health route risponde.
            if isinstance(route_info, dict) and int(route_info.get("http_status", 0) or 0) == 200:
                discovered.append(name)
        if discovered:
            return discovered

    return list(fallback or ["edge"])


def _make_markdown_report(results: dict[str, Any]) -> str:
    ts = results["meta"]["timestamp"]
    tts_url = results["meta"]["tts_url"]
    stt_url = results["meta"]["stt_url"]
    runs_per_text = results["meta"]["runs_per_text"]

    lines: list[str] = []
    lines.append("# Report benchmark complessivo TTS + STT")
    lines.append("")
    lines.append(f"Data esecuzione: {ts}")
    lines.append(f"TTS endpoint: `{tts_url}`")
    lines.append(f"STT endpoint: `{stt_url}`")
    lines.append(f"Run per frase: `{runs_per_text}`")
    lines.append("")
    lines.append("## Contesto test")
    lines.append("- Le metriche TTS usano endpoint `/synthesize` e misurano latenza end-to-end.")
    lines.append("- Le metriche STT usano audio WAV/PCM generato in fase benchmark e endpoint Whisper.")
    lines.append("- Accuratezza STT misurata con WER (Word Error Rate): più basso = migliore.")
    lines.append("")

    lines.append("## Risultati TTS")
    tts_summary = results["tts"]["summary"]
    if not tts_summary:
        lines.append("- Nessun risultato TTS disponibile.")
    else:
        lines.append("| Engine | Successo | Latenza media | P95 | Audio medio | RTF medio |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in tts_summary:
            lines.append(
                f"| {row['engine']} | {row['success_rate']:.0%} | {_format_ms(row['latency_avg_ms'])} | "
                f"{_format_ms(row['latency_p95_ms'])} | {_format_float(row['audio_avg_s'], 2)} s | "
                f"{_format_float(row['rtf_avg'], 3)} |"
            )
    lines.append("")

    lines.append("## Risultati STT")
    stt_summary = results["stt"]["summary"]
    if not stt_summary:
        lines.append("- Nessun risultato STT disponibile.")
    else:
        lines.append("| Modalità | Successo | Latenza media | P95 | WER medio |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in stt_summary:
            lines.append(
                f"| {row['mode']} | {row['success_rate']:.0%} | {_format_ms(row['latency_avg_ms'])} | "
                f"{_format_ms(row['latency_p95_ms'])} | {_format_float(row['wer_avg'], 3)} |"
            )
    lines.append("")

    lines.append("## Note operative")
    lines.append("- Il benchmark non altera la configurazione runtime del sistema.")
    lines.append("- In caso di failure, verificare log e disponibilità modelli nei rispettivi container/server.")
    lines.append("- Per confronti nel tempo usare sempre stesso set frasi e stesso numero di run.")
    lines.append("")

    lines.append("## Frasi usate")
    for idx, text in enumerate(results["meta"]["texts"], start=1):
        lines.append(f"{idx}. {text}")
    lines.append("")

    return "\n".join(lines)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    tts_url = args.tts_url.rstrip("/")
    stt_url = args.stt_url.rstrip("/")
    texts = [t.strip() for t in args.texts if t.strip()]
    runs = args.runs

    results: dict[str, Any] = {
        "meta": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tts_url": tts_url,
            "stt_url": stt_url,
            "runs_per_text": runs,
            "texts": texts,
        },
        "services": {},
        "tts": {"runs": [], "summary": []},
        "stt": {"runs": [], "summary": []},
    }

    # Service discovery
    tts_status_code, tts_status, tts_err = _http_get_json(f"{tts_url}/status")
    stt_info_code, stt_info, stt_err = _http_get_json(f"{stt_url}/info")
    results["services"]["tts_status"] = {"status_code": tts_status_code, "payload": tts_status, "error": tts_err}
    results["services"]["stt_info"] = {"status_code": stt_info_code, "payload": stt_info, "error": stt_err}

    available_engines = _discover_tts_engines(tts_status)

    tts_runs: list[TTSRun] = []
    stt_runs: list[STTRun] = []
    stt_reference_audios: list[tuple[int, str, bytes, bytes]] = []  # text_idx, text, pcm, wav

    # TTS benchmark
    for engine in available_engines:
        for text_idx, text in enumerate(texts):
            for run_idx in range(runs):
                payload = {
                    "text": text,
                    "language": args.language,
                    "engine": engine,
                    "speaker": args.speaker,
                    "speed": 1.0,
                }
                t0 = time.perf_counter()
                status, pcm, headers, err = _http_post_json(
                    f"{tts_url}/synthesize",
                    payload=payload,
                    timeout_s=args.tts_timeout,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                x_engine = headers.get("x-engine")
                pcm_bytes = len(pcm)
                audio_s = pcm_bytes / (args.sample_rate * 2) if pcm_bytes else 0.0
                rtf = (elapsed_ms / 1000.0) / audio_s if audio_s > 0 else None
                ok = status == 200 and pcm_bytes > 0 and (x_engine == engine)

                tts_run = TTSRun(
                    engine=engine,
                    text_idx=text_idx,
                    run_idx=run_idx,
                    ok=ok,
                    status=status,
                    elapsed_ms=elapsed_ms,
                    pcm_bytes=pcm_bytes,
                    audio_s=audio_s,
                    rtf=rtf,
                    x_engine=x_engine,
                    error=None if ok else (err or f"status={status}, x_engine={x_engine}, bytes={pcm_bytes}"),
                )
                tts_runs.append(tts_run)

                # Primo audio valido di un engine "stabile" usato come riferimento STT.
                if engine == args.stt_reference_engine and run_idx == 0 and text_idx < len(texts) and ok:
                    wav_data = _pcm_to_wav_bytes(pcm, sample_rate=args.sample_rate)
                    stt_reference_audios.append((text_idx, text, pcm, wav_data))

    # fallback: se engine di riferimento non disponibile, usa primo audio TTS valido
    if not stt_reference_audios:
        for run in tts_runs:
            if run.ok:
                text = texts[run.text_idx]
                # recupera il PCM dal re-run specifico
                payload = {
                    "text": text,
                    "language": args.language,
                    "engine": run.engine,
                    "speaker": args.speaker,
                    "speed": 1.0,
                }
                status, pcm, _, _ = _http_post_json(
                    f"{tts_url}/synthesize",
                    payload=payload,
                    timeout_s=args.tts_timeout,
                )
                if status == 200 and pcm:
                    wav_data = _pcm_to_wav_bytes(pcm, sample_rate=args.sample_rate)
                    stt_reference_audios.append((run.text_idx, text, pcm, wav_data))
                if len(stt_reference_audios) >= len(texts):
                    break

    # STT benchmark (modalita file + raw)
    for text_idx, text, pcm_data, wav_data in stt_reference_audios:
        for run_idx in range(runs):
            # /transcribe con WAV
            t0 = time.perf_counter()
            code, payload, err = _http_post_multipart(
                f"{stt_url}/transcribe",
                fields={"language": args.language, "detect_language": "false"},
                file_field="audio",
                filename=f"text_{text_idx}.wav",
                file_bytes=wav_data,
                content_type="audio/wav",
                timeout_s=args.stt_timeout,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rec_text = (payload or {}).get("text", "") if isinstance(payload, dict) else ""
            rec_lang = (payload or {}).get("language") if isinstance(payload, dict) else None
            stt_error = (payload or {}).get("error") if isinstance(payload, dict) else None
            ok = code == 200 and not stt_error and bool(rec_text.strip())
            wer_value = _wer(text, rec_text) if ok else None
            stt_runs.append(
                STTRun(
                    mode="transcribe_wav",
                    text_idx=text_idx,
                    run_idx=run_idx,
                    ok=ok,
                    status=code,
                    elapsed_ms=elapsed_ms,
                    recognized_text=rec_text,
                    wer=wer_value,
                    language=rec_lang,
                    error=None if ok else (stt_error or err or f"status={code}"),
                )
            )

            # /transcribe_raw con PCM
            t0 = time.perf_counter()
            code, payload, err = _http_post_raw(
                f"{stt_url}/transcribe_raw",
                raw_pcm_bytes=pcm_data,
                sample_rate=args.sample_rate,
                language=args.language,
                timeout_s=args.stt_timeout,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rec_text = (payload or {}).get("text", "") if isinstance(payload, dict) else ""
            rec_lang = (payload or {}).get("language") if isinstance(payload, dict) else None
            stt_error = (payload or {}).get("error") if isinstance(payload, dict) else None
            ok = code == 200 and not stt_error and bool(rec_text.strip())
            wer_value = _wer(text, rec_text) if ok else None
            stt_runs.append(
                STTRun(
                    mode="transcribe_raw",
                    text_idx=text_idx,
                    run_idx=run_idx,
                    ok=ok,
                    status=code,
                    elapsed_ms=elapsed_ms,
                    recognized_text=rec_text,
                    wer=wer_value,
                    language=rec_lang,
                    error=None if ok else (stt_error or err or f"status={code}"),
                )
            )

    # Summary TTS
    tts_by_engine: dict[str, list[TTSRun]] = {}
    for r in tts_runs:
        tts_by_engine.setdefault(r.engine, []).append(r)

    for engine in sorted(tts_by_engine):
        rows = tts_by_engine[engine]
        successes = [r for r in rows if r.ok]
        latencies = [r.elapsed_ms for r in successes]
        audios = [r.audio_s for r in successes]
        rtfs = [r.rtf for r in successes if r.rtf is not None]
        results["tts"]["summary"].append(
            {
                "engine": engine,
                "runs_total": len(rows),
                "runs_ok": len(successes),
                "success_rate": (len(successes) / len(rows)) if rows else 0.0,
                "latency_avg_ms": _safe_mean(latencies),
                "latency_p95_ms": _safe_p95(latencies),
                "audio_avg_s": _safe_mean(audios),
                "rtf_avg": _safe_mean(rtfs),
            }
        )

    # Summary STT
    stt_by_mode: dict[str, list[STTRun]] = {}
    for r in stt_runs:
        stt_by_mode.setdefault(r.mode, []).append(r)

    for mode in sorted(stt_by_mode):
        rows = stt_by_mode[mode]
        successes = [r for r in rows if r.ok]
        latencies = [r.elapsed_ms for r in successes]
        wers = [r.wer for r in successes if r.wer is not None]
        results["stt"]["summary"].append(
            {
                "mode": mode,
                "runs_total": len(rows),
                "runs_ok": len(successes),
                "success_rate": (len(successes) / len(rows)) if rows else 0.0,
                "latency_avg_ms": _safe_mean(latencies),
                "latency_p95_ms": _safe_p95(latencies),
                "wer_avg": _safe_mean(wers),
            }
        )

    # Raw runs export
    results["tts"]["runs"] = [r.__dict__ for r in tts_runs]
    results["stt"]["runs"] = [r.__dict__ for r in stt_runs]

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark combinato TTS + STT")
    parser.add_argument("--tts-url", default="http://127.0.0.1:8092", help="Base URL TTS server")
    parser.add_argument("--stt-url", default="http://127.0.0.1:8091", help="Base URL STT server")
    parser.add_argument("--language", default="it", help="Lingua test (default: it)")
    parser.add_argument("--speaker", default="ryan", help="Speaker TTS dove applicabile")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Sample rate PCM TTS")
    parser.add_argument("--runs", type=int, default=2, help="Numero run per frase")
    parser.add_argument("--tts-timeout", type=int, default=180, help="Timeout chiamate TTS (s)")
    parser.add_argument("--stt-timeout", type=int, default=300, help="Timeout chiamate STT (s)")
    parser.add_argument(
        "--stt-reference-engine",
        default="edge",
        help="Engine TTS da usare per generare audio di riferimento STT",
    )
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        default=[],
        help="Frase custom (ripetibile). Se assente, usa frasi default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.texts:
        args.texts = list(DEFAULT_TEXTS)

    results = run_benchmark(args)

    out_dir = Path("benchmark") / f"system_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "results.json"
    md_path = out_dir / "REPORT_BENCHMARK_TTS_STT.md"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_make_markdown_report(results), encoding="utf-8")

    print(f"Benchmark completato.")
    print(f"- JSON: {json_path}")
    print(f"- Report: {md_path}")


if __name__ == "__main__":
    main()
