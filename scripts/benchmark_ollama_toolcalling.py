#!/usr/bin/env python3
"""
Benchmark Ollama models for:
- Tool calling + latency
- Quality Q&A receptionist (hybrid scoring: rules + LLM judge)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_QUESTIONS_FILE = "benchmark/receptionist_questions_50.json"
REALTIME_AVG_TARGET_MS = 800.0
REALTIME_P95_TARGET_MS = 1500.0
REALTIME_AVG_BORDERLINE_MS = 1500.0
REALTIME_P95_BORDERLINE_MS = 3000.0
REALTIME_TTFT_TARGET_MS = 400.0
REALTIME_TTFT_BORDERLINE_MS = 900.0


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Ritorna meteo corrente per una citta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_meeting",
            "description": "Prenota una riunione in calendario.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "time": {"type": "string"},
                    "person": {"type": "string"},
                },
                "required": ["date", "time", "person"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_invoice",
            "description": "Cerca una fattura per numero.",
            "parameters": {
                "type": "object",
                "properties": {"invoice_id": {"type": "string"}},
                "required": ["invoice_id"],
            },
        },
    },
]


@dataclass
class ToolTask:
    id: str
    prompt: str
    expected_tool: str | None
    required_args: list[str]
    expect_no_tool: bool = False


@dataclass
class QualityQuestion:
    id: str
    question: str
    intent: str
    required_points: list[str]
    forbidden_points: list[str]
    priority: str


TOOL_TASKS = [
    ToolTask(
        id="weather_rome",
        prompt="Devo sapere il meteo a Roma in celsius. Usa lo strumento corretto.",
        expected_tool="get_weather",
        required_args=["city", "unit"],
    ),
    ToolTask(
        id="meeting_booking",
        prompt=(
            "Prenota una riunione domani alle 15:30 con Giulia Rossi. "
            "Usa lo strumento corretto."
        ),
        expected_tool="book_meeting",
        required_args=["date", "time", "person"],
    ),
    ToolTask(
        id="invoice_lookup",
        prompt="Cerca la fattura INV-2026-0042 con lo strumento disponibile.",
        expected_tool="search_invoice",
        required_args=["invoice_id"],
    ),
    ToolTask(
        id="smalltalk_no_tool",
        prompt="Dimmi una frase motivazionale di una riga in italiano.",
        expected_tool=None,
        required_args=[],
        expect_no_tool=True,
    ),
]


ITALIAN_HINT_WORDS = {
    "il",
    "lo",
    "la",
    "gli",
    "le",
    "un",
    "una",
    "di",
    "che",
    "per",
    "con",
    "del",
    "della",
    "delle",
    "sono",
    "posso",
    "puo",
    "prenotazione",
    "camera",
    "check",
    "orario",
}


HALLUCINATION_RISK_PHRASES = [
    "garantito",
    "sicuro al cento per cento",
    "sempre disponibile",
    "mai un problema",
    "prezzo fisso sempre",
    "senza nessuna eccezione",
]


def post_json(url: str, payload: dict[str, Any], timeout_s: int = 240) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} su {url}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"Errore chiamata {url}: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Risposta non JSON da {url}: {raw[:400]}") from exc


def get_models(ollama_url: str) -> list[str]:
    req = request.Request(f"{ollama_url}/api/tags", method="GET")
    with request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in payload.get("models", [])]


def ns_to_ms(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value) / 1_000_000.0


def format_ms(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.0f} ms"
    return "n/d"


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) >= 20:
        return statistics.quantiles(values, n=20)[18]
    return max(values)


def _extract_message_text(response: dict[str, Any]) -> str:
    message = response.get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    # Fallback robusti: alcune varianti streaming usano campi top-level.
    for key in ("response", "content", "text"):
        value = response.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def call_chat(
    ollama_url: str,
    model: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    keep_alive: str = "30m",
    timeout_s: int = 240,
) -> tuple[dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": 0},
        "messages": messages,
    }
    if tools is not None:
        payload["tools"] = tools
    start = time.perf_counter()
    response = post_json(f"{ollama_url}/api/chat", payload, timeout_s=timeout_s)
    wall_ms = (time.perf_counter() - start) * 1000.0
    return response, wall_ms


def call_chat_stream(
    ollama_url: str,
    model: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]] | None = None,
    keep_alive: str = "30m",
    timeout_s: int = 240,
) -> tuple[dict[str, Any], float, float | None, float]:
    payload: dict[str, Any] = {
        "model": model,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {"temperature": 0},
        "messages": messages,
    }
    if tools is not None:
        payload["tools"] = tools

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=f"{ollama_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    chunks: list[str] = []
    final_payload: dict[str, Any] | None = None
    ttft_ms: float | None = None
    start = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                raw = line.decode("utf-8", errors="replace").strip()
                if not raw:
                    continue
                try:
                    part = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg = part.get("message") or {}
                content: str = ""
                msg_content = msg.get("content")
                if isinstance(msg_content, str):
                    content = msg_content
                elif isinstance(msg_content, list):
                    # Compat: alcuni backend serializzano il contenuto come lista di segmenti.
                    content = "".join(
                        str(seg.get("text") or "")
                        for seg in msg_content
                        if isinstance(seg, dict)
                    )

                if not content:
                    # Fallback per varianti generate-style o provider compatibili.
                    for key in ("response", "content", "text"):
                        value = part.get(key)
                        if isinstance(value, str) and value:
                            content = value
                            break
                if content:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000.0
                    chunks.append(content)

                if part.get("done"):
                    final_payload = part
                    break
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} su {ollama_url}/api/chat: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"Errore chiamata streaming {ollama_url}/api/chat: {exc}") from exc

    wall_ms = (time.perf_counter() - start) * 1000.0
    stream_total_ms = wall_ms
    collected_content = "".join(chunks).strip()
    final_message = {"role": "assistant", "content": collected_content}

    response: dict[str, Any] = {
        "message": final_message,
        "total_duration": None,
        "load_duration": None,
        "prompt_eval_duration": None,
        "eval_duration": None,
        "eval_count": None,
        "prompt_eval_count": None,
    }
    if isinstance(final_payload, dict):
        for key in (
            "total_duration",
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
            "eval_count",
            "prompt_eval_count",
        ):
            response[key] = final_payload.get(key)
        final_msg = final_payload.get("message")
        if isinstance(final_msg, dict):
            final_content = str(final_msg.get("content") or "").strip()
            if not final_content:
                for key in ("response", "content", "text"):
                    value = final_payload.get(key)
                    if isinstance(value, str) and value.strip():
                        final_content = value.strip()
                        break
            merged_content = collected_content or final_content
            response["message"] = {
                **final_msg,
                "content": merged_content,
            }
    return response, wall_ms, ttft_ms, stream_total_ms


def validate_tool_call(task: ToolTask, response: dict[str, Any]) -> tuple[bool, str]:
    message = response.get("message", {})
    tool_calls = message.get("tool_calls") or []

    if task.expect_no_tool:
        if tool_calls:
            return False, "Tool chiamato ma non richiesto"
        return True, "Corretto: nessun tool chiamato"

    if not tool_calls:
        return False, "Nessun tool call presente"

    first = tool_calls[0]
    fn = (first.get("function") or {}).get("name")
    if fn != task.expected_tool:
        return False, f"Tool errato: atteso={task.expected_tool}, ottenuto={fn}"

    args = (first.get("function") or {}).get("arguments") or {}
    if not isinstance(args, dict):
        return False, "Argomenti non in formato oggetto"

    missing = [arg for arg in task.required_args if arg not in args]
    if missing:
        return False, f"Argomenti mancanti: {', '.join(missing)}"

    return True, "Tool e argomenti validi"


def benchmark_tool_model(
    ollama_url: str,
    model: str,
    repeats: int,
    warmups: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        call_chat(
            ollama_url=ollama_url,
            model=model,
            messages=[{"role": "user", "content": "Scrivi solo pronto"}],
            tools=[],
            keep_alive="2h",
        )

    rows: list[dict[str, Any]] = []
    for task in TOOL_TASKS:
        for run in range(1, repeats + 1):
            response, wall_ms = call_chat(
                ollama_url=ollama_url,
                model=model,
                tools=TOOL_DEFINITIONS,
                keep_alive="2h",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Sei un assistente che deve usare tool quando necessario. "
                            "Se un tool e richiesto rispondi con tool call."
                        ),
                    },
                    {"role": "user", "content": task.prompt},
                ],
            )
            ok, note = validate_tool_call(task, response)
            rows.append(
                {
                    "task_id": task.id,
                    "run": run,
                    "tool_call_ok": ok,
                    "validation_note": note,
                    "wall_ms": wall_ms,
                    "total_duration_ms": ns_to_ms(response.get("total_duration")),
                    "load_duration_ms": ns_to_ms(response.get("load_duration")),
                    "prompt_eval_duration_ms": ns_to_ms(response.get("prompt_eval_duration")),
                    "eval_duration_ms": ns_to_ms(response.get("eval_duration")),
                    "eval_count": response.get("eval_count"),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "raw_tool_calls": (response.get("message") or {}).get("tool_calls"),
                }
            )

    latencies = [r["wall_ms"] for r in rows]
    tool_tasks_rows = [r for r in rows if r["task_id"] != "smalltalk_no_tool"]
    all_tool_policy_rows = rows

    tool_ok = sum(1 for r in tool_tasks_rows if r["tool_call_ok"])
    tool_total = max(1, len(tool_tasks_rows))
    policy_ok = sum(1 for r in all_tool_policy_rows if r["tool_call_ok"])
    policy_total = max(1, len(all_tool_policy_rows))

    token_s_values: list[float] = []
    for r in rows:
        eval_count = r.get("eval_count")
        eval_duration_ms = r.get("eval_duration_ms")
        if isinstance(eval_count, int) and eval_duration_ms and eval_duration_ms > 0:
            token_s_values.append(eval_count / (eval_duration_ms / 1000.0))

    return {
        "model": model,
        "runs": rows,
        "metrics": {
            "avg_wall_ms": statistics.mean(latencies),
            "p50_wall_ms": statistics.median(latencies),
            "p95_wall_ms": (
                statistics.quantiles(latencies, n=20)[18]
                if len(latencies) >= 20
                else max(latencies)
            ),
            "tool_call_success_rate": tool_ok / tool_total,
            "tool_policy_success_rate": policy_ok / policy_total,
            "avg_tokens_per_sec": statistics.mean(token_s_values) if token_s_values else None,
        },
    }


def rank_tool_models(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    min_latency = min(r["metrics"]["avg_wall_ms"] for r in results)
    max_latency = max(r["metrics"]["avg_wall_ms"] for r in results)
    denom = max(1e-9, max_latency - min_latency)
    ranked = []
    for r in results:
        success = r["metrics"]["tool_policy_success_rate"]
        latency = r["metrics"]["avg_wall_ms"]
        latency_score = 1.0 - ((latency - min_latency) / denom)
        score = 0.65 * success + 0.35 * latency_score
        ranked.append(
            {
                "model": r["model"],
                "score": score,
                "tool_policy_success_rate": success,
                "avg_wall_ms": latency,
                "avg_tokens_per_sec": r["metrics"]["avg_tokens_per_sec"],
            }
        )
    return sorted(ranked, key=lambda x: x["score"], reverse=True)


def write_tool_markdown_report(
    path: Path,
    ollama_url: str,
    repeats: int,
    warmups: int,
    results: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    lines.append("# Benchmark Ollama: Tool Calling + Latenza")
    lines.append("")
    lines.append(f"- Data UTC: `{now}`")
    lines.append(f"- Endpoint: `{ollama_url}`")
    lines.append(f"- Repeats per task: `{repeats}`")
    lines.append(f"- Warmup per modello: `{warmups}`")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    for i, row in enumerate(ranking, start=1):
        tok_s = row["avg_tokens_per_sec"]
        tok_s_txt = f"{tok_s:.1f}" if isinstance(tok_s, (int, float)) else "n/d"
        lines.append(
            f"{i}. **{row['model']}** - score={row['score']:.3f}, "
            f"tool_policy={row['tool_policy_success_rate']:.1%}, "
            f"avg_latency={row['avg_wall_ms']:.0f} ms, "
            f"tok/s={tok_s_txt}"
        )
    lines.append("")
    lines.append("## Dettaglio per modello")
    lines.append("")
    for item in results:
        m = item["metrics"]
        lines.append(f"### {item['model']}")
        lines.append(
            f"- Tool success (solo task con tool): **{m['tool_call_success_rate']:.1%}**"
        )
        lines.append(
            f"- Tool policy success (incluso no-tool): **{m['tool_policy_success_rate']:.1%}**"
        )
        lines.append(f"- Avg latency: **{m['avg_wall_ms']:.0f} ms**")
        lines.append(f"- P50 latency: **{m['p50_wall_ms']:.0f} ms**")
        lines.append(f"- P95 latency: **{m['p95_wall_ms']:.0f} ms**")
        if m["avg_tokens_per_sec"] is not None:
            lines.append(f"- Avg tokens/s: **{m['avg_tokens_per_sec']:.1f}**")
        else:
            lines.append("- Avg tokens/s: **n/d**")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_receptionist_system_prompt() -> str:
    return (
        "Sei Receptionist assistente vocale ultra veloce.\n"
        "Priorita assoluta velocita e sintesi.\n\n"
        "Regole fondamentali\n"
        "Risposte ultra brevi massimo due frasi e mai oltre trenta parole.\n"
        "Vai dritto al punto senza preamboli saluti inutili o ripetizioni.\n"
        "Rispondi nella stessa lingua dell utente.\n\n"
        "Stile\n"
        "Rispondi come receptionist professionale diretto chiaro utile.\n"
        "Se non sai qualcosa dillo chiaramente e proponi il prossimo passo.\n"
        "Preferisci risposte secche e precise.\n\n"
        "Formato tts\n"
        "Non usare simboli speciali.\n"
        "Non usare emoji.\n"
        "Non usare elenchi puntati scrivi in modo discorsivo."
    )


def load_quality_questions(path: Path) -> list[QualityQuestion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions: list[QualityQuestion] = []
    for item in raw:
        questions.append(
            QualityQuestion(
                id=str(item["id"]),
                question=str(item["question"]),
                intent=str(item["intent"]),
                required_points=[str(x).lower() for x in item.get("required_points", [])],
                forbidden_points=[str(x).lower() for x in item.get("forbidden_points", [])],
                priority=str(item.get("priority", "medium")),
            )
        )
    return questions


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _looks_italian(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    if not tokens:
        return False
    score = sum(1 for t in tokens if t in ITALIAN_HINT_WORDS)
    return score >= 3 or (score >= 2 and len(tokens) < 10)


def _sentence_count(text: str) -> int:
    chunks = [x.strip() for x in re.split(r"[.!?]+", text) if x.strip()]
    return len(chunks)


def _has_markdown_bullets(text: str) -> bool:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            return True
        if re.match(r"^\d+\.\s+", stripped):
            return True
    return False


def evaluate_rule_based(question: QualityQuestion, answer_text: str) -> dict[str, Any]:
    answer_norm = _normalize_text(answer_text)

    required_matches = [p for p in question.required_points if p and p in answer_norm]
    forbidden_matches = [p for p in question.forbidden_points if p and p in answer_norm]

    required_ratio = (
        len(required_matches) / len(question.required_points)
        if question.required_points
        else 1.0
    )
    forbidden_ratio = (
        len(forbidden_matches) / len(question.forbidden_points)
        if question.forbidden_points
        else 0.0
    )

    is_italian = _looks_italian(answer_text)
    sentences = _sentence_count(answer_text)
    short_enough = 1 <= sentences <= 3 and len(answer_text.split()) <= 35
    no_bullets = not _has_markdown_bullets(answer_text)

    policy_checks = [is_italian, short_enough, no_bullets]
    policy_score = 100.0 * (sum(1 for x in policy_checks if x) / len(policy_checks))

    base_content_score = max(0.0, (required_ratio * 100.0) - (forbidden_ratio * 70.0))

    risk_hits = [p for p in HALLUCINATION_RISK_PHRASES if p in answer_norm]
    hallucination_risk = min(1.0, (len(risk_hits) + len(forbidden_matches)) / 4.0)
    hallucination_penalty = hallucination_risk * 20.0

    final_rule_score = max(
        0.0,
        min(100.0, 0.7 * base_content_score + 0.3 * policy_score - hallucination_penalty),
    )

    return {
        "rule_score": final_rule_score,
        "required_match_ratio": required_ratio,
        "required_matches": required_matches,
        "forbidden_matches": forbidden_matches,
        "policy_score": policy_score,
        "is_italian": is_italian,
        "sentence_count": sentences,
        "short_enough": short_enough,
        "no_markdown_bullets": no_bullets,
        "hallucination_risk": hallucination_risk,
        "hallucination_hits": risk_hits,
    }


def _extract_json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        direct = json.loads(text)
        if isinstance(direct, dict):
            return direct
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def judge_with_llm(
    ollama_url: str,
    judge_model: str,
    system_context: str,
    question: QualityQuestion,
    answer_text: str,
) -> dict[str, Any]:
    judge_prompt = (
        "Valuta la risposta di un receptionist AI con punteggio da uno a dieci.\n"
        "Restituisci SOLO JSON con chiavi: score clarity usefulness tone safety correctness "
        "hallucination_risk notes.\n"
        "score e i sottopunteggi devono essere numeri tra uno e dieci.\n"
        "hallucination_risk deve essere numero tra zero e uno.\n\n"
        f"Contesto sistema:\n{system_context}\n\n"
        f"Domanda cliente: {question.question}\n"
        f"Intento: {question.intent}\n"
        f"Punti richiesti: {', '.join(question.required_points)}\n"
        f"Punti vietati: {', '.join(question.forbidden_points)}\n"
        f"Risposta modello: {answer_text}\n"
    )
    response, wall_ms = call_chat(
        ollama_url=ollama_url,
        model=judge_model,
        keep_alive="2h",
        timeout_s=240,
        messages=[
            {
                "role": "system",
                "content": "Sei un valutatore severo e coerente. Produci solo JSON valido.",
            },
            {"role": "user", "content": judge_prompt},
        ],
    )
    content = _extract_message_text(response)
    parsed = _extract_json_from_text(content) or {}

    def _num(key: str, default: float) -> float:
        raw = parsed.get(key, default)
        if isinstance(raw, (int, float)):
            return float(raw)
        try:
            return float(str(raw).strip())
        except Exception:
            return default

    score = max(1.0, min(10.0, _num("score", 5.0)))
    hallucination_risk = max(0.0, min(1.0, _num("hallucination_risk", 0.5)))
    return {
        "judge_score_0_100": score * 10.0,
        "judge_latency_ms": wall_ms,
        "judge_raw": parsed,
        "judge_notes": str(parsed.get("notes", "")).strip(),
        "judge_hallucination_risk": hallucination_risk,
    }


def benchmark_quality_model(
    ollama_url: str,
    model: str,
    questions: list[QualityQuestion],
    quality_repeats: int,
    warmups: int,
    judge_model: str,
    system_prompt: str,
    verbose_questions: bool,
) -> dict[str, Any]:
    for _ in range(warmups):
        call_chat(
            ollama_url=ollama_url,
            model=model,
            keep_alive="2h",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Dimmi solo pronto"},
            ],
        )

    rows: list[dict[str, Any]] = []
    total_q = len(questions) * quality_repeats
    current_idx = 0

    for question in questions:
        for run in range(1, quality_repeats + 1):
            current_idx += 1
            if verbose_questions:
                print(f"[QUALITY][{model}] [Q{current_idx}/{total_q}] {question.question}")

            response, wall_ms, ttft_ms, stream_total_ms = call_chat_stream(
                ollama_url=ollama_url,
                model=model,
                keep_alive="2h",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question.question},
                ],
                timeout_s=240,
            )
            answer_text = _extract_message_text(response)
            rule_eval = evaluate_rule_based(question, answer_text)
            judge_eval = judge_with_llm(
                ollama_url=ollama_url,
                judge_model=judge_model,
                system_context=system_prompt,
                question=question,
                answer_text=answer_text,
            )

            final_score = (
                (rule_eval["rule_score"] * 0.65)
                + (judge_eval["judge_score_0_100"] * 0.35)
            )
            hallucination_risk = (
                (rule_eval["hallucination_risk"] * 0.6)
                + (judge_eval["judge_hallucination_risk"] * 0.4)
            )

            if verbose_questions:
                ttft_txt = f"{ttft_ms:.0f}ms" if ttft_ms is not None else "n/d"
                print(
                    f"[QUALITY][{model}] -> latency={wall_ms:.0f}ms ttft={ttft_txt} "
                    f"stream_total={stream_total_ms:.0f}ms "
                    f"rule={rule_eval['rule_score']:.1f} "
                    f"judge={judge_eval['judge_score_0_100']:.1f} "
                    f"final={final_score:.1f}"
                )

            rows.append(
                {
                    "question_id": question.id,
                    "question": question.question,
                    "intent": question.intent,
                    "priority": question.priority,
                    "run": run,
                    "answer": answer_text,
                    "wall_ms": wall_ms,
                    "ttft_ms": ttft_ms,
                    "stream_total_ms": stream_total_ms,
                    "total_duration_ms": ns_to_ms(response.get("total_duration")),
                    "load_duration_ms": ns_to_ms(response.get("load_duration")),
                    "prompt_eval_duration_ms": ns_to_ms(response.get("prompt_eval_duration")),
                    "eval_duration_ms": ns_to_ms(response.get("eval_duration")),
                    "eval_count": response.get("eval_count"),
                    "prompt_eval_count": response.get("prompt_eval_count"),
                    "rule_eval": rule_eval,
                    "judge_eval": judge_eval,
                    "final_quality_score": final_score,
                    "hallucination_risk": hallucination_risk,
                }
            )

    latencies = [r["wall_ms"] for r in rows]
    ttft_values = [r["ttft_ms"] for r in rows if isinstance(r.get("ttft_ms"), (int, float))]
    stream_total_values = [
        r["stream_total_ms"] for r in rows if isinstance(r.get("stream_total_ms"), (int, float))
    ]
    scores = [r["final_quality_score"] for r in rows]
    policy_ok = [
        (r["rule_eval"]["is_italian"] and r["rule_eval"]["short_enough"] and r["rule_eval"]["no_markdown_bullets"])
        for r in rows
    ]
    hallucination_flags = [r["hallucination_risk"] >= 0.5 for r in rows]

    token_s_values: list[float] = []
    for r in rows:
        eval_count = r.get("eval_count")
        eval_duration_ms = r.get("eval_duration_ms")
        if isinstance(eval_count, int) and eval_duration_ms and eval_duration_ms > 0:
            token_s_values.append(eval_count / (eval_duration_ms / 1000.0))

    sorted_rows = sorted(rows, key=lambda x: x["final_quality_score"], reverse=True)
    examples = []
    if sorted_rows:
        take_top = sorted_rows[:2]
        take_bottom = sorted_rows[-2:] if len(sorted_rows) >= 2 else []
        take_mid = [sorted_rows[len(sorted_rows) // 2]]
        merged = take_top + take_mid + take_bottom
        used = set()
        for row in merged:
            key = (row["question_id"], row["run"])
            if key in used:
                continue
            used.add(key)
            examples.append(
                {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "final_quality_score": row["final_quality_score"],
                    "rule_score": row["rule_eval"]["rule_score"],
                    "judge_score": row["judge_eval"]["judge_score_0_100"],
                    "note": row["judge_eval"]["judge_notes"] or "Nessuna nota",
                }
            )

    return {
        "model": model,
        "runs": rows,
        "examples": examples[:5],
        "metrics": {
            "quality_avg": statistics.mean(scores),
            "quality_p50": statistics.median(scores),
            "quality_p95": (
                statistics.quantiles(scores, n=20)[18] if len(scores) >= 20 else max(scores)
            ),
            "policy_compliance_rate": sum(1 for x in policy_ok if x) / max(1, len(policy_ok)),
            "hallucination_risk_rate": (
                sum(1 for x in hallucination_flags if x) / max(1, len(hallucination_flags))
            ),
            "avg_wall_ms": statistics.mean(latencies),
            "p50_wall_ms": statistics.median(latencies),
            "p95_wall_ms": (
                statistics.quantiles(latencies, n=20)[18]
                if len(latencies) >= 20
                else max(latencies)
            ),
            "avg_ttft_ms": (statistics.mean(ttft_values) if ttft_values else None),
            "p50_ttft_ms": (statistics.median(ttft_values) if ttft_values else None),
            "p95_ttft_ms": (
                statistics.quantiles(ttft_values, n=20)[18]
                if len(ttft_values) >= 20
                else (max(ttft_values) if ttft_values else None)
            ),
            "avg_stream_total_ms": (
                statistics.mean(stream_total_values) if stream_total_values else None
            ),
            "avg_tokens_per_sec": statistics.mean(token_s_values) if token_s_values else None,
        },
    }


def rank_quality_models(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    return sorted(
        [
            {
                "model": r["model"],
                "quality_avg": r["metrics"]["quality_avg"],
                "policy_compliance_rate": r["metrics"]["policy_compliance_rate"],
                "hallucination_risk_rate": r["metrics"]["hallucination_risk_rate"],
                "avg_wall_ms": r["metrics"]["avg_wall_ms"],
                "avg_ttft_ms": r["metrics"].get("avg_ttft_ms"),
                "avg_tokens_per_sec": r["metrics"]["avg_tokens_per_sec"],
            }
            for r in results
        ],
        key=lambda x: x["quality_avg"],
        reverse=True,
    )


def rank_quality_latency(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return []
    min_latency = min(r["metrics"]["avg_wall_ms"] for r in results)
    max_latency = max(r["metrics"]["avg_wall_ms"] for r in results)
    ttft_candidates = [
        r["metrics"]["avg_ttft_ms"]
        for r in results
        if isinstance(r["metrics"].get("avg_ttft_ms"), (int, float))
    ]
    min_ttft = min(ttft_candidates) if ttft_candidates else None
    max_ttft = max(ttft_candidates) if ttft_candidates else None
    denom = max(1e-9, max_latency - min_latency)
    ttft_denom = (
        max(1e-9, float(max_ttft - min_ttft))
        if min_ttft is not None and max_ttft is not None
        else None
    )
    ranked = []
    for r in results:
        q = r["metrics"]["quality_avg"] / 100.0
        l = r["metrics"]["avg_wall_ms"]
        latency_score = 1.0 - ((l - min_latency) / denom)
        avg_ttft = r["metrics"].get("avg_ttft_ms")
        if (
            isinstance(avg_ttft, (int, float))
            and min_ttft is not None
            and ttft_denom is not None
        ):
            ttft_score = 1.0 - ((float(avg_ttft) - float(min_ttft)) / ttft_denom)
        else:
            ttft_score = 0.5
        combined = 0.70 * q + 0.20 * latency_score + 0.10 * ttft_score
        ranked.append(
            {
                "model": r["model"],
                "combined_score": combined,
                "quality_avg": r["metrics"]["quality_avg"],
                "avg_wall_ms": l,
                "avg_ttft_ms": avg_ttft,
                "policy_compliance_rate": r["metrics"]["policy_compliance_rate"],
            }
        )
    return sorted(ranked, key=lambda x: x["combined_score"], reverse=True)


def _concurrency_probe_request(
    ollama_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: int,
) -> dict[str, Any]:
    try:
        response, wall_ms = call_chat(
            ollama_url=ollama_url,
            model=model,
            keep_alive="2h",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout_s=timeout_s,
        )
        answer_text = _extract_message_text(response)
        ok = bool(answer_text.strip())
        return {
            "ok": ok,
            "wall_ms": wall_ms,
            "answer_chars": len(answer_text),
            "error": "" if ok else "risposta_vuota",
        }
    except Exception as exc:
        return {
            "ok": False,
            "wall_ms": None,
            "answer_chars": 0,
            "error": str(exc),
        }


def benchmark_concurrency_model(
    ollama_url: str,
    model: str,
    system_prompt: str,
    warmups: int,
    max_workers: int,
    rounds_per_level: int,
    timeout_s: int,
    min_success_rate: float,
    degradation_factor: float,
    realtime_avg_threshold_ms: float,
    realtime_p95_threshold_ms: float,
    stop_on_realtime_degradation: bool,
) -> dict[str, Any]:
    for _ in range(warmups):
        call_chat(
            ollama_url=ollama_url,
            model=model,
            keep_alive="2h",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Rispondi solo ok"},
            ],
        )

    levels: list[dict[str, Any]] = []
    baseline_avg_ms: float | None = None
    baseline_p95_ms: float | None = None
    max_stable_concurrency = 0
    stable_reasons: list[str] = []
    stopped_early_for_realtime = False
    stop_reason = ""

    for level in range(1, max(1, max_workers) + 1):
        started = time.perf_counter()
        requests_total = level * max(1, rounds_per_level)
        ok_count = 0
        errors: list[str] = []
        latencies: list[float] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
            futures: list[concurrent.futures.Future[dict[str, Any]]] = []
            for round_idx in range(max(1, rounds_per_level)):
                for worker_idx in range(level):
                    futures.append(
                        executor.submit(
                            _concurrency_probe_request,
                            ollama_url,
                            model,
                            system_prompt,
                            (
                                "Conferma disponibilita reception con una frase breve "
                                f"test={level}-{round_idx}-{worker_idx}"
                            ),
                            timeout_s,
                        )
                    )

            for future in futures:
                result = future.result()
                if result["ok"]:
                    ok_count += 1
                    latencies.append(float(result["wall_ms"]))
                elif result["error"]:
                    errors.append(result["error"])

        elapsed_s = max(0.001, time.perf_counter() - started)
        success_rate = ok_count / max(1, requests_total)
        avg_latency_ms = statistics.mean(latencies) if latencies else None
        p95_latency_ms = p95(latencies) if latencies else None
        throughput_rps = ok_count / elapsed_s

        if level == 1:
            baseline_avg_ms = avg_latency_ms or 1.0
            baseline_p95_ms = p95_latency_ms or baseline_avg_ms

        baseline_avg = baseline_avg_ms or 1.0
        baseline_p95 = baseline_p95_ms or baseline_avg
        avg_limit = baseline_avg * degradation_factor
        p95_limit = baseline_p95 * degradation_factor
        realtime_compatible = (
            avg_latency_ms is not None
            and p95_latency_ms is not None
            and avg_latency_ms <= realtime_avg_threshold_ms
            and p95_latency_ms <= realtime_p95_threshold_ms
        )

        stable = (
            success_rate >= min_success_rate
            and avg_latency_ms is not None
            and p95_latency_ms is not None
            and avg_latency_ms <= avg_limit
            and p95_latency_ms <= p95_limit
        )
        if stable:
            max_stable_concurrency = level
        else:
            if success_rate < min_success_rate:
                stable_reasons.append(
                    f"livello={level}: success_rate {success_rate:.1%} sotto soglia {min_success_rate:.1%}"
                )
            elif avg_latency_ms is not None and p95_latency_ms is not None:
                stable_reasons.append(
                    f"livello={level}: latenza oltre limite (avg {avg_latency_ms:.0f}>{avg_limit:.0f} o p95 {p95_latency_ms:.0f}>{p95_limit:.0f})"
                )

        print(
            f"[CONCURRENCY][{model}] level={level} req={requests_total} "
            f"ok={ok_count} success={success_rate:.1%} "
            f"avg={format_ms(avg_latency_ms)} p95={format_ms(p95_latency_ms)} "
            f"rps={throughput_rps:.2f} stable={stable} realtime={realtime_compatible}"
        )

        levels.append(
            {
                "level": level,
                "requests_total": requests_total,
                "ok_count": ok_count,
                "error_count": requests_total - ok_count,
                "success_rate": success_rate,
                "avg_wall_ms": avg_latency_ms,
                "p95_wall_ms": p95_latency_ms,
                "throughput_rps": throughput_rps,
                "stable": stable,
                "realtime_compatible": realtime_compatible,
                "sample_errors": errors[:3],
            }
        )

        if stop_on_realtime_degradation and not realtime_compatible:
            stopped_early_for_realtime = True
            stop_reason = (
                f"livello={level}: fuori soglia realtime "
                f"(avg {format_ms(avg_latency_ms)} > {realtime_avg_threshold_ms:.0f} ms "
                f"o p95 {format_ms(p95_latency_ms)} > {realtime_p95_threshold_ms:.0f} ms)"
            )
            stable_reasons.append(stop_reason)
            print(f"[CONCURRENCY][{model}] early-stop: {stop_reason}")
            break

    stable_levels = [x for x in levels if x["stable"]]
    best_throughput = max((x["throughput_rps"] for x in stable_levels), default=0.0)
    recommended = max(1, int(max_stable_concurrency * 0.8)) if max_stable_concurrency else 1

    return {
        "model": model,
        "levels": levels,
        "metrics": {
            "max_tested_concurrency": max_workers,
            "max_stable_concurrency": max_stable_concurrency,
            "recommended_concurrency": recommended,
            "best_stable_throughput_rps": best_throughput,
            "baseline_avg_ms": baseline_avg_ms,
            "baseline_p95_ms": baseline_p95_ms,
            "min_success_rate": min_success_rate,
            "degradation_factor": degradation_factor,
            "realtime_avg_threshold_ms": realtime_avg_threshold_ms,
            "realtime_p95_threshold_ms": realtime_p95_threshold_ms,
            "stop_on_realtime_degradation": stop_on_realtime_degradation,
            "stopped_early_for_realtime": stopped_early_for_realtime,
            "tested_levels": len(levels),
            "stop_reason": stop_reason,
            "stability_notes": stable_reasons[:5],
        },
    }


def rank_concurrency_models(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [
        {
            "model": item["model"],
            "max_stable_concurrency": item["metrics"]["max_stable_concurrency"],
            "recommended_concurrency": item["metrics"]["recommended_concurrency"],
            "best_stable_throughput_rps": item["metrics"]["best_stable_throughput_rps"],
            "baseline_avg_ms": item["metrics"]["baseline_avg_ms"],
        }
        for item in results
    ]
    return sorted(
        ranked,
        key=lambda x: (
            x["max_stable_concurrency"],
            x["best_stable_throughput_rps"],
            -(x["baseline_avg_ms"] or 999999),
        ),
        reverse=True,
    )


def write_concurrency_markdown_report(
    path: Path,
    ollama_url: str,
    rounds_per_level: int,
    max_workers: int,
    min_success_rate: float,
    degradation_factor: float,
    realtime_avg_threshold_ms: float,
    realtime_p95_threshold_ms: float,
    stop_on_realtime_degradation: bool,
    results: list[dict[str, Any]],
    ranking: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Benchmark Ollama: Concorrenza Massima per Modello")
    lines.append("")
    lines.append(f"- Data UTC: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- Endpoint: `{ollama_url}`")
    lines.append(f"- Livello massimo testato: `{max_workers}`")
    lines.append(f"- Round per livello: `{rounds_per_level}`")
    lines.append(f"- Soglia successo minima: `{min_success_rate:.1%}`")
    lines.append(f"- Fattore degrado latenza consentito: `{degradation_factor:.1f}x`")
    lines.append(
        f"- Soglia realtime avg/p95: `{realtime_avg_threshold_ms:.0f} ms / {realtime_p95_threshold_ms:.0f} ms`"
    )
    lines.append(f"- Early-stop su degrado realtime: `{stop_on_realtime_degradation}`")
    lines.append("")
    lines.append("## Ranking concorrenza stabile")
    lines.append("")
    for i, row in enumerate(ranking, start=1):
        lines.append(
            f"{i}. **{row['model']}** - max_stable={row['max_stable_concurrency']}, "
            f"recommended={row['recommended_concurrency']}, "
            f"best_throughput={row['best_stable_throughput_rps']:.2f} req/s, "
            f"baseline_avg={format_ms(row['baseline_avg_ms'])}"
        )
    lines.append("")
    lines.append("## Dettaglio per modello")
    lines.append("")
    for item in results:
        m = item["metrics"]
        lines.append(f"### {item['model']}")
        lines.append(f"- Max stable concurrency: **{m['max_stable_concurrency']}**")
        lines.append(f"- Recommended concurrency: **{m['recommended_concurrency']}**")
        lines.append(f"- Best stable throughput: **{m['best_stable_throughput_rps']:.2f} req/s**")
        lines.append(
            f"- Baseline latency: **avg {format_ms(m['baseline_avg_ms'])}, p95 {format_ms(m['baseline_p95_ms'])}**"
        )
        lines.append(f"- Tested levels: **{m.get('tested_levels', len(item.get('levels', [])))}**")
        lines.append(
            f"- Early-stop realtime: **{m.get('stopped_early_for_realtime', False)}**"
        )
        if m.get("stop_reason"):
            lines.append(f"- Stop reason: **{m['stop_reason']}**")
        if m.get("stability_notes"):
            lines.append("- Note stabilita:")
            for note in m["stability_notes"]:
                lines.append(f"  - {note}")
        lines.append("")
        lines.append("#### Livelli testati")
        for lvl in item.get("levels", []):
            lines.append(
                f"- level={lvl['level']}, req={lvl['requests_total']}, "
                f"success={lvl['success_rate']:.1%}, avg={format_ms(lvl['avg_wall_ms'])}, "
                f"p95={format_ms(lvl['p95_wall_ms'])}, rps={lvl['throughput_rps']:.2f}, "
                f"stable={lvl['stable']}, realtime={lvl.get('realtime_compatible')}"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_realtime_profile(avg_ms: float, p95_ms: float) -> dict[str, Any]:
    return _evaluate_realtime_profile_with_ttft(avg_ms, p95_ms, None, None)


def _evaluate_realtime_profile_with_ttft(
    avg_ms: float,
    p95_ms: float,
    avg_ttft_ms: float | None,
    p95_ttft_ms: float | None,
) -> dict[str, Any]:
    ttft_good = (
        avg_ttft_ms is not None
        and p95_ttft_ms is not None
        and avg_ttft_ms <= REALTIME_TTFT_TARGET_MS
        and p95_ttft_ms <= REALTIME_TTFT_BORDERLINE_MS
    )
    ttft_borderline = (
        avg_ttft_ms is not None
        and p95_ttft_ms is not None
        and avg_ttft_ms <= REALTIME_TTFT_BORDERLINE_MS
        and p95_ttft_ms <= (REALTIME_TTFT_BORDERLINE_MS * 1.6)
    )

    if (
        avg_ms <= REALTIME_AVG_TARGET_MS
        and p95_ms <= REALTIME_P95_TARGET_MS
        and (avg_ttft_ms is None or ttft_good)
    ):
        return {
            "label": "realtime_ready",
            "compatible_low_latency": True,
            "reason": "Tempo prima risposta e latenza totale entro target realtime.",
        }
    if (
        avg_ms <= REALTIME_AVG_BORDERLINE_MS
        and p95_ms <= REALTIME_P95_BORDERLINE_MS
        and (avg_ttft_ms is None or ttft_borderline)
    ):
        return {
            "label": "borderline",
            "compatible_low_latency": False,
            "reason": "Usabile ma con ritardo percepibile su avvio risposta o completamento.",
        }
    return {
        "label": "not_realtime",
        "compatible_low_latency": False,
        "reason": "Tempo prima risposta o latenza totale troppo alti per realtime naturale.",
    }


def annotate_realtime_metrics(results: list[dict[str, Any]]) -> None:
    for result in results:
        m = result["metrics"]
        profile = _evaluate_realtime_profile_with_ttft(
            avg_ms=float(m["avg_wall_ms"]),
            p95_ms=float(m["p95_wall_ms"]),
            avg_ttft_ms=(float(m["avg_ttft_ms"]) if m.get("avg_ttft_ms") is not None else None),
            p95_ttft_ms=(float(m["p95_ttft_ms"]) if m.get("p95_ttft_ms") is not None else None),
        )
        m["realtime_profile"] = profile["label"]
        m["compatible_low_latency"] = profile["compatible_low_latency"]
        m["realtime_reason"] = profile["reason"]


def build_prompt_optimization(
    model: str,
    metrics: dict[str, Any],
    base_prompt: str,
) -> dict[str, Any]:
    profile = metrics.get("realtime_profile", "not_realtime")
    tuned_lines = [base_prompt, "", "Ottimizzazione latenza per benchmark"]
    notes: list[str] = []

    if profile == "realtime_ready":
        tuned_lines.extend(
            [
                "Rispondi in una frase quando possibile.",
                "Massimo venti parole salvo richiesta esplicita.",
                "Evita dettagli opzionali non richiesti.",
            ]
        )
        notes.append("Profilo veloce: ottimizzazione leggera senza perdere qualità.")
    elif profile == "borderline":
        tuned_lines.extend(
            [
                "Rispondi sempre in una frase breve.",
                "Massimo diciotto parole.",
                "Se serve approfondire chiedi prima conferma in una domanda breve.",
                "Evita esempi o alternative multiple nella prima risposta.",
            ]
        )
        notes.append("Profilo borderline: riduzione aggressiva lunghezza per stabilizzare la latenza.")
    else:
        tuned_lines.extend(
            [
                "Modalita ultra rapida attiva.",
                "Risposta massimo dodici parole.",
                "Usa struttura fissa: esito breve poi prossima azione.",
                "Se mancano dati chiedi una sola informazione essenziale.",
                "Non fornire spiegazioni lunghe finche utente non le chiede.",
            ]
        )
        notes.append("Profilo lento: prompt molto restrittivo per ridurre token generati.")

    return {
        "model": model,
        "realtime_profile": profile,
        "compatible_low_latency": metrics.get("compatible_low_latency", False),
        "avg_wall_ms": metrics.get("avg_wall_ms"),
        "p95_wall_ms": metrics.get("p95_wall_ms"),
        "avg_ttft_ms": metrics.get("avg_ttft_ms"),
        "p95_ttft_ms": metrics.get("p95_ttft_ms"),
        "notes": notes,
        "optimized_prompt": "\n".join(tuned_lines).strip(),
    }


def write_optimized_prompts_files(
    prompts: list[dict[str, Any]],
    md_path: Path,
    json_path: Path,
) -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompts": prompts,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Prompt ottimizzati per LLM benchmark")
    lines.append("")
    for p in prompts:
        lines.append(f"## {p['model']}")
        lines.append(
            f"- Profilo realtime: `{p['realtime_profile']}` | "
            f"compatibile: `{p['compatible_low_latency']}` | "
            f"avg: `{p['avg_wall_ms']:.0f} ms` | p95: `{p['p95_wall_ms']:.0f} ms` | "
            f"avg_ttft: `{format_ms(p.get('avg_ttft_ms'))}` | "
            f"p95_ttft: `{format_ms(p.get('p95_ttft_ms'))}`"
        )
        for note in p.get("notes", []):
            lines.append(f"- {note}")
        lines.append("")
        lines.append("```text")
        lines.append(p["optimized_prompt"])
        lines.append("```")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_markdown_report(
    path: Path,
    ollama_url: str,
    judge_model: str,
    quality_repeats: int,
    questions_count: int,
    results: list[dict[str, Any]],
    ranking_quality: list[dict[str, Any]],
    ranking_quality_latency: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    lines.append("# Benchmark Ollama: Qualita Risposte Receptionist")
    lines.append("")
    lines.append(f"- Data UTC: `{now}`")
    lines.append(f"- Endpoint: `{ollama_url}`")
    lines.append(f"- Domande standard: `{questions_count}`")
    lines.append(f"- Repeats per domanda: `{quality_repeats}`")
    lines.append(f"- Judge model: `{judge_model}`")
    lines.append("")

    lines.append("## Ranking qualita")
    lines.append("")
    for i, row in enumerate(ranking_quality, start=1):
        lines.append(
            f"{i}. **{row['model']}** - quality_avg={row['quality_avg']:.1f}, "
            f"policy={row['policy_compliance_rate']:.1%}, "
            f"hallucination_risk={row['hallucination_risk_rate']:.1%}, "
            f"avg_latency={row['avg_wall_ms']:.0f} ms, "
            f"avg_ttft={format_ms(row.get('avg_ttft_ms'))}"
        )
    lines.append("")

    lines.append("## Compatibilita realtime (latenza)")
    lines.append("")
    for item in sorted(results, key=lambda x: x["metrics"]["avg_wall_ms"]):
        m = item["metrics"]
        lines.append(
            f"- **{item['model']}**: avg={m['avg_wall_ms']:.0f} ms, "
            f"p95={m['p95_wall_ms']:.0f} ms, "
            f"avg_ttft={format_ms(m.get('avg_ttft_ms'))}, "
            f"p95_ttft={format_ms(m.get('p95_ttft_ms'))}, "
            f"profilo={m.get('realtime_profile', 'n/d')}, "
            f"compatibile={m.get('compatible_low_latency', False)}. "
            f"Motivo: {m.get('realtime_reason', 'n/d')}"
        )
    lines.append("")

    lines.append("## Ranking combinato qualita e latenza")
    lines.append("")
    for i, row in enumerate(ranking_quality_latency, start=1):
        lines.append(
            f"{i}. **{row['model']}** - combined={row['combined_score']:.3f}, "
            f"quality_avg={row['quality_avg']:.1f}, "
            f"avg_latency={row['avg_wall_ms']:.0f} ms, "
            f"avg_ttft={format_ms(row.get('avg_ttft_ms'))}, "
            f"policy={row['policy_compliance_rate']:.1%}"
        )
    lines.append("")

    lines.append("## Dettaglio per modello")
    lines.append("")
    for item in results:
        m = item["metrics"]
        lines.append(f"### {item['model']}")
        lines.append(f"- Quality avg: **{m['quality_avg']:.1f}/100**")
        lines.append(f"- Quality p50: **{m['quality_p50']:.1f}**")
        lines.append(f"- Quality p95: **{m['quality_p95']:.1f}**")
        lines.append(f"- Policy compliance: **{m['policy_compliance_rate']:.1%}**")
        lines.append(f"- Hallucination risk rate: **{m['hallucination_risk_rate']:.1%}**")
        lines.append(f"- Avg latency: **{m['avg_wall_ms']:.0f} ms**")
        lines.append(f"- P50 latency: **{m['p50_wall_ms']:.0f} ms**")
        lines.append(f"- P95 latency: **{m['p95_wall_ms']:.0f} ms**")
        lines.append(f"- Avg TTFT streaming: **{format_ms(m.get('avg_ttft_ms'))}**")
        lines.append(f"- P95 TTFT streaming: **{format_ms(m.get('p95_ttft_ms'))}**")
        lines.append(
            f"- Realtime profile: **{m.get('realtime_profile', 'n/d')}** "
            f"(compatibile={m.get('compatible_low_latency', False)})"
        )
        lines.append(f"- Realtime note: **{m.get('realtime_reason', 'n/d')}**")
        if m["avg_tokens_per_sec"] is not None:
            lines.append(f"- Avg tokens/s: **{m['avg_tokens_per_sec']:.1f}**")
        else:
            lines.append("- Avg tokens/s: **n/d**")
        lines.append("")
        lines.append("#### Esempi affrontati")
        for ex in item.get("examples", []):
            q_text = ex["question"].replace("\n", " ").strip()
            a_text = ex["answer"].replace("\n", " ").strip()
            if len(a_text) > 220:
                a_text = a_text[:220] + "..."
            lines.append(
                f"- Q `{ex['question_id']}`: {q_text}\n"
                f"  R: {a_text}\n"
                f"  Score finale={ex['final_quality_score']:.1f}, "
                f"rule={ex['rule_score']:.1f}, judge={ex['judge_score']:.1f}. "
                f"Nota: {ex['note']}"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Ollama tool-calling and quality.")
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="URL Ollama API (default: http://127.0.0.1:11434)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Lista modelli da testare. Se vuoto, usa tutti i modelli disponibili.",
    )
    parser.add_argument("--repeats", type=int, default=2, help="Ripetizioni per task tool.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup per modello.")
    parser.add_argument(
        "--out-json",
        default="benchmark/ollama_toolcalling_results.json",
        help="Path output JSON benchmark tool.",
    )
    parser.add_argument(
        "--out-md",
        default="benchmark/ollama_toolcalling_report.md",
        help="Path output Markdown benchmark tool.",
    )

    parser.add_argument(
        "--quality-benchmark",
        action="store_true",
        help="Abilita benchmark qualita Q&A receptionist.",
    )
    parser.add_argument(
        "--questions-file",
        default=DEFAULT_QUESTIONS_FILE,
        help="Dataset domande qualità.",
    )
    parser.add_argument(
        "--judge-model",
        default="llama3.2:latest",
        help="Modello Ollama usato come giudice qualità.",
    )
    parser.add_argument(
        "--quality-repeats",
        type=int,
        default=1,
        help="Ripetizioni per ogni domanda qualità.",
    )
    parser.add_argument(
        "--out-quality-json",
        default="benchmark/ollama_quality_results.json",
        help="Path output JSON benchmark qualità.",
    )
    parser.add_argument(
        "--out-quality-md",
        default="benchmark/ollama_quality_report.md",
        help="Path output Markdown benchmark qualità.",
    )
    parser.add_argument(
        "--out-prompts-md",
        default="benchmark/ollama_optimized_prompts_by_model.md",
        help="Path output Markdown prompt ottimizzati per modello.",
    )
    parser.add_argument(
        "--out-prompts-json",
        default="benchmark/ollama_optimized_prompts_by_model.json",
        help="Path output JSON prompt ottimizzati per modello.",
    )
    parser.add_argument(
        "--skip-tool-benchmark",
        action="store_true",
        help="Salta benchmark tool-calling.",
    )
    parser.add_argument(
        "--verbose-questions",
        action="store_true",
        help="Mostra log domande durante benchmark qualità.",
    )
    parser.add_argument(
        "--no-verbose-questions",
        action="store_true",
        help="Disabilita log domande benchmark qualità.",
    )
    parser.add_argument(
        "--concurrency-benchmark",
        action="store_true",
        help="Abilita benchmark concorrenza per stimare la massima parallelizzazione stabile.",
    )
    parser.add_argument(
        "--concurrency-max-workers",
        type=int,
        default=max(2, min(12, (os.cpu_count() or 4) * 2)),
        help="Livello massimo di concorrenza da testare per modello.",
    )
    parser.add_argument(
        "--concurrency-rounds",
        type=int,
        default=2,
        help="Round per livello di concorrenza (richieste totali = livello * round).",
    )
    parser.add_argument(
        "--concurrency-timeout-s",
        type=int,
        default=120,
        help="Timeout in secondi per singola richiesta del benchmark concorrenza.",
    )
    parser.add_argument(
        "--concurrency-min-success-rate",
        type=float,
        default=1.0,
        help="Success rate minima per considerare stabile un livello di concorrenza.",
    )
    parser.add_argument(
        "--concurrency-degradation-factor",
        type=float,
        default=3.0,
        help="Moltiplicatore massimo ammesso su avg e p95 rispetto al baseline (livello 1).",
    )
    parser.add_argument(
        "--concurrency-realtime-avg-ms",
        type=float,
        default=REALTIME_AVG_TARGET_MS,
        help="Soglia massima avg latency per considerare il livello compatibile realtime.",
    )
    parser.add_argument(
        "--concurrency-realtime-p95-ms",
        type=float,
        default=REALTIME_P95_TARGET_MS,
        help="Soglia massima p95 latency per considerare il livello compatibile realtime.",
    )
    parser.add_argument(
        "--concurrency-stop-on-realtime-degradation",
        action="store_true",
        help="Forza early-stop quando esce dalle soglie realtime (default attivo).",
    )
    parser.add_argument(
        "--no-concurrency-stop-on-realtime-degradation",
        action="store_true",
        help="Non interrompe il test anche se la latenza non e piu realtime.",
    )
    parser.add_argument(
        "--out-concurrency-json",
        default="benchmark/ollama_concurrency_results.json",
        help="Path output JSON benchmark concorrenza.",
    )
    parser.add_argument(
        "--out-concurrency-md",
        default="benchmark/ollama_concurrency_report.md",
        help="Path output Markdown benchmark concorrenza.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verbose_questions = args.verbose_questions or not args.no_verbose_questions

    models = args.models or get_models(args.ollama_url)
    if not models:
        raise RuntimeError("Nessun modello trovato in Ollama.")

    all_payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ollama_url": args.ollama_url,
        "models": models,
    }

    if not args.skip_tool_benchmark:
        tool_results = []
        for model in models:
            print(f"[INFO] Tool benchmark modello: {model}")
            model_result = benchmark_tool_model(
                ollama_url=args.ollama_url,
                model=model,
                repeats=args.repeats,
                warmups=args.warmups,
            )
            tool_results.append(model_result)
        tool_ranking = rank_tool_models(tool_results)
        out_json = Path(args.out_json)
        out_md = Path(args.out_md)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        tool_payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ollama_url": args.ollama_url,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "tasks": [asdict(t) for t in TOOL_TASKS],
            "results": tool_results,
            "ranking": tool_ranking,
        }
        out_json.write_text(
            json.dumps(tool_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_tool_markdown_report(
            path=out_md,
            ollama_url=args.ollama_url,
            repeats=args.repeats,
            warmups=args.warmups,
            results=tool_results,
            ranking=tool_ranking,
        )
        print(f"[OK] Tool JSON: {out_json}")
        print(f"[OK] Tool Markdown: {out_md}")
        all_payload["tool_benchmark"] = tool_payload

    if args.quality_benchmark:
        questions_path = Path(args.questions_file)
        questions = load_quality_questions(questions_path)
        if not questions:
            raise RuntimeError("Dataset domande qualità vuoto.")
        system_prompt = get_receptionist_system_prompt()

        quality_results = []
        for model in models:
            print(f"[INFO] Quality benchmark modello: {model}")
            model_result = benchmark_quality_model(
                ollama_url=args.ollama_url,
                model=model,
                questions=questions,
                quality_repeats=args.quality_repeats,
                warmups=args.warmups,
                judge_model=args.judge_model,
                system_prompt=system_prompt,
                verbose_questions=verbose_questions,
            )
            quality_results.append(model_result)

        ranking_quality = rank_quality_models(quality_results)
        ranking_quality_latency = rank_quality_latency(quality_results)
        annotate_realtime_metrics(quality_results)
        optimized_prompts = [
            build_prompt_optimization(
                model=result["model"],
                metrics=result["metrics"],
                base_prompt=system_prompt,
            )
            for result in quality_results
        ]

        out_quality_json = Path(args.out_quality_json)
        out_quality_md = Path(args.out_quality_md)
        out_prompts_md = Path(args.out_prompts_md)
        out_prompts_json = Path(args.out_prompts_json)
        out_quality_json.parent.mkdir(parents=True, exist_ok=True)
        out_quality_md.parent.mkdir(parents=True, exist_ok=True)

        quality_payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ollama_url": args.ollama_url,
            "judge_model": args.judge_model,
            "quality_repeats": args.quality_repeats,
            "questions_file": str(questions_path),
            "questions_count": len(questions),
            "system_prompt": system_prompt,
            "questions": [asdict(q) for q in questions],
            "results": quality_results,
            "ranking_quality": ranking_quality,
            "ranking_quality_latency": ranking_quality_latency,
            "optimized_prompts_by_model": optimized_prompts,
        }

        out_quality_json.write_text(
            json.dumps(quality_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_quality_markdown_report(
            path=out_quality_md,
            ollama_url=args.ollama_url,
            judge_model=args.judge_model,
            quality_repeats=args.quality_repeats,
            questions_count=len(questions),
            results=quality_results,
            ranking_quality=ranking_quality,
            ranking_quality_latency=ranking_quality_latency,
        )
        write_optimized_prompts_files(
            prompts=optimized_prompts,
            md_path=out_prompts_md,
            json_path=out_prompts_json,
        )
        print(f"[OK] Quality JSON: {out_quality_json}")
        print(f"[OK] Quality Markdown: {out_quality_md}")
        print(f"[OK] Optimized prompts MD: {out_prompts_md}")
        print(f"[OK] Optimized prompts JSON: {out_prompts_json}")
        all_payload["quality_benchmark"] = quality_payload

    if args.concurrency_benchmark:
        system_prompt = get_receptionist_system_prompt()
        concurrency_results = []
        safe_max_workers = max(1, args.concurrency_max_workers)
        safe_rounds = max(1, args.concurrency_rounds)
        safe_timeout_s = max(10, args.concurrency_timeout_s)
        safe_min_success = max(0.1, min(1.0, args.concurrency_min_success_rate))
        safe_degradation = max(1.1, args.concurrency_degradation_factor)
        safe_realtime_avg = max(50.0, args.concurrency_realtime_avg_ms)
        safe_realtime_p95 = max(100.0, args.concurrency_realtime_p95_ms)
        safe_stop_realtime = (
            args.concurrency_stop_on_realtime_degradation
            or not args.no_concurrency_stop_on_realtime_degradation
        )

        for model in models:
            print(f"[INFO] Concurrency benchmark modello: {model}")
            model_result = benchmark_concurrency_model(
                ollama_url=args.ollama_url,
                model=model,
                system_prompt=system_prompt,
                warmups=args.warmups,
                max_workers=safe_max_workers,
                rounds_per_level=safe_rounds,
                timeout_s=safe_timeout_s,
                min_success_rate=safe_min_success,
                degradation_factor=safe_degradation,
                realtime_avg_threshold_ms=safe_realtime_avg,
                realtime_p95_threshold_ms=safe_realtime_p95,
                stop_on_realtime_degradation=safe_stop_realtime,
            )
            concurrency_results.append(model_result)

        concurrency_ranking = rank_concurrency_models(concurrency_results)
        out_concurrency_json = Path(args.out_concurrency_json)
        out_concurrency_md = Path(args.out_concurrency_md)
        out_concurrency_json.parent.mkdir(parents=True, exist_ok=True)
        out_concurrency_md.parent.mkdir(parents=True, exist_ok=True)
        concurrency_payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ollama_url": args.ollama_url,
            "max_workers": safe_max_workers,
            "rounds_per_level": safe_rounds,
            "timeout_s": safe_timeout_s,
            "min_success_rate": safe_min_success,
            "degradation_factor": safe_degradation,
            "realtime_avg_threshold_ms": safe_realtime_avg,
            "realtime_p95_threshold_ms": safe_realtime_p95,
            "stop_on_realtime_degradation": safe_stop_realtime,
            "results": concurrency_results,
            "ranking": concurrency_ranking,
        }
        out_concurrency_json.write_text(
            json.dumps(concurrency_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_concurrency_markdown_report(
            path=out_concurrency_md,
            ollama_url=args.ollama_url,
            rounds_per_level=safe_rounds,
            max_workers=safe_max_workers,
            min_success_rate=safe_min_success,
            degradation_factor=safe_degradation,
            realtime_avg_threshold_ms=safe_realtime_avg,
            realtime_p95_threshold_ms=safe_realtime_p95,
            stop_on_realtime_degradation=safe_stop_realtime,
            results=concurrency_results,
            ranking=concurrency_ranking,
        )
        print(f"[OK] Concurrency JSON: {out_concurrency_json}")
        print(f"[OK] Concurrency Markdown: {out_concurrency_md}")
        all_payload["concurrency_benchmark"] = concurrency_payload

    if not args.skip_tool_benchmark and not args.quality_benchmark:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
