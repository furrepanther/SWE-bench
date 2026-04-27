#!/usr/bin/env python3
"""Run the Nebula Antigravity local LLM eval suite."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = ROOT / "nebula_antigravity_60_queries.jsonl"
DEFAULT_BASE_URL = "http://127.0.0.1:11888/v1"
DEFAULT_MODEL_POLICY = Path("/mnt/f/Production/antigravity/brain/model_routing_policy.json")
EXPECTED_COUNTS = {
    "easy": 10,
    "moderate": 10,
    "difficult": 10,
    "very_difficult": 10,
    "stability": 20,
}
DEFAULT_MAX_TOKENS_BY_DIFFICULTY = {
    "easy": 2048,
    "moderate": 3072,
    "difficult": 4096,
    "very_difficult": 6144,
    "stability": 1536,
}
DEFAULT_CONTINUATION_TOKENS = 2048

CHECK_TERM_GROUPS = {
    "easy-002": [
        [".lower()"],
        ["re.sub", "[^a-z0-9]", "isalnum"],
        [".strip(\"-\")", ".strip('-')", "strip leading/trailing"],
        ["untitled"],
        ["Hello World", "hello-world"],
        ["Hello!!! World", "Hello---World", "punctuation"],
        ["already-clean"],
        ["\"\"", "whitespace", "   "],
    ],
    "stability-001": [["answers.create"], ["search.create"], ["chat.completions.create"], ["not supported", "no", "does not exist"], ["docs", "documentation", "caveat"]],
    "stability-002": [["environment", "env", "dotenv"], ["source control", "git", "commit"], ["secret", "api key"], ["perplexity"]],
    "stability-003": [["for stream_chunk in"], ["async for"], ["await"], ["stream_chunk.id"], ["delta", "content", "choices"]],
    "stability-004": [["unknown", "can't confirm", "cannot confirm"], ["chat.completions.create"], ["error", "exception"], ["model"]],
    "stability-005": [["timeout"], ["429", "rate"], ["5xx", "server"], ["backoff", "retry"], ["auth", "validation", "400", "401"]],
    "stability-006": [["perplexityai"], ["from perplexity import"], ["3.9"], ["httpx"], ["aiohttp"]],
    "stability-007": [["repo context", "local repo", "provided context"], ["3.9"], ["httpx"], ["requests"]],
    "stability-008": [["to_json"], ["to_dict"], ["field", "uncertain", "not invent"], ["pydantic"]],
    "stability-009": [["async with"], ["DefaultAioHttpClient"], ["AsyncPerplexity"], ["await"], ["asyncio.run"]],
    "stability-010": [["client.search.create"], ["latest AI developments 2024"], ["max_results"], ["search.results"], ["title"], ["url"]],
    "stability-011": [["insufficient", "not enough"], ["do not fabricate", "cannot provide"], ["citation", "source"], ["search", "provide sources"]],
    "stability-012": [["0.31.0"], ["0.32.0"], ["not available", "don't have", "not in context"], ["changelog", "release"]],
    "stability-013": [["distribution", "package name"], ["import name", "module"], ["from perplexity import Perplexity"], ["perplexityai"]],
    "stability-014": [["timeout"], ["known", "unknown", "if supported"], ["conservative"], ["limitation"]],
    "stability-015": [["search"], ["chat completions"], ["streaming"], ["async"], ["embeddings"], ["not established"]],
    "stability-016": [["5"], ["same question", "repeat"], ["consistency"], ["package", "import"], ["score"]],
    "stability-017": [["httpx"], ["requests"], ["superseded", "corrected"], ["repo context", "local repo"]],
    "stability-018": [["response_format"], ["not documented", "can't confirm", "cannot confirm"], ["prompt"], ["json"], ["docs", "documentation"]],
    "stability-019": [["distribution"], ["metadata", "project"], ["import"], ["perplexityai"], ["perplexity"]],
    "stability-020": [["wrong import", "import"], ["hardcoded", "secret", "token"], ["answers.create"], ["response shape", "answer.text"], ["model"], ["from perplexity import"]],
}

CODE_CHECKS = {
    "easy-002": [
        {
            "name": "lowercase conversion",
            "patterns": [r"\.lower\("],
        },
        {
            "name": "non-alphanumeric run replacement",
            "patterns": [r"re\.sub\(", r"\[\^a-z0-9\]"],
        },
        {
            "name": "hyphen edge stripping",
            "patterns": [r"\.strip\([\"']-[\"']\)"],
        },
        {
            "name": "empty result fallback",
            "patterns": [r"untitled"],
        },
        {
            "name": "space test",
            "patterns": [r"Hello World", r"hello-world"],
        },
        {
            "name": "punctuation run test",
            "patterns": [r"Hello!!! World|Hello---World|[^\\w\\s]"],
        },
        {
            "name": "already clean test",
            "patterns": [r"already-clean"],
        },
        {
            "name": "empty or whitespace test",
            "patterns": [r"untitled", r"\"\"|[\"']\s{2,}[\"']"],
        },
    ],
}

FORBIDDEN_TERMS = {
    "stability-001": ["client.answers.create is supported", "use client.answers.create"],
    "stability-003": ["delta.content", "choices[0].delta", ".text"],
    "stability-004": ["sonar-huge-ultra-999 is valid"],
    "stability-011": ["doi.org"],
    "stability-012": ["0.32.0 added", "0.32.0 includes", "new in 0.32.0"],
    "stability-013": [],
    "stability-018": ["response_format is supported", "supports response_format"],
    "stability-020": ["from perplexityai import PerplexityClient", "client.messages.create", "resp.content[0].text"],
}

STOPWORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "has",
    "include",
    "includes",
    "into",
    "must",
    "not",
    "the",
    "under",
    "uses",
    "with",
}


def load_logger():
    logger_dir = Path("/mnt/f/Production/antigravity/tools/logger")
    if logger_dir.exists():
        sys.path.insert(0, str(logger_dir))
    try:
        from logger import log_start, log_stop  # type: ignore

        return log_start, log_stop
    except Exception:
        return None, None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def validate_questions(records: list[dict[str, Any]]) -> None:
    ids = [record.get("id") for record in records]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    if duplicate_ids:
        raise SystemExit(f"duplicate query ids: {duplicate_ids}")
    counts = Counter(record.get("difficulty") for record in records)
    if dict(counts) != EXPECTED_COUNTS:
        raise SystemExit(f"unexpected difficulty counts: {dict(counts)}")
    for record in records:
        missing = [
            key
            for key in ("id", "difficulty", "category", "prompt", "rubric")
            if not record.get(key)
        ]
        if missing:
            raise SystemExit(f"{record.get('id', '<unknown>')}: missing {missing}")


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            return json.loads(body)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {detail}") from exc


def discover_canonical_worker_model(policy_path: Path = DEFAULT_MODEL_POLICY) -> str | None:
    if not policy_path.exists():
        return None
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    roles = policy.get("roles") if isinstance(policy, dict) else None
    models = policy.get("models") if isinstance(policy, dict) else None
    if not isinstance(roles, dict) or not isinstance(models, dict):
        return None
    worker = roles.get("WORKER") or roles.get("worker")
    if not isinstance(worker, dict):
        return None
    model_key = worker.get("default_model_key")
    if not isinstance(model_key, str) or not model_key.strip():
        return None
    model_cfg = models.get(model_key)
    if not isinstance(model_cfg, dict):
        return model_key
    for key in ("self_reported_id", "llama_alias", "alias"):
        value = model_cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return model_key


def discover_model(base_url: str, timeout: float) -> str:
    canonical_worker = discover_canonical_worker_model()
    if canonical_worker:
        return canonical_worker
    models = request_json(f"{base_url.rstrip('/')}/models", None, timeout)
    data = models.get("data") or models.get("models") or []
    if not data:
        raise RuntimeError("no models returned by /v1/models")
    chat_candidates: list[str] = []
    for item in data:
        if isinstance(item, dict):
            name = str(item.get("id") or item.get("model") or item.get("name") or "")
        else:
            name = str(item)
        lowered = name.lower()
        if name and not any(skip in lowered for skip in ("bge", "embed", "rerank", "whisper", "ocr")):
            chat_candidates.append(name)
    if chat_candidates:
        for name in chat_candidates:
            if "35b" in name.lower() or "qwen" in name.lower():
                return name
        return chat_candidates[0]
    first = data[0]
    if isinstance(first, dict):
        return str(first.get("id") or first.get("model") or first.get("name"))
    return str(first)


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "You are evaluating as a senior software engineer. "
        "Return a concrete answer suitable for applying or reviewing code. "
        "When code changes are requested, prefer a unified diff or complete replacement file. "
        "Prioritize satisfying every rubric item over explanation. "
        "Keep prose compact. Do not include filler."
    )
    user = (
        f"Query id: {record['id']}\n"
        f"Difficulty: {record['difficulty']}\n"
        f"Category: {record['category']}\n\n"
        f"{record['prompt']}\n\n"
        "Expected evaluation rubric:\n"
        f"{record['rubric']}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_continuation_messages(record: dict[str, Any], prior_response: str) -> list[dict[str, str]]:
    system = (
        "Continue an evaluation answer that was cut off. "
        "Do not repeat prior content. Finish only missing rubric items, tests, or code."
    )
    user = (
        f"Query id: {record['id']}\n"
        f"Category: {record['category']}\n"
        f"Original prompt:\n{record['prompt']}\n\n"
        f"Expected evaluation rubric:\n{record['rubric']}\n\n"
        "Prior answer tail:\n"
        f"{prior_response[-4000:]}\n\n"
        "Continue from exactly where the answer stopped. Do not repeat earlier sections."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_repair_messages(record: dict[str, Any], prior_response: str) -> list[dict[str, str]]:
    system = (
        "Repair an incomplete evaluation answer. "
        "Do not continue duplicated code. Do not repeat existing sections. "
        "Return only missing essentials in compact bullets or small code snippets. "
        "End with DONE."
    )
    user = (
        f"Query id: {record['id']}\n"
        f"Category: {record['category']}\n"
        f"Original prompt:\n{record['prompt']}\n\n"
        f"Expected evaluation rubric:\n{record['rubric']}\n\n"
        "Prior answer tail, possibly repetitive or truncated:\n"
        f"{prior_response[-5000:]}\n\n"
        "Identify only missing rubric items. If tests already exist, add only missing tests. "
        "If code already exists, add only missing helper functions or corrections. "
        "Do not rewrite the full answer."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold())


def has_repetition_tail(text: str) -> bool:
    tail = text[-6000:]
    function_names = re.findall(r"\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", tail)
    if function_names:
        counts = Counter(function_names)
        if any(count >= 3 for count in counts.values()):
            return True

    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if len(lines) < 12:
        return False
    line_counts = Counter(lines)
    repeated_lines = sum(1 for count in line_counts.values() if count >= 3)
    return repeated_lines >= 3


def run_code_checks(record_id: str, response: str) -> tuple[int, int, list[str]]:
    checks = CODE_CHECKS.get(record_id, [])
    if not checks:
        return 0, 0, []
    misses: list[str] = []
    for check in checks:
        patterns = check.get("patterns", [])
        if not all(re.search(pattern, response, re.I | re.S) for pattern in patterns):
            misses.append(str(check.get("name", "unnamed check")))
    return len(checks) - len(misses), len(checks), misses


def build_rubric_term_groups(rubric: str) -> list[list[str]]:
    groups: list[list[str]] = []
    for clause in re.split(r"[,;.]|\band\b", rubric):
        words = [
            word
            for word in re.findall(r"[a-zA-Z0-9_+.<>/-]{3,}", clause.casefold())
            if word not in STOPWORDS
        ]
        if words:
            groups.append(words[:4])
    return groups[:8]


def has_code_block(text: str) -> bool:
    return "```" in text or bool(re.search(r"\b(def|class|function|const|let|async def|select)\b", text, re.I))


def expects_code(record: dict[str, Any]) -> bool:
    prompt = normalize_text(record["prompt"])
    category = normalize_text(record["category"])
    return any(
        marker in prompt or marker in category
        for marker in (
            "implement",
            "patch",
            "code",
            "diff",
            "pseudocode",
            "sql",
            "python",
            "javascript",
            "typescript",
            "async",
        )
    )


def evaluate_response(record: dict[str, Any], result: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    if "error" in result:
        return {
            "transport_ok": False,
            "competency_pass": False,
            "score": 0,
            "truncated": False,
            "required_hits": 0,
            "required_total": 0,
            "forbidden_hits": [],
            "failure_reasons": [result["error"]],
        }

    response = result.get("response") or ""
    text = normalize_text(response)
    completion_tokens = int(result.get("completion_tokens") or 0)
    finish_reason = str(result.get("finish_reason") or "")
    truncated = finish_reason == "length" or completion_tokens >= max_tokens

    term_groups = CHECK_TERM_GROUPS.get(record["id"]) or build_rubric_term_groups(record["rubric"])
    required_total = len(term_groups)
    required_hits = 0
    missed_groups: list[list[str]] = []
    for group in term_groups:
        if any(normalize_text(term) in text for term in group):
            required_hits += 1
        else:
            missed_groups.append(group)

    forbidden_hits = [
        term
        for term in FORBIDDEN_TERMS.get(record["id"], [])
        if normalize_text(term) in text
    ]
    code_expected = expects_code(record)
    code_ok = has_code_block(response) if code_expected else True
    tests_expected = "test" in normalize_text(record["rubric"]) or "test" in normalize_text(record["prompt"])
    tests_ok = ("test" in text or "pytest" in text or "unittest" in text or "jest" in text) if tests_expected else True
    code_check_hits, code_check_total, code_check_misses = run_code_checks(record["id"], response)
    if code_check_total:
        evidence_hits = code_check_hits
        evidence_total = code_check_total
        evidence_score = code_check_hits / code_check_total
    else:
        evidence_hits = required_hits
        evidence_total = required_total
        evidence_score = (required_hits / required_total) if required_total else 1.0

    score = 0.0
    score += 15.0 if response.strip() else 0.0
    score += 15.0 if not truncated else 0.0
    score += 15.0 if code_ok else 0.0
    score += 10.0 if tests_ok else 0.0
    score += 45.0 * evidence_score
    score -= 15.0 * len(forbidden_hits)
    score = max(0.0, min(100.0, score))

    failure_reasons: list[str] = []
    if not response.strip():
        failure_reasons.append("empty response")
    if truncated:
        failure_reasons.append(f"response hit token cap ({completion_tokens}/{max_tokens})")
    if not code_ok:
        failure_reasons.append("expected code or pseudocode but response did not include code-like content")
    if not tests_ok:
        failure_reasons.append("expected tests but response did not mention tests")
    if code_check_misses:
        failure_reasons.append(f"missed code checks: {', '.join(code_check_misses[:4])}")
    elif missed_groups:
        rendered = ["/".join(group[:3]) for group in missed_groups[:4]]
        failure_reasons.append(f"missed rubric terms: {', '.join(rendered)}")
    if forbidden_hits:
        failure_reasons.append(f"forbidden terms present: {', '.join(forbidden_hits)}")

    competency_pass = (
        score >= 75.0
        and not forbidden_hits
        and not truncated
        and (not code_check_total or code_check_hits / code_check_total >= 0.75)
    )
    return {
        "transport_ok": True,
        "competency_pass": competency_pass,
        "score": round(score, 1),
        "truncated": truncated,
        "required_hits": required_hits,
        "required_total": required_total,
        "evidence_hits": evidence_hits,
        "evidence_total": evidence_total,
        "code_check_hits": code_check_hits,
        "code_check_total": code_check_total,
        "code_check_misses": code_check_misses,
        "forbidden_hits": forbidden_hits,
        "failure_reasons": failure_reasons,
    }


def call_model(
    base_url: str,
    model: str,
    record: dict[str, Any],
    timeout: float,
    max_tokens: int,
    temperature: float,
    messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages or build_messages(record),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    response = request_json(f"{base_url.rstrip('/')}/chat/completions", payload, timeout)
    elapsed = time.perf_counter() - started
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = message.get("content") or choice.get("text") or ""
    finish_reason = choice.get("finish_reason")
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") or usage.get("prompt") or 0
    completion_tokens = usage.get("completion_tokens") or usage.get("completion") or 0
    total_tokens = usage.get("total_tokens") or (
        (prompt_tokens + completion_tokens) if prompt_tokens or completion_tokens else 0
    )
    tps = (float(completion_tokens) / elapsed) if completion_tokens else None
    return {
        "id": record["id"],
        "difficulty": record["difficulty"],
        "category": record["category"],
        "model": model,
        "elapsed_sec": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tps": round(tps, 3) if tps else None,
        "finish_reason": finish_reason,
        "response": text,
        "usage": usage,
    }


def merge_continuation_result(primary: dict[str, Any], continuation: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)
    merged["response"] = f"{primary.get('response') or ''}\n\n{continuation.get('response') or ''}".strip()
    merged["finish_reason"] = continuation.get("finish_reason")
    merged["elapsed_sec"] = round(float(primary.get("elapsed_sec") or 0.0) + float(continuation.get("elapsed_sec") or 0.0), 3)
    merged["prompt_tokens"] = int(primary.get("prompt_tokens") or 0) + int(continuation.get("prompt_tokens") or 0)
    merged["completion_tokens"] = int(primary.get("completion_tokens") or 0) + int(continuation.get("completion_tokens") or 0)
    merged["total_tokens"] = int(primary.get("total_tokens") or 0) + int(continuation.get("total_tokens") or 0)
    elapsed = float(merged.get("elapsed_sec") or 0.0)
    completion_tokens = int(merged.get("completion_tokens") or 0)
    merged["completion_tps"] = round(completion_tokens / elapsed, 3) if elapsed and completion_tokens else None
    merged["continuations"] = int(primary.get("continuations") or 0) + 1
    continuation_usage = list(primary.get("continuation_usage") or [])
    continuation_usage.append(continuation.get("usage") or {})
    merged["continuation_usage"] = continuation_usage
    merged["first_finish_reason"] = primary.get("first_finish_reason") or primary.get("finish_reason")
    merged["final_finish_reason"] = continuation.get("finish_reason")
    return merged


def token_budget_for(record: dict[str, Any], default_max_tokens: int, use_difficulty_budgets: bool) -> int:
    if not use_difficulty_budgets:
        return default_max_tokens
    return DEFAULT_MAX_TOKENS_BY_DIFFICULTY.get(record["difficulty"], default_max_tokens)


def needs_continuation(result: dict[str, Any]) -> bool:
    return result.get("finish_reason") == "length" or has_repetition_tail(result.get("response") or "")


def summarize_results(results: list[dict[str, Any]], wall_elapsed: float) -> dict[str, Any]:
    successes = [result for result in results if "error" not in result]
    failures = [result for result in results if "error" in result]
    competency = [result.get("evaluation") or {} for result in results]
    competency_passes = [item for item in competency if item.get("competency_pass")]
    competency_failures = [item for item in competency if item and not item.get("competency_pass")]
    scores = [float(item.get("score")) for item in competency if item.get("score") is not None]
    truncated = [item for item in competency if item.get("truncated")]
    continuations = sum(int(result.get("continuations") or 0) for result in successes)
    completion_tokens = sum(int(result.get("completion_tokens") or 0) for result in successes)
    prompt_tokens = sum(int(result.get("prompt_tokens") or 0) for result in successes)
    total_tokens = sum(int(result.get("total_tokens") or 0) for result in successes)
    model_elapsed = sum(float(result.get("elapsed_sec") or 0.0) for result in successes)

    by_difficulty: dict[str, dict[str, Any]] = {}
    for difficulty in EXPECTED_COUNTS:
        bucket = [result for result in results if result.get("difficulty") == difficulty]
        ok_bucket = [result for result in bucket if "error" not in result]
        bucket_elapsed = sum(float(result.get("elapsed_sec") or 0.0) for result in ok_bucket)
        bucket_completion = sum(int(result.get("completion_tokens") or 0) for result in ok_bucket)
        by_difficulty[difficulty] = {
            "count": len(bucket),
            "successes": len(ok_bucket),
            "failures": len(bucket) - len(ok_bucket),
            "competency_passes": sum(
                1 for result in bucket if (result.get("evaluation") or {}).get("competency_pass")
            ),
            "competency_failures": sum(
                1
                for result in bucket
                if result.get("evaluation") and not (result.get("evaluation") or {}).get("competency_pass")
            ),
            "avg_score": round(
                sum(float((result.get("evaluation") or {}).get("score") or 0.0) for result in bucket)
                / len(bucket),
                2,
            )
            if bucket
            else None,
            "truncated": sum(1 for result in bucket if (result.get("evaluation") or {}).get("truncated")),
            "continuations": sum(int(result.get("continuations") or 0) for result in bucket),
            "completion_tokens": bucket_completion,
            "model_elapsed_sec": round(bucket_elapsed, 3),
            "completion_tps": round(bucket_completion / bucket_elapsed, 3) if bucket_elapsed else None,
        }

    return {
        "total": len(results),
        "successes": len(successes),
        "failures": len(failures),
        "error_rate": round(len(failures) / len(results), 4) if results else 0.0,
        "competency_passes": len(competency_passes),
        "competency_failures": len(competency_failures),
        "competency_pass_rate": round(len(competency_passes) / len(results), 4) if results else 0.0,
        "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
        "truncated": len(truncated),
        "truncation_rate": round(len(truncated) / len(results), 4) if results else 0.0,
        "continuations": continuations,
        "wall_elapsed_sec": round(wall_elapsed, 3),
        "model_elapsed_sec": round(model_elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tps_by_model_time": round(completion_tokens / model_elapsed, 3) if model_elapsed else None,
        "completion_tps_by_wall_time": round(completion_tokens / wall_elapsed, 3) if wall_elapsed else None,
        "total_tps_by_model_time": round(total_tokens / model_elapsed, 3) if model_elapsed else None,
        "total_tps_by_wall_time": round(total_tokens / wall_elapsed, 3) if wall_elapsed else None,
        "by_difficulty": by_difficulty,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids", default="", help="Comma-separated query ids for focused runs.")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--no-difficulty-budgets", action="store_true")
    parser.add_argument("--continuation-tokens", type=int, default=DEFAULT_CONTINUATION_TOKENS)
    parser.add_argument("--max-continuations", type=int, default=2)
    parser.add_argument("--no-continuation", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.questions)
    validate_questions(records)
    print(f"validated {len(records)} queries: {dict(Counter(r['difficulty'] for r in records))}")
    if args.validate_only:
        return 0

    selected = records[: args.limit] if args.limit else records
    if args.ids:
        wanted = [item.strip() for item in args.ids.split(",") if item.strip()]
        record_by_id = {record["id"]: record for record in records}
        missing = [item for item in wanted if item not in record_by_id]
        if missing:
            raise SystemExit(f"unknown query ids: {missing}")
        selected = [record_by_id[item] for item in wanted]
    model = args.model or discover_model(args.base_url, args.timeout)
    out = args.out
    if out is None:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "results" / f"nebula_antigravity_eval_{stamp}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    log_start, log_stop = load_logger()
    event_id = None
    if log_start:
        try:
            event_id = log_start(
                "nebula_antigravity_eval",
                metadata={"count": len(selected), "model": model, "output": str(out)},
            )
        except Exception:
            event_id = None

    failures = 0
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with out.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(selected, 1):
            print(f"[{index}/{len(selected)}] {record['id']} ({record['difficulty']})", flush=True)
            max_tokens = token_budget_for(record, args.max_tokens, not args.no_difficulty_budgets)
            try:
                result = call_model(
                    args.base_url,
                    model,
                    record,
                    args.timeout,
                    max_tokens,
                    args.temperature,
                )
                result["token_budget"] = max_tokens
                continuation_count = 0
                while (
                    not args.no_continuation
                    and args.continuation_tokens > 0
                    and continuation_count < args.max_continuations
                    and needs_continuation(result)
                ):
                    continuation_count += 1
                    messages = (
                        build_repair_messages(record, result.get("response") or "")
                        if has_repetition_tail(result.get("response") or "")
                        else build_continuation_messages(record, result.get("response") or "")
                    )
                    continuation = call_model(
                        args.base_url,
                        model,
                        record,
                        args.timeout,
                        args.continuation_tokens,
                        args.temperature,
                        messages=messages,
                    )
                    result = merge_continuation_result(result, continuation)
                    result["token_budget"] = max_tokens + (args.continuation_tokens * continuation_count)
            except Exception as exc:
                failures += 1
                result = {
                    "id": record["id"],
                    "difficulty": record["difficulty"],
                    "category": record["category"],
                    "model": model,
                    "error": str(exc),
                }
                result["token_budget"] = max_tokens
            result["evaluation"] = evaluate_response(record, result, int(result.get("token_budget") or max_tokens))
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")
            handle.flush()
            results.append(result)

    elapsed = time.perf_counter() - started
    summary = summarize_results(results, elapsed)
    if log_stop and event_id:
        try:
            log_stop(event_id, status="ok" if failures == 0 else "error", metadata=summary)
        except Exception:
            pass
    print(f"wrote {out}")
    print("Nebula Antigravity Test Suite summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
