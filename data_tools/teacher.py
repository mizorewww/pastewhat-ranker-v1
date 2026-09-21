"""Audited Kimi Code requests with no credentials in requests, logs, or Git.

The model identifier is rolling. Every successful response's actual model string,
usage, request hash and raw response hash are retained; this is not a pinned
teacher weight revision. Audit storage must be split-specific.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from data_tools.rate_limit import AccountCoordinator, AccountPaused, retry_after_seconds


DEFAULT_ENDPOINT = "https://api.kimi.com/coding/v1/chat/completions"
DEFAULT_MODEL = "kimi-for-coding"
CREDENTIAL_FILE = Path.home() / "Library/Application Support/PasteWhat/credentials/kimi.key"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n")
    temporary.replace(path)


class TeacherError(RuntimeError):
    pass


@dataclass(frozen=True)
class TeacherResult:
    parsed: Any
    audit_id: str
    model: str
    usage: dict[str, Any]
    response_sha256: str
    cache_hit: bool


class TeacherClient:
    """Thread-safe HTTP client; clients must never share cross-split audit dirs."""

    def __init__(
        self,
        audit_dir: str | Path,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        min_interval: float = 0.5,
        timeout: float = 480,
        max_attempts: int = 6,
    ) -> None:
        if endpoint not in (
            DEFAULT_ENDPOINT,
            "https://api.kimi.ai/coding/v1/chat/completions",
        ):
            raise ValueError("Only documented Kimi Code endpoints are allowed")
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint
        self.model = model
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.coordinator = AccountCoordinator(min_interval=min_interval)
        token = os.environ.get("KIMI_API_KEY", "").strip()
        if not token and CREDENTIAL_FILE.is_file():
            if CREDENTIAL_FILE.stat().st_mode & 0o077:
                raise TeacherError("Credential file must have mode 0600")
            token = CREDENTIAL_FILE.read_text().strip()
        if not token:
            raise TeacherError("KIMI_API_KEY or the restricted local credential file is required")
        self._token = token

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 16384,
        temperature: float = 1.0,
        thinking: str | None = None,
        reasoning_effort: str | None = None,
        response_format: str | None = None,
        phase: str,
        request_id: str,
    ) -> TeacherResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if response_format is not None:
            if response_format != "json_object":
                raise ValueError("The optional compatibility probe supports only json_object")
            body["response_format"] = {"type": response_format}
        if thinking is not None:
            if thinking not in ("enabled", "disabled"):
                raise ValueError("thinking must be enabled or disabled")
            # Documented in moonshotai/kimi-code's own Kimi provider; this is
            # an extra OpenAI-compatible field, not a fabricated client identity.
            body["thinking"] = {"type": thinking}
        if reasoning_effort is not None and reasoning_effort not in ("low", "high", "max"):
            raise ValueError("reasoning_effort must be low, high, or max")
        if thinking == "disabled":
            if reasoning_effort is not None:
                raise ValueError("Disabled thinking cannot include reasoning_effort")
        else:
            # K2.8 Preview defaults to max when omitted. Set the documented
            # recommended high explicitly; production waits for a paired probe.
            default_effort = "high"
            marker = self.coordinator.directory / "production-ready.json"
            if marker.is_file():
                policy = json.loads(marker.read_text())
                if policy.get("ready") is True and policy.get("reasoning_effort") in ("high", "max"):
                    default_effort = policy["reasoning_effort"]
            body["reasoning_effort"] = reasoning_effort or default_effort
        request_bytes = canonical_bytes(body)
        audit_id = sha256(canonical_bytes({"endpoint": self.endpoint, "body": body}))
        path = self.audit_dir / f"{audit_id}.json"
        previous = None
        if path.is_file():
            audit = json.loads(path.read_text())
            if audit.get("status") == "success":
                return self._result(audit, cache_hit=True)
            previous = audit
        attempts = list(previous.get("attempts", [])) if previous else []
        if previous and previous.get("status") == "request_started":
            attempts.append({"error_type": "previous_started_attempt_has_no_observed_completion", "started_at": previous.get("attempt_started_at"), "pid": previous.get("pid"), "usage_known": False})
        started = utc_now()
        for attempt in range(self.max_attempts):
            try:
                lease = self.coordinator.acquire(self.timeout)
            except AccountPaused as exc:
                raise TeacherError(str(exc)) from None
            request = Request(
                self.endpoint,
                data=request_bytes,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "PasteWhat-Ranker/0.1 (synthetic-data-research; truthful-client)",
                },
                method="POST",
            )
            before = time.monotonic()
            # Persist the exact sanitized request before opening the connection.
            # If a worker is interrupted, its unknown response/usage remains
            # visible instead of disappearing from the production history.
            atomic_json(path, {"status": "request_started", "audit_id": audit_id, "phase": phase, "request_id": request_id, "started_at": started, "attempt_started_at": utc_now(), "pid": os.getpid(), "lease_id": lease, "endpoint": self.endpoint, "request": body, "request_sha256": sha256(request_bytes), "attempts": attempts})
            try:
                try:
                    with urlopen(request, timeout=self.timeout) as response:
                        raw = response.read()
                        response_headers = dict(response.headers)
                finally:
                    self.coordinator.release(lease)
                decoded = json.loads(raw)
                audit = {
                    "status": "success",
                    "audit_id": audit_id,
                    "phase": phase,
                    "request_id": request_id,
                    "started_at": started,
                    "completed_at": utc_now(),
                    "pid": os.getpid(),
                    "endpoint": self.endpoint,
                    "request": body,
                    "request_sha256": sha256(request_bytes),
                    "response": decoded,
                    "response_sha256": sha256(raw),
                    "response_headers": {k: v for k, v in response_headers.items() if k.lower() in ("x-request-id", "request-id", "date", "content-type")},
                    "elapsed_seconds": time.monotonic() - before,
                    "attempts": attempts,
                    "teacher_model_is_rolling": True,
                }
                # Validate before claiming successful usable JSON, but retain the
                # actual HTTP response even when the model breaks the contract.
                try:
                    result = self._result(audit, cache_hit=False)
                except TeacherError as exc:
                    audit["status"] = "invalid_response"
                    audit["validation_error"] = str(exc)
                    atomic_json(path, audit)
                    raise
                atomic_json(path, audit)
                return result
            except HTTPError as exc:
                raw_error = exc.read()
                # Only response hash and status are persisted, never headers or
                # an exception string that could contain credential material.
                safe_error = raw_error.decode(errors="replace").replace(self._token, "[REDACTED]")[:1000]
                retryable = exc.code in (408, 429, 500, 502, 503, 504)
                delay = min(60.0, 2 ** (attempt + 1) + random.random())
                server_delay = retry_after_seconds(exc.headers)
                delay = max(delay, server_delay or 0)
                classification = self.coordinator.record_http_failure(exc.code, safe_error, exc.headers, delay)
                detail = {"attempt": attempt + 1, "http_status": exc.code, "classification": classification, "response_sha256": sha256(raw_error), "message": safe_error, "elapsed_seconds": time.monotonic() - before, "retry_after_seconds": server_delay}
                attempts.append(detail)
                # A later acquire may wait for quota/operator maintenance. Save
                # this observed response before that wait, rather than leaving
                # the completed attempt falsely marked as still in flight.
                atomic_json(path, {"status": "retry_wait", "audit_id": audit_id, "phase": phase, "request_id": request_id, "started_at": started, "last_attempt_completed_at": utc_now(), "pid": os.getpid(), "endpoint": self.endpoint, "request": body, "request_sha256": sha256(request_bytes), "attempts": attempts, "last_attempt_response_observed": True, "last_attempt_usage_known": False})
                if not retryable or attempt + 1 == self.max_attempts:
                    atomic_json(path, {"status": "http_error", "audit_id": audit_id, "phase": phase, "request_id": request_id, "started_at": started, "completed_at": utc_now(), "endpoint": self.endpoint, "request": body, "request_sha256": sha256(request_bytes), "attempts": attempts})
                    raise TeacherError(f"Kimi HTTP {exc.code}; audit {audit_id}; no response accepted") from None
                time.sleep(delay)
            except (URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                attempts.append({"attempt": attempt + 1, "error_type": type(exc).__name__, "elapsed_seconds": time.monotonic() - before})
                atomic_json(path, {"status": "retry_wait", "audit_id": audit_id, "phase": phase, "request_id": request_id, "started_at": started, "last_attempt_completed_at": utc_now(), "pid": os.getpid(), "endpoint": self.endpoint, "request": body, "request_sha256": sha256(request_bytes), "attempts": attempts, "last_attempt_response_observed": False, "last_attempt_usage_known": False})
                if attempt + 1 == self.max_attempts:
                    atomic_json(path, {"status": "transport_error", "audit_id": audit_id, "phase": phase, "request_id": request_id, "started_at": started, "completed_at": utc_now(), "endpoint": self.endpoint, "request": body, "request_sha256": sha256(request_bytes), "attempts": attempts})
                    raise TeacherError(f"Kimi transport error; audit {audit_id}; no response accepted") from None
                time.sleep(min(60.0, 2 ** (attempt + 1) + random.random()))
        raise AssertionError("Unreachable request state")

    @staticmethod
    def _result(audit: dict[str, Any], *, cache_hit: bool) -> TeacherResult:
        response = audit["response"]
        choices = response.get("choices", [])
        if len(choices) != 1:
            raise TeacherError("Expected exactly one teacher completion")
        choice = choices[0]
        if choice.get("finish_reason") not in ("stop", "end_turn"):
            raise TeacherError(f"Incomplete teacher response: {choice.get('finish_reason')}")
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise TeacherError("Teacher returned no JSON content")
        content = content.strip()
        if content.startswith("```"):
            match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                content = match.group(1)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            raise TeacherError("Teacher completion was not valid JSON") from None
        return TeacherResult(parsed=parsed, audit_id=audit["audit_id"], model=response.get("model", "unknown"), usage=response.get("usage", {}), response_sha256=audit["response_sha256"], cache_hit=cache_hit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", default="local/teacher-probe")
    args = parser.parse_args()
    client = TeacherClient(args.audit_dir, max_attempts=2)
    result = client.complete_json(
        "You are a JSON API for a software research project. Return only valid JSON.",
        'Return exactly {"ok":true,"task":"clipboard-candidate-ranking"}. Do not include explanations.',
        max_tokens=2048,
        temperature=1.0,
        phase="connectivity-probe",
        request_id="probe-v1",
    )
    print(json.dumps({"ok": result.parsed.get("ok") is True, "model": result.model, "usage": result.usage, "audit_id": result.audit_id, "cache_hit": result.cache_hit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
