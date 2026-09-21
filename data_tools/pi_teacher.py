"""Isolated pi/devin/swe-2 teacher calls with provider-specific audit/cache state."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid

from data_tools.rate_limit import AccountCoordinator, AccountPaused
from data_tools.resources import load_resources
from data_tools.teacher import TeacherClient, TeacherError, atomic_json, audit_identity, canonical_bytes, sha256, utc_now, verify_audit_identity

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "tools/pi_teacher_extension.ts"
PROVIDER_EXTENSION = ROOT.parent / "pi-devin/extensions/index.ts"
RATE_DIRECTORY = ROOT / "local/pi-swe2-account-rate"
POLICY_PATH = ROOT / "configs/teacher_transition_swe2.json"


def pi_coordinator():
    # This is an independent local courtesy cap, not a provider entitlement.
    resources = load_resources()
    return AccountCoordinator(RATE_DIRECTORY, max_in_flight=resources[0]["global_max_in_flight"] if resources else 6)


def private_bytes(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def normalize_usage(raw):
    """Preserve reported counters without inventing tokens from provider zeros."""
    raw = raw if isinstance(raw, dict) else {}
    counters = {key: value for key, value in raw.items() if key in ("input", "output", "cacheRead", "cacheWrite", "totalTokens") and type(value) in (int, float) and value >= 0}
    if not counters or not any(counters.values()):
        return {}, "unknown_all_zero_or_absent_provider_counters"
    usage = {}
    for source, target in (("input", "prompt_tokens"), ("output", "completion_tokens"), ("totalTokens", "total_tokens")):
        if source in counters:
            usage[target] = counters[source]
    return usage, "provider_reported_input_output_total; cacheRead/cacheWrite remain separate; inclusion not inferred"


def observed_pi_completions(stdout, common):
    """Cost observations survive receipt, JSON-label and process-exit rejection."""
    result, seen = [], set()
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message", {})
        if not isinstance(message, dict):
            continue
        if event.get("type") != "message_end" or message.get("role") != "assistant":
            continue
        key = sha256(canonical_bytes(message))
        if key in seen:
            continue
        seen.add(key)
        usage, semantics = normalize_usage(message.get("usage"))
        content = message.get("content", [])
        content = content if isinstance(content, list) else []
        normalized = {"model": message.get("model", "unknown"), "choices": [{"finish_reason": message.get("stopReason"), "message": {"role": "assistant", "content": "\n".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")}}], "usage": usage}
        result.append({**common, "attempt_id": common["attempt_id"] + ":message:" + key, "response": normalized, "response_sha256": sha256(canonical_bytes(normalized)), "raw_usage": message.get("usage"), "usage_known": bool(usage), "accounting_semantics": semantics, "completion_validated": False})
    return result


def pi_error_messages(stdout):
    errors = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message", {})
        if isinstance(message, dict) and message.get("role") == "assistant" and message.get("stopReason") in ("error", "aborted"):
            errors.append(str(message.get("errorMessage") or message["stopReason"]))
        elif event.get("type") in ("error", "auto_retry_start", "auto_retry_end") and (event.get("error") or event.get("errorMessage")):
            errors.append(str(event.get("errorMessage") or event.get("error")))
    return errors


def record_pi_failure(coordinator, *, timed_out, stdout, stderr):
    # Inspect provider errors, not synthetic candidate/assistant text that may
    # itself mention permissions or rate limits.
    text = ("\n".join(pi_error_messages(stdout)) + "\n" + stderr.decode(errors="replace")).lower()
    authentication = any(word in text for word in ("unauthorized", "authentication failed", "invalid api key", "invalid token", "permission denied", "insufficient credits", "quota exceeded", "entitlement"))
    transient = timed_out or any(word in text for word in ("timeout", "timed out", "temporarily", "overloaded", "rate limit", "429", "502", "503", "504", "econnreset", "etimedout", "econnrefused", "enotfound", "fetch failed", "ended without an eos trailer", "network error", "socket hang up", "connection reset"))
    resources = load_resources()
    with coordinator._state() as state:
        failures = state.setdefault("pi_failure_counts", {})
        kind = "authentication_or_entitlement" if authentication else "transient_transport" if transient else "configuration_or_unclassified_process_error"
        failures[kind] = failures.get(kind, 0) + 1
        state["last_pi_failure"] = {"classification": kind, "at": time.time(), "stdout_sha256": sha256(stdout), "stderr_sha256": sha256(stderr)}
        if re.search(r"\b429\b", text) and resources is not None:
            document, binding = resources
            previous_cap = state["max_in_flight"]
            state["max_in_flight"] = min(previous_cap, document["fallback_max_in_flight"])
            reduction = {"at": time.time(), "reason": "observed_provider_429", "resource_supplement": binding, "previous_max_in_flight": previous_cap, "effective_max_in_flight": state["max_in_flight"], "automatic_reincrease": False}
            state["resource_concurrency_reduction"] = reduction
            state.setdefault("resource_concurrency_reduction_history", []).append(reduction)
        if transient and not authentication:
            delay = min(900, 60 * 2 ** min(failures[kind] - 1, 4))
            state["cooldown_until"] = max(state.get("cooldown_until", 0), time.time() + delay)
            state["pause_reason"] = "pi_transient_transport_backoff"
            state["reset_source"] = "bounded_local_retry_backoff"
        else:
            state["blocked_reason"] = "pi_" + kind
    return kind


def parse_pi_completion(stdout, receipt_bytes, bound_bytes, *, provider="devin", model="swe-2"):
    bound = json.loads(bound_bytes)
    receipt = json.loads(receipt_bytes)
    expected = {"version": "pastewhat-pi-teacher-receipt-v1", "transport": "pi-cli-json", "provider": provider, "requested_model": model, "request_sha256": sha256(bound_bytes), "max_tokens": bound["max_tokens"], "thinking": bound["thinking"], "provider_call_count": 1, "isolated": True, "context_message_count": 1, "tools_count": 0, "usage_source": "pi-devin-provider-reported-or-unknown"}
    if any(receipt.get(key) != value for key, value in expected.items()) or not isinstance(receipt.get("actual_model"), str) or not receipt["actual_model"]:
        raise ValueError("Pi receipt does not prove the exact single-call visible input")
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("Pi output contains a non-event JSON value")
    endings = [event["message"] for event in events if event.get("type") == "message_end" and event.get("message", {}).get("role") == "assistant"]
    if len(endings) != 1:
        raise ValueError("Pi must return exactly one assistant message_end")
    message = endings[0]
    if message.get("stopReason") != "stop" or message.get("provider") != provider or message.get("model") != model or any(item.get("type") == "toolCall" for item in message.get("content", [])):
        raise ValueError("Pi returned a different model or attempted an unavailable tool")
    content = "\n".join(item["text"] for item in message.get("content", []) if item.get("type") == "text")
    usage, semantics = normalize_usage(message.get("usage"))
    normalized = {"model": receipt["actual_model"], "choices": [{"finish_reason": message.get("stopReason"), "message": {"role": "assistant", "content": content}}], "usage": usage}
    return {"receipt": receipt, "response": normalized, "response_sha256": sha256(canonical_bytes(normalized)), "response_format": "normalized-from-pi-message-end; raw-events-preserved", "raw_usage": message.get("usage"), "usage_known": bool(usage), "accounting_semantics": semantics, "reported_message_model": message.get("model")}


class PiTeacherClient:
    """One pi process and at most one provider call per attempted completion."""

    def __init__(self, audit_dir, *, timeout=480):
        self.audit_dir = Path(audit_dir).resolve()
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.model = "swe-2"
        self.provider = "devin"
        self.coordinator = pi_coordinator()
        self.binary = shutil.which("pi")
        if not self.binary or not EXTENSION.is_file() or not PROVIDER_EXTENSION.is_file():
            raise TeacherError("Pi transport configuration is missing; no request sent")
        version = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=15, check=True).stdout.strip()
        self.policy = json.loads(POLICY_PATH.read_text())
        self.policy_sha256 = sha256(POLICY_PATH.read_bytes())
        self.runtime = {"pi_version": version, "pi_executable_sha256": sha256(Path(self.binary).resolve().read_bytes()), "teacher_extension_sha256": sha256(EXTENSION.read_bytes()), "provider_extension_sha256": sha256(PROVIDER_EXTENSION.read_bytes())}
        if self.policy.get("provider") != self.provider or self.policy.get("model") != self.model or self.policy.get("transport") != "pi-cli-json" or any(self.policy["runtime"].get(key) != value for key, value in self.runtime.items()):
            raise TeacherError("Pi transport runtime differs from the registered teacher transition")
        pins_path = ROOT / self.policy["runtime"]["pins_path"]
        if sha256(pins_path.read_bytes()) != self.policy["runtime"]["pins_sha256"]:
            raise TeacherError("Pi transport source pins changed")
        pins = json.loads(pins_path.read_text())
        provider_root = ROOT / pins["provider"]["repository"]
        if any(sha256((provider_root / name).read_bytes()) != expected for name, expected in pins["provider"]["source_files"].items()):
            raise TeacherError("Pi transport provider source differs from pinned files")
        self.source_files = [{"path": str(source.relative_to(ROOT)), "sha256": sha256(source.read_bytes())} for source in (POLICY_PATH, pins_path, EXTENSION)]
        self.runtime.update(teacher_transition_sha256=self.policy_sha256, runtime_pins_sha256=self.policy["runtime"]["pins_sha256"], isolation_flags=["--offline", "--no-session", "--no-tools", "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files", "--no-approve"], stdin="DEVNULL", telemetry="disabled", agent_system_prompt="Execute only the isolated bound teacher request.")

    _result = staticmethod(TeacherClient._result)

    def _artifact(self, path, content):
        private_bytes(path, content)
        return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha256(content), "bytes": len(content)}

    def complete_json(self, system, user, *, max_tokens=16384, temperature=1.0, thinking=None, reasoning_effort=None, response_format=None, phase, request_id):
        requested_effort = reasoning_effort or ("off" if thinking == "disabled" else "medium")
        effective_effort = self.policy["reasoning_map"].get(requested_effort)
        if effective_effort not in ("medium", "high", "max") or type(max_tokens) is not int or not 1 <= max_tokens <= 65536:
            raise ValueError("Unsupported pi teacher effort or output budget")
        request = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "max_tokens": max_tokens, "reasoning_effort": effective_effort}
        bound = {"system": system, "user": user, "max_tokens": max_tokens, "thinking": effective_effort}
        bound_bytes = canonical_bytes(bound)
        if sha256(POLICY_PATH.read_bytes()) != self.policy_sha256:
            raise TeacherError("Pi transport transition policy changed during production")
        base = {"transport": "pi-cli-json", "provider": self.provider, "runtime": self.runtime, "source_files": self.source_files, "teacher_transition_sha256": self.policy_sha256, "request": request, "request_sha256": sha256(canonical_bytes(request)), "bound_request_sha256": sha256(bound_bytes), "phase": phase, "request_id": request_id, "requested_controls": {"temperature": temperature, "thinking": thinking, "reasoning_effort": reasoning_effort, "response_format": response_format}, "unapplied_controls": ["temperature", "response_format"], "effective_thinking": effective_effort}
        identifier = audit_identity(base)
        base["audit_id"] = identifier
        path = self.audit_dir / (identifier + ".json")
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and previous.get("status") == "success":
            verify_audit_identity(previous, identifier)
            return self._result(previous, cache_hit=True)
        history = list(previous.get("prior_observed_responses", [])) if previous else []
        if previous and previous.get("response") is not None:
            history.append({key: value for key, value in previous.items() if key != "prior_observed_responses"})
        elif previous:
            history.extend(previous.get("observed_completions", []))
        attempts = list(previous.get("attempts", [])) if previous else []
        if previous and previous.get("status") == "request_started":
            attempts.append({"error_type": "previous_started_attempt_has_no_observed_completion", "attempt_id": previous.get("attempt_id"), "usage_known": False})

        def save(record):
            if history:
                record["prior_observed_responses"] = history
            atomic_json(path, record)

        try:
            lease = self.coordinator.acquire(self.timeout)
        except AccountPaused as error:
            raise TeacherError(f"Pi account paused: {error}") from None
        attempt_id = uuid.uuid4().hex
        started = utc_now()
        before = time.monotonic()
        common = {**base, "pid": os.getpid(), "lease_id": lease, "attempt_id": attempt_id, "started_at": started, "attempts": attempts}
        resources = load_resources()
        if resources is not None:
            # Scheduling is audit-only: it must not invalidate a successful
            # teacher response cache or alter the actual bound request.
            common["resource_supplement"] = resources[1]
            common["effective_local_max_in_flight"] = self.coordinator.status()["max_in_flight"]
        save({**common, "status": "request_started", "attempt_started_at": started})
        artifacts = self.audit_dir / "pi-events" / identifier / attempt_id
        try:
            with tempfile.TemporaryDirectory(prefix="pastewhat-pi-teacher-") as directory:
                working = Path(directory)
                request_file, receipt_file = working / "request.json", working / "receipt.json"
                private_bytes(request_file, bound_bytes)
                environment = os.environ.copy()
                environment.update(PASTEWHAT_PI_REQUEST_FILE=str(request_file), PASTEWHAT_PI_RECEIPT_FILE=str(receipt_file), PI_TELEMETRY="0")
                command = [self.binary, "--provider", self.provider, "--model", self.model, "--thinking", effective_effort, "--mode", "json", "-p", "--offline", "--no-session", "--no-tools", "--no-extensions", "-e", str(EXTENSION), "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files", "--no-approve", "--system-prompt", self.runtime["agent_system_prompt"], "Execute bound synthetic teacher request."]
                process = subprocess.Popen(command, cwd=working, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                timed_out = False
                try:
                    stdout, stderr = process.communicate(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        stdout, stderr = process.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        stdout, stderr = process.communicate()
                receipt_bytes = receipt_file.read_bytes() if receipt_file.exists() else b""
            self.coordinator.release(lease)
            raw_events = self._artifact(artifacts / "events.jsonl", stdout)
            raw_receipt = self._artifact(artifacts / "receipt.json", receipt_bytes)
            evidence = {"raw_event_files": [raw_events], "receipt_file": raw_receipt, "bound_request_file": self._artifact(artifacts / "request.json", bound_bytes), "stderr_sha256": sha256(stderr), "stderr_bytes": len(stderr), "process_returncode": process.returncode, "elapsed_seconds": time.monotonic() - before, "completed_at": utc_now(), "observed_completions": observed_pi_completions(stdout, common)}
            provider_errors = pi_error_messages(stdout)
            if timed_out or process.returncode != 0 or provider_errors:
                evidence["provider_error_sha256"] = [sha256(error.encode()) for error in provider_errors]
                attempts.append({"attempt_id": attempt_id, "error_type": "PiTimeout" if timed_out else "PiProviderError" if provider_errors else "PiProcessError", "exit_code": process.returncode, "usage_known": any(row["usage_known"] for row in evidence["observed_completions"])})
                save({**common, **evidence, "status": "transport_error", "attempts": attempts})
                record_pi_failure(self.coordinator, timed_out=timed_out, stdout=stdout, stderr=stderr)
                raise TeacherError(f"Pi transport failure; audit {identifier}; preserve unknown usage and inspect actual provider error")
            audit = {**common, **evidence, **parse_pi_completion(stdout, receipt_bytes, bound_bytes, provider=self.provider, model=self.model), "status": "success", "teacher_model_is_rolling": True}
            try:
                result = self._result(audit, cache_hit=False)
            except TeacherError as error:
                audit.update(status="invalid_response", validation_error=str(error))
                save(audit)
                raise
            save(audit)
            return result
        except TeacherError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            record = {**common, "status": "invalid_response", "completed_at": utc_now(), "validation_error": str(error), "usage_known": False}
            if "evidence" in locals():
                record.update(evidence)
            save(record)
            raise TeacherError(f"Pi response validation failed; audit {identifier}: {error}") from None
        finally:
            self.coordinator.release(lease)
