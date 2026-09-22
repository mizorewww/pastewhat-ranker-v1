"""Isolated pi/devin/swe-2 teacher calls with provider-specific audit/cache state."""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from data_tools.rate_limit import AccountCoordinator, AccountPaused
from data_tools.resources import load_resources
from data_tools.teacher import (
    TeacherClient,
    TeacherError,
    atomic_json,
    audit_identity,
    canonical_bytes,
    sha256,
    utc_now,
    verify_audit_identity,
)

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "tools/pi_teacher_extension.ts"
PROVIDER_EXTENSION = ROOT.parent / "pi-devin/extensions/index.ts"
RATE_DIRECTORY = ROOT / "local/pi-swe2-account-rate"
POLICY_PATH = ROOT / "configs/teacher_transition_swe2.json"
CORRECTION_PATH = ROOT / "configs/teacher_correction_swe2_uid.json"


class PiModelIdentityError(ValueError):
    pass


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


def pi_rate_limit_evidence(stdout, stderr=b""):
    """Read provider error events, never generated candidate text or billing."""
    text = ("\n".join(pi_error_messages(stdout)) + "\n" + stderr.decode(errors="replace")).lower()
    numeric_429 = bool(re.search(r"\b429\b", text))
    explicit_limit = "reached free model rate limit" in text
    if not numeric_429 and not explicit_limit:
        return None
    countdowns = []
    for number, unit in re.findall(r"\byour limit will reset in\s+(\d+)\s+(seconds?|minutes?|hours?)\b", text):
        multiplier = 1 if unit.startswith("second") else 60 if unit.startswith("minute") else 3600
        seconds = int(number) * multiplier
        if 0 <= seconds <= 7 * 86400:
            countdowns.append(seconds)
    return {"signal": "provider_error_contains_429" if numeric_429 else "explicit_provider_free_model_rate_limit", "http_status_observed": False, "provider_reset_countdown_seconds": max(countdowns) if countdowns else None, "stdout_sha256": sha256(stdout), "stderr_sha256": sha256(stderr)}


def record_pi_failure(coordinator, *, timed_out, stdout, stderr, process_returncode=None):
    # Inspect provider errors, not synthetic candidate/assistant text that may
    # itself mention permissions or rate limits.
    text = ("\n".join(pi_error_messages(stdout)) + "\n" + stderr.decode(errors="replace")).lower()
    authentication = any(word in text for word in ("unauthorized", "authentication failed", "invalid api key", "invalid token", "permission denied", "insufficient credits", "quota exceeded", "entitlement"))
    # Pi can exit with shell-style 128+signal codes when the local process is
    # interrupted. A signal without provider error evidence is not a durable
    # configuration failure; keep its unknown usage audit and bounded backoff.
    interrupted = (process_returncode in {int(sig) + 128 for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGKILL)} or process_returncode in {-int(sig) for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGKILL)}) and not pi_error_messages(stdout) and not stderr.strip()
    transient = timed_out or interrupted or any(word in text for word in ("timeout", "timed out", "temporarily", "overloaded", "rate limit", "429", "502", "503", "504", "econnreset", "etimedout", "econnrefused", "enotfound", "fetch failed", "ended without an eos trailer", "network error", "socket hang up", "connection reset"))
    rate_limit = pi_rate_limit_evidence(stdout, stderr)
    resources = load_resources()
    with coordinator._state() as state:
        failures = state.setdefault("pi_failure_counts", {})
        kind = "authentication_or_entitlement" if authentication else "provider_rate_limit" if rate_limit else "transient_transport" if transient else "configuration_or_unclassified_process_error"
        failures[kind] = failures.get(kind, 0) + 1
        if kind == "transient_transport":
            state["pi_transient_failure_streak"] = state.get("pi_transient_failure_streak", 0) + 1
        state["last_pi_failure"] = {"classification": kind, "at": time.time(), "stdout_sha256": sha256(stdout), "stderr_sha256": sha256(stderr), "process_returncode": process_returncode, "local_interruption": interrupted}
        if rate_limit is not None and resources is not None:
            document, binding = resources
            previous_cap = state["max_in_flight"]
            state["max_in_flight"] = min(previous_cap, document["fallback_max_in_flight"])
            reduction = {"at": time.time(), "reason": "observed_provider_rate_limit", "evidence": rate_limit, "resource_supplement": binding, "previous_max_in_flight": previous_cap, "effective_max_in_flight": state["max_in_flight"], "automatic_reincrease": False}
            state["resource_concurrency_reduction"] = reduction
            state.setdefault("resource_concurrency_reduction_history", []).append(reduction)
        if rate_limit is not None and not authentication:
            observed_at = time.time()
            countdown = rate_limit["provider_reset_countdown_seconds"]
            delay = countdown + 5 if countdown is not None else min(900, 60 * 2 ** min(failures[kind] - 1, 4))
            evidence = {**rate_limit, "observed_at": observed_at, "safety_seconds": 5 if countdown is not None else None, "deadline": observed_at + delay, "reference": "completed_provider_error_events_plus_reported_countdown"}
            state["last_pi_rate_limit"] = evidence
            state.setdefault("pi_rate_limit_history", []).append(evidence)
            state["cooldown_until"] = max(state.get("cooldown_until", 0), evidence["deadline"])
            state["pause_reason"] = "pi_provider_rate_limit"
            state["reset_source"] = "provider_error_countdown_with_safety_margin" if countdown is not None else "bounded_local_retry_backoff"
        elif transient and not authentication:
            delay = min(900, 60 * 2 ** min(state["pi_transient_failure_streak"] - 1, 4))
            state["cooldown_until"] = max(state.get("cooldown_until", 0), time.time() + delay)
            state["pause_reason"] = "pi_transient_transport_backoff"
            state["reset_source"] = "bounded_local_retry_backoff"
        else:
            state["blocked_reason"] = "pi_" + kind
    return kind


def record_pi_success(coordinator, *, audit_id, attempt_id):
    """A validated fresh completion ends only the transient failure streak."""
    with coordinator._state() as state:
        state["pi_transient_failure_streak"] = 0
        state["last_pi_success"] = {"at": time.time(), "audit_id": audit_id, "attempt_id": attempt_id}


def parse_pi_completion(stdout, receipt_bytes, bound_bytes, *, provider="devin", model="swe-2"):
    bound = json.loads(bound_bytes)
    receipt = json.loads(receipt_bytes)
    expected = {"version": "pastewhat-pi-teacher-receipt-v1", "transport": "pi-cli-json", "provider": provider, "requested_model": model, "request_sha256": sha256(bound_bytes), "max_tokens": bound["max_tokens"], "thinking": bound["thinking"], "provider_call_count": 1, "isolated": True, "context_message_count": 1, "tools_count": 0, "usage_source": "pi-devin-provider-reported-or-unknown"}
    if any(receipt.get(key) != value for key, value in expected.items()) or not isinstance(receipt.get("actual_model"), str) or not receipt["actual_model"]:
        raise ValueError("Pi receipt does not prove the exact single-call visible input")
    if bound["thinking"] not in ("medium", "high", "max") or receipt["actual_model"] != model + "-" + bound["thinking"]:
        raise PiModelIdentityError("Pi actual UID differs from the requested SWE-2 thinking variant")
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
        self.correction = None
        self.correction_sha256 = None
        registered = self.policy["runtime"]
        self.extension = EXTENSION
        if CORRECTION_PATH.is_file():
            self.correction = json.loads(CORRECTION_PATH.read_text())
            self.correction_sha256 = sha256(CORRECTION_PATH.read_bytes())
            expected_mapping = {level: "swe-2-" + level for level in ("medium", "high", "max")}
            if self.correction.get("version") != "pastewhat-teacher-runtime-correction-v1" or self.correction.get("parent_transition") != {"path": str(POLICY_PATH.relative_to(ROOT)), "sha256": self.policy_sha256} or self.correction.get("expected_model_mapping") != expected_mapping or any(self.correction.get(key) != self.policy.get(key) for key in ("run_id", "run_plan_sha256")):
                raise TeacherError("Pi runtime correction differs from its registered parent or exact model mapping")
            registered = self.correction["runtime"]
            self.extension = ROOT / registered["teacher_extension_path"]
        self.runtime = {"pi_version": version, "pi_executable_sha256": sha256(Path(self.binary).resolve().read_bytes()), "teacher_extension_sha256": sha256(self.extension.read_bytes()), "provider_extension_sha256": sha256(PROVIDER_EXTENSION.read_bytes())}
        if self.policy.get("provider") != self.provider or self.policy.get("model") != self.model or self.policy.get("transport") != "pi-cli-json" or any(registered.get(key) != value for key, value in self.runtime.items()):
            raise TeacherError("Pi transport runtime differs from the registered teacher transition")
        pins_path = ROOT / registered["pins_path"]
        if sha256(pins_path.read_bytes()) != registered["pins_sha256"]:
            raise TeacherError("Pi transport source pins changed")
        pins = json.loads(pins_path.read_text())
        provider_root = ROOT / pins["provider"]["repository"]
        if any(sha256((provider_root / name).read_bytes()) != expected for name, expected in pins["provider"]["source_files"].items()):
            raise TeacherError("Pi transport provider source differs from pinned files")
        sources = [POLICY_PATH, pins_path, self.extension]
        if self.correction is not None:
            sources.append(CORRECTION_PATH)
        self.source_files = [{"path": str(source.relative_to(ROOT)), "sha256": sha256(source.read_bytes())} for source in sources]
        self.runtime.update(teacher_transition_sha256=self.policy_sha256, runtime_pins_sha256=registered["pins_sha256"], isolation_flags=["--offline", "--no-session", "--no-tools", "--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files", "--no-approve"], stdin="DEVNULL", telemetry="disabled", agent_system_prompt="Execute only the isolated bound teacher request.")
        self.legacy_runtime = dict(self.runtime)
        self.legacy_runtime.update(teacher_extension_sha256=self.policy["runtime"]["teacher_extension_sha256"], runtime_pins_sha256=self.policy["runtime"]["pins_sha256"])
        if self.correction is not None:
            self.runtime["teacher_correction_sha256"] = self.correction_sha256

    _result = staticmethod(TeacherClient._result)

    def _artifact(self, path, content):
        private_bytes(path, content)
        return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha256(content), "bytes": len(content)}

    def _checked_cache(self, audit, identifier, bound_bytes):
        verify_audit_identity(audit, identifier)
        artifacts = audit["raw_event_files"] + [audit["receipt_file"], audit["bound_request_file"]]
        contents = {}
        for evidence in artifacts:
            payload = (ROOT / evidence["path"]).read_bytes()
            if sha256(payload) != evidence["sha256"]:
                raise TeacherError("Pi cached response evidence changed")
            contents[evidence["path"]] = payload
        if contents[audit["bound_request_file"]["path"]] != bound_bytes or len(audit["raw_event_files"]) != 1:
            raise TeacherError("Pi cached response is not the exact bound request")
        replay = parse_pi_completion(contents[audit["raw_event_files"][0]["path"]], contents[audit["receipt_file"]["path"]], bound_bytes, provider=self.provider, model=self.model)
        if replay["response_sha256"] != audit["response_sha256"] or replay["receipt"] != audit["receipt"]:
            raise TeacherError("Pi cached response differs from its raw completion")
        return self._result(audit, cache_hit=True)

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
        if self.correction is not None and sha256(CORRECTION_PATH.read_bytes()) != self.correction_sha256:
            raise TeacherError("Pi transport correction changed during production")
        base = {"transport": "pi-cli-json", "provider": self.provider, "runtime": self.runtime, "source_files": self.source_files, "teacher_transition_sha256": self.policy_sha256, "request": request, "request_sha256": sha256(canonical_bytes(request)), "bound_request_sha256": sha256(bound_bytes), "phase": phase, "request_id": request_id, "requested_controls": {"temperature": temperature, "thinking": thinking, "reasoning_effort": reasoning_effort, "response_format": response_format}, "unapplied_controls": ["temperature", "response_format"], "effective_thinking": effective_effort}
        if self.correction is not None:
            base["teacher_correction"] = {"path": str(CORRECTION_PATH.relative_to(ROOT)), "sha256": self.correction_sha256}
        identifier = audit_identity(base)
        base["audit_id"] = identifier
        path = self.audit_dir / (identifier + ".json")
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and previous.get("status") == "success":
            return self._checked_cache(previous, identifier, bound_bytes)
        if self.correction is not None and previous is None:
            legacy_id = audit_identity({**base, "runtime": self.legacy_runtime})
            legacy_path = self.audit_dir / (legacy_id + ".json")
            if legacy_path.exists():
                legacy = json.loads(legacy_path.read_text())
                if legacy.get("status") == "success":
                    try:
                        return self._checked_cache(legacy, legacy_id, bound_bytes)
                    except (TeacherError, ValueError):
                        # Preserve rejected historical evidence and use the
                        # corrected provider for this still-unfinished request.
                        pass
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
                command = [self.binary, "--provider", self.provider, "--model", self.model, "--thinking", effective_effort, "--mode", "json", "-p", "--offline", "--no-session", "--no-tools", "--no-extensions", "-e", str(self.extension), "--no-skills", "--no-prompt-templates", "--no-themes", "--no-context-files", "--no-approve", "--system-prompt", self.runtime["agent_system_prompt"], "Execute bound synthetic teacher request."]
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
                record_pi_failure(self.coordinator, timed_out=timed_out, stdout=stdout, stderr=stderr, process_returncode=process.returncode)
                raise TeacherError(f"Pi transport failure; audit {identifier}; preserve unknown usage and inspect actual provider error")
            audit = {**common, **evidence, **parse_pi_completion(stdout, receipt_bytes, bound_bytes, provider=self.provider, model=self.model), "status": "success", "teacher_model_is_rolling": True}
            try:
                result = self._result(audit, cache_hit=False)
            except TeacherError as error:
                audit.update(status="invalid_response", validation_error=str(error))
                save(audit)
                raise
            save(audit)
            record_pi_success(self.coordinator, audit_id=identifier, attempt_id=attempt_id)
            return result
        except TeacherError:
            raise
        except PiModelIdentityError as error:
            record = {**common, "status": "transport_error", "completed_at": utc_now(), "validation_error": str(error), "usage_known": False}
            if "evidence" in locals():
                record.update(evidence)
            save(record)
            record_pi_failure(self.coordinator, timed_out=False, stdout=b"", stderr=str(error).encode())
            raise TeacherError(f"Pi transport failure; exact model identity rejected; audit {identifier}") from None
        except (OSError, ValueError, KeyError, TypeError) as error:
            record = {**common, "status": "invalid_response", "completed_at": utc_now(), "validation_error": str(error), "usage_known": False}
            if "evidence" in locals():
                record.update(evidence)
            save(record)
            raise TeacherError(f"Pi response validation failed; audit {identifier}: {error}") from None
        finally:
            self.coordinator.release(lease)
