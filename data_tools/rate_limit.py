"""Account-wide request coordination, without prompts or credentials.

The file lock is shared by Train/Dev and the independent evaluator. This is a
local courtesy limit, not a claim about the account's server-side entitlement.
"""

from __future__ import annotations

from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import uuid


DEFAULT_DIRECTORY = Path(__file__).resolve().parents[1] / "local/kimi-account-rate"
DEFAULT_MAX_IN_FLIGHT = 6


class AccountPaused(RuntimeError):
    pass


def retry_after_seconds(headers, now=None):
    value = headers.get("Retry-After") if headers else None
    if not value:
        return None
    now = time.time() if now is None else now
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - now)
        except (TypeError, ValueError, OverflowError):
            return None


def classify_http_failure(status, response_text):
    """Classify current official Kimi Code errors; do not reinterpret 403 as 429."""
    message = response_text.lower()
    if status == 403:
        if "5-hour" in message or "5 hour" in message or "5h" in message:
            return "quota_5h", 5 * 60 * 60
        if "weekly" in message or "7-day" in message or "7 day" in message:
            return "quota_weekly", 7 * 24 * 60 * 60
        if "monthly" in message or "month" in message:
            return "quota_monthly", 31 * 24 * 60 * 60
        if "concurrent" in message:
            return "account_concurrent_restriction", None
        if "quota" in message or "usage limit" in message:
            return "quota_unknown_reset", None
        return "permission_denied", None
    if status == 401:
        return "authentication_or_model_entitlement", None
    if status == 402:
        return "membership_status", None
    if status == 429:
        return "short_term_rate_or_overload", 0
    return "transient_http_error", 0


class AccountCoordinator:
    def __init__(self, directory=DEFAULT_DIRECTORY, *, min_interval=0.5, max_in_flight=DEFAULT_MAX_IN_FLIGHT):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "state.json"
        self.lock_path = self.directory / "state.lock"
        self.min_interval = max(0.0, min_interval)
        self.max_in_flight = max(1, max_in_flight)

    @contextmanager
    def _state(self):
        with self.lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.is_file() else {
                "version": 1, "leases": {}, "next_start_at": 0.0,
                "cooldown_until": 0.0, "blocked_reason": None,
                "http_error_counts": {}, "max_in_flight": self.max_in_flight,
            }
            # Existing processes cannot raise a previously selected local cap.
            state["max_in_flight"] = min(state.get("max_in_flight", self.max_in_flight), self.max_in_flight)
            now = time.time()
            for identifier, lease in list(state["leases"].items()):
                try:
                    os.kill(lease["pid"], 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    alive = True
                if not alive or lease["expires_at"] < now:
                    del state["leases"][identifier]
            try:
                yield state
            finally:
                state["updated_at"] = time.time()
                temporary = self.path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
                with temporary.open("w") as stream:
                    json.dump(state, stream, sort_keys=True, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(self.path)

    def status(self):
        with self._state() as state:
            now = time.time()
            pause = state.get("blocked_reason") or (state.get("pause_reason") if state.get("cooldown_until", 0) > now else None)
            return {
                "paused": bool(pause), "reason": pause,
                "cooldown_until": state.get("cooldown_until", 0),
                "remaining_seconds": max(0, round(state.get("cooldown_until", 0) - now)),
                "blocked_until_external_change": bool(state.get("blocked_reason")),
                "in_flight": len(state["leases"]),
                "max_in_flight": state["max_in_flight"],
                "http_error_counts": dict(state["http_error_counts"]),
                **({"pi_failure_counts": dict(state["pi_failure_counts"])} if "pi_failure_counts" in state else {}),
                **({"resource_activation": state["resource_activation"]} if "resource_activation" in state else {}),
                **({"resource_concurrency_reduction": state["resource_concurrency_reduction"]} if "resource_concurrency_reduction" in state else {}),
            }

    def record_console_reset(self, *, observed_at, countdown_seconds, safety_seconds, source_url):
        """Record an independently observed official account reset countdown.

        This is an explicit operator action, never a response to a failed request.
        It only refines an unknown-reset five-hour fallback; permission, weekly,
        monthly and server Retry-After restrictions cannot be shortened here.
        """
        if source_url != "https://www.kimi.com/code/console":
            raise ValueError("Reset evidence must be the official account console")
        if not 0 < countdown_seconds <= 5 * 60 * 60 or safety_seconds < 60:
            raise ValueError("Invalid reset countdown or insufficient safety margin")
        now = time.time()
        if observed_at > now or now - observed_at > 30 * 60:
            raise ValueError("Console observation must be recent and not in the future")
        resume_at = observed_at + countdown_seconds + safety_seconds
        if resume_at <= now:
            raise ValueError("Conservative observed reset must still be in the future")
        with self._state() as state:
            if state.get("blocked_reason") or state.get("pause_reason") != "quota_5h":
                raise ValueError("Only a five-hour quota cooldown can use this evidence")
            if state.get("reset_source") != "conservative_full_window_from_error":
                raise ValueError("Only an unknown server reset fallback can be refined")
            if observed_at < state.get("last_http_error", {}).get("at", 0):
                raise ValueError("Console observation predates the last quota error")
            evidence = {
                "source_url": source_url, "observed_at": observed_at,
                "visible_countdown_seconds": countdown_seconds,
                "safety_seconds": safety_seconds, "resume_at": resume_at,
                "previous_fallback_until": state["cooldown_until"],
                "quota_error": dict(state.get("last_http_error", {})),
                "recorded_at": now,
            }
            state.setdefault("reset_evidence_history", []).append(evidence)
            state["cooldown_until"] = resume_at
            state["reset_source"] = "official_console_countdown_with_safety_margin"
            return evidence

    def begin_operator_drain(self, *, seconds=600, reason="prioritize unchanged pilot batches"):
        """Pause new starts for an explicit local scheduler handoff, not quota.

        Existing leases finish normally. This cannot replace a provider pause,
        change the local cap or extend any account entitlement.
        """
        if not 60 <= seconds <= 900:
            raise ValueError("An operator drain must be bounded to1–15 minutes")
        with self._state() as state:
            now = time.time()
            if state.get("blocked_reason") or state.get("cooldown_until", 0) > now:
                raise ValueError("Provider/account pause already active; cannot begin an operator drain")
            record = {"token": uuid.uuid4().hex, "started_at": now, "deadline": now + seconds,
                      "reason": reason, "operator_action": True,
                      "previous": {key: state.get(key) for key in ("cooldown_until", "pause_reason", "reset_source")},
                      "leases_at_start": {key: dict(value) for key, value in state["leases"].items()},
                      "http_error_counts_at_start": dict(state["http_error_counts"])}
            state["operator_drain"] = record
            state["cooldown_until"] = record["deadline"]
            state["pause_reason"] = "operator_scheduler_drain"
            state["reset_source"] = "bounded_local_scheduler_maintenance"
            return record

    def finish_operator_drain(self, token):
        """Release only this operator's pause, preserving newer provider limits."""
        with self._state() as state:
            record = state.get("operator_drain")
            if not record or record.get("token") != token:
                raise ValueError("Operator drain ownership does not match")
            owned = state.get("pause_reason") == "operator_scheduler_drain" and state.get("cooldown_until") == record["deadline"] and not state.get("blocked_reason")
            event = {**record, "finished_at": time.time(), "released_own_pause": owned,
                     "remaining_leases": len(state["leases"]), "http_error_counts_at_finish": dict(state["http_error_counts"])}
            if owned:
                for key, value in record["previous"].items():
                    if value is None:
                        state.pop(key, None)
                    else:
                        state[key] = value
            state.setdefault("operator_drain_history", []).append(event)
            state.pop("operator_drain", None)
            return event

    def acquire(self, request_timeout):
        identifier = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
        while True:
            with self._state() as state:
                now = time.time()
                if state.get("blocked_reason"):
                    raise AccountPaused(f"Kimi account paused: {state['blocked_reason']}; external state change required")
                remaining = state.get("cooldown_until", 0) - now
                if remaining > 0 and state.get("pause_reason", "").startswith("quota_"):
                    raise AccountPaused(f"Kimi quota cooldown active; {round(remaining)} seconds remain")
                wait = max(0.0, remaining, state.get("next_start_at", 0) - now)
                if wait <= 0 and len(state["leases"]) < state["max_in_flight"]:
                    state["leases"][identifier] = {"pid": os.getpid(), "expires_at": now + request_timeout + 60}
                    state["next_start_at"] = now + self.min_interval
                    return identifier
                if wait <= 0:
                    wait = 0.25
            time.sleep(min(30.0, wait))

    def release(self, identifier):
        with self._state() as state:
            state["leases"].pop(identifier, None)

    def record_http_failure(self, status, response_text, headers, suggested_delay):
        reason, fallback = classify_http_failure(status, response_text)
        server_delay = retry_after_seconds(headers)
        with self._state() as state:
            key = str(status)
            state["http_error_counts"][key] = state["http_error_counts"].get(key, 0) + 1
            state["last_http_error"] = {"status": status, "reason": reason, "at": time.time()}
            if status in (401, 402, 403):
                if reason.startswith("quota_") and (server_delay is not None or fallback is not None):
                    delay = server_delay if server_delay is not None else fallback
                    state["cooldown_until"] = max(state.get("cooldown_until", 0), time.time() + delay)
                    state["pause_reason"] = reason
                    state["reset_source"] = "retry_after" if server_delay is not None else "conservative_full_window_from_error"
                else:
                    state["blocked_reason"] = reason
            elif status == 429:
                delay = max(suggested_delay, server_delay or 0)
                state["cooldown_until"] = max(state.get("cooldown_until", 0), time.time() + delay)
                state["pause_reason"] = reason
                state["reset_source"] = "retry_after_or_backoff"
        return reason
