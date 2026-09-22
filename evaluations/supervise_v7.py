"""Supervise only the registered Calibration/Test v7 data producers.

This process reads production metadata, never examples or labels. It adopts
existing producers without signalling them and resumes only after a verified
exit and an unlocked split. The producer owns batch/spec validation and the
shared Pi client owns all account rate limits.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import time
from collections import deque
from pathlib import Path

from data_tools.teacher import atomic_json, utc_now
from evaluations.common import sha256
from run_contract import load_run_plan

ROOT = Path(__file__).resolve().parents[1]
RUN_PLAN = ROOT / "configs/run_plan_efficient.json"
RUN_PLAN_SHA256 = "e5491476a01f3cd3d1b3f3778ac90f4e1dbeadca0e001a078003633266b7963f"
SAMPLING_SHA256 = {
    "calibration": "70283a34699bf3c61bd4281d1d16f1741909c225c387a615bdaf8b10b47d635d",
    "test": "dccf18d2d93df15a8dfbeab936972c95e3c05ae628a8d61244b058fef8d37c3b",
}
SPLITS = ("calibration", "test")
WORKERS = 2
MAX_SITUATIONS = 8
RESTART_WINDOW_SECONDS = 3600
MAX_RESTARTS_PER_WINDOW = 3


def command(split: str) -> list[str]:
    return [str(ROOT / ".venv/bin/python"), "-m", "evaluations.produce_v7",
            "--run-plan", "configs/run_plan_efficient.json", "--split", split,
            "--workers", str(WORKERS), "--max-situations", str(MAX_SITUATIONS)]


def process_identity(pid: int, expected_command: list[str]) -> dict | None:
    if type(pid) is not int or pid < 1:
        return None
    observed = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                              capture_output=True, text=True, check=False)
    if observed.returncode or not observed.stdout.strip():
        return None
    try:
        actual = shlex.split(observed.stdout.strip())
    except ValueError:
        return None
    if actual != expected_command:
        return None
    birth = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                           capture_output=True, text=True, check=False)
    if birth.returncode or not birth.stdout.strip():
        return None
    return {"pid": pid, "started_local": birth.stdout.strip(), "command": actual}


def same_process(prior: dict | None, expected_command: list[str]) -> dict | None:
    current = process_identity(prior.get("pid"), expected_command) if isinstance(prior, dict) else None
    return current if current and current["started_local"] == prior.get("started_local") else None


def split_lock_held(directory: Path) -> bool:
    with (directory / ".production.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def account_pause() -> dict:
    directory = ROOT / "local/pi-swe2-account-rate"
    with (directory / "state.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        state = json.loads((directory / "state.json").read_text())
    now = time.time()
    return {"blocked_reason": state.get("blocked_reason"),
            "cooldown_until": state.get("cooldown_until", 0),
            "paused": bool(state.get("blocked_reason") or state.get("cooldown_until", 0) > now),
            "max_in_flight": state.get("max_in_flight")}


def terminal(directory: Path, plan, split: str) -> str | None:
    path = directory / "production-completion.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    if any(value.get(key) != expected for key, expected in plan.binding().items()):
        raise ValueError("Heldout completion belongs to another run")
    status = value.get("status")
    if status not in {"data_frozen_and_audited", "finite_backfill_exhausted"}:
        raise ValueError("Unexpected heldout completion status")
    if status == "data_frozen_and_audited":
        data = ROOT / plan.data_path(split)
        manifest_path = ROOT / "data/evaluator-manifests" / plan.run_id / (split + ".manifest.json")
        if not data.is_file() or not manifest_path.is_file():
            raise ValueError("Completed heldout split lacks frozen data or manifest")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("episodes") != plan.target(split) or manifest.get("sha256") != sha256(data)
                or any(manifest.get(key) != expected for key, expected in plan.binding().items())):
            raise ValueError("Frozen heldout manifest is not bound to its data")
        audit_path = ROOT / manifest["audit_path"]
        audit = json.loads(audit_path.read_text())
        if (manifest.get("audit_sha256") != sha256(audit_path)
                or audit.get("passed") is not True or audit.get("data_sha256") != sha256(data)):
            raise ValueError("Completed heldout split lacks its passing independent audit")
    return status


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps({"at": utc_now(), **event}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_split_source(directory: Path, plan, split: str) -> None:
    plan.verify_unchanged()
    sampling = directory / "sampling.json"
    if sha256(sampling) != SAMPLING_SHA256[split]:
        raise ValueError("Heldout source sampling changed: " + split)
    binding = json.loads((directory / "owner-binding.json").read_text())
    if any(binding.get(key) != expected for key, expected in plan.binding().items()):
        raise ValueError("Heldout owner binding changed: " + split)


def next_action(*, completion: str | None, identity: dict | None,
                split_locked: bool, paused: bool, recent_restarts: int) -> str:
    if completion:
        return completion if identity is None else "finishing_" + completion
    if identity:
        return "running_or_waiting_for_provider"
    if split_locked:
        return "another_producer_holds_split_lock"
    if paused:
        return "waiting_for_shared_provider"
    if recent_restarts >= MAX_RESTARTS_PER_WINDOW:
        return "restart_limit_requires_review"
    return "launch"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monitor-seconds", type=int, default=20)
    args = parser.parse_args()
    if not 5 <= args.monitor_seconds <= 60:
        parser.error("monitor-seconds must be 5–60")
    os.chdir(ROOT)
    plan = load_run_plan(RUN_PLAN)
    if plan.sha256 != RUN_PLAN_SHA256:
        raise SystemExit("Only the registered v7 run can be supervised")
    supervisor = ROOT / "local/evaluator-v7" / plan.run_id / "supervisor-v7"
    supervisor.mkdir(parents=True, exist_ok=True)
    lock = (supervisor / ".supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = json.loads((supervisor / "status.json").read_text()) if (supervisor / "status.json").is_file() else {}
    if previous and any(previous.get(key) != expected for key, expected in plan.binding().items()):
        raise ValueError("Supervisor state belongs to a different registered run")
    history_path = supervisor / "events.jsonl"
    recent = {split: deque(maxlen=MAX_RESTARTS_PER_WINDOW + 1) for split in SPLITS}
    if history_path.is_file():
        for line in history_path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") == "launched" and event.get("split") in recent:
                recent[event["split"]].append(float(event["unix_time"]))
    owned: dict[str, dict | None] = {split: None for split in SPLITS}
    children: dict[str, subprocess.Popen] = {}
    for split in SPLITS:
        directory = ROOT / "local/evaluator-v7" / plan.run_id / split
        verify_split_source(directory, plan, split)
        producer = directory / "production-process.json"
        if not producer.is_file():
            continue
        value = json.loads(producer.read_text())
        if any(value.get(key) != expected for key, expected in plan.binding().items()):
            raise ValueError("Producer process record belongs to another run")
        identity = process_identity(value.get("pid"), command(split))
        if identity:
            owned[split] = identity
            append_event(history_path, {"event": "adopted", "split": split, **identity})
    atomic_json(supervisor / "process.json", {**plan.binding(), "pid": os.getpid(), "started_at": utc_now(),
                "splits": list(SPLITS), "workers_per_split": WORKERS, "max_situations": MAX_SITUATIONS,
                "student_inference_used": False})
    while True:
        plan.verify_unchanged()
        account = account_pause()
        status = {**plan.binding(), "supervisor_pid": os.getpid(), "updated_at": utc_now(),
                  "account": account, "student_inference_used": False, "splits": {}}
        active_or_waiting = False
        for split in SPLITS:
            directory = ROOT / "local/evaluator-v7" / plan.run_id / split
            verify_split_source(directory, plan, split)
            identity = same_process(owned[split], command(split))
            child = children.get(split)
            exit_code = child.poll() if child else None
            if owned[split] and not identity:
                append_event(history_path, {"event": "exited", "split": split, "pid": owned[split]["pid"],
                                            "started_local": owned[split]["started_local"],
                                            "exit_code": exit_code, "exit_code_known": child is not None})
                owned[split] = None
                children.pop(split, None)
            result = {"pid": identity["pid"] if identity else None,
                      "pid_started_local": identity["started_local"] if identity else None,
                      "last_exit_code": exit_code if child and exit_code is not None else None,
                      "restart_attempts_last_hour": sum(time.time() - t < RESTART_WINDOW_SECONDS for t in recent[split]),
                      "completion": None}
            completion = terminal(directory, plan, split)
            action = next_action(completion=completion, identity=identity,
                                 split_locked=split_lock_held(directory) if identity is None and not completion else False,
                                 paused=account["paused"],
                                 recent_restarts=result["restart_attempts_last_hour"])
            result["completion"] = completion
            result["state"] = action
            if action == "launch":
                log_path = directory / "supervised-production.log"
                with log_path.open("ab") as output:
                    proc = subprocess.Popen(command(split), cwd=ROOT, stdin=subprocess.DEVNULL,
                                            stdout=output, stderr=subprocess.STDOUT, start_new_session=True,
                                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"})
                at = time.time()
                recent[split].append(at)
                observed = process_identity(proc.pid, command(split))
                append_event(history_path, {"event": "launched", "split": split, "unix_time": at,
                                            "pid": proc.pid, "started_local": observed["started_local"] if observed else None,
                                            "command": command(split), "log": str(log_path.relative_to(ROOT))})
                if observed is None:
                    append_event(history_path, {"event": "early_exit_or_unverified_identity", "split": split,
                                                "pid": proc.pid, "exit_code": proc.poll()})
                    result["state"] = "early_exit_or_unverified_identity"
                else:
                    owned[split], children[split] = observed, proc
                    with (directory / "supervised-caffeinate.log").open("ab") as output:
                        companion = subprocess.Popen(["/usr/bin/caffeinate", "-w", str(proc.pid)],
                                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                                                     stderr=subprocess.STDOUT, start_new_session=True)
                    append_event(history_path, {"event": "caffeinate_started", "split": split,
                                                "pid": companion.pid, "producer_pid": proc.pid})
                result.update(pid=proc.pid, pid_started_local=observed["started_local"] if observed else None,
                              restart_attempts_last_hour=result["restart_attempts_last_hour"] + 1,
                              log=str(log_path.relative_to(ROOT)))
                active_or_waiting = True
            else:
                active_or_waiting |= action not in {"data_frozen_and_audited", "finite_backfill_exhausted"}
            status["splits"][split] = result
        atomic_json(supervisor / "status.json", status)
        if not active_or_waiting:
            append_event(history_path, {"event": "all_splits_terminal", "states": {
                split: status["splits"][split]["state"] for split in SPLITS}})
            return
        time.sleep(args.monitor_seconds)


if __name__ == "__main__":
    main()
