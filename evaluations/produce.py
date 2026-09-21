"""Durable Calibration/Test generation supervisor; never runs student Test.

Accepted slots and rejected attempts stay on disk. This process only synthesizes
and audits data. It cannot fit a calibrator, select a checkpoint, score Test, or
publish a model. Actual provider cooldowns are enforced by the shared client.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from data_tools.teacher import atomic_json, utc_now
from data_tools.rate_limit import AccountCoordinator
from evaluations.generate import passed_current_gates

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "local/evaluator-generation-v4"
HELDOUT = ROOT / "local/evaluator-heldout-v4"


def progress(split: str) -> dict:
    episodes, rejected, unverified = {}, 0, 0
    for path in (STATE / split).glob("*.json"):
        # One-row pipeline probes are not part of the release allocation.
        if path.name.endswith("-0000-1.json"):
            continue
        state = json.loads(path.read_text())
        rejected += len(state.get("rejected_attempts", []))
        for episode in state.get("episodes", []):
            verified = passed_current_gates(episode)
            if verified:
                episodes[episode["id"]] = episode
            else:
                unverified += 1
    return {"accepted_with_all_current_gates": len(episodes), "legacy_or_unverified_rows": unverified,
            "label_counts": dict(Counter(row["label"]["decision"] if row["label"]["decision"] == "select" else row["label"]["abstain_reason"] for row in episodes.values())),
            "families_with_accepted_rows": len({row["family_id"] for row in episodes.values()}),
            "rejected_attempt_events": rejected}


def launch(split: str, workers: int, directory: Path):
    log = (directory / (split + ".log")).open("a", buffering=1)
    command = [sys.executable, "-m", "evaluations.generate", "--split", split,
               "--batch-size", "6", "--workers", str(workers), "--output", str(HELDOUT / (split + ".jsonl"))]
    log.write(json.dumps({"event": "launch", "time": utc_now(), "command": command}) + "\n")
    return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT), log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-per-split", type=int, default=2)
    parser.add_argument("--monitor-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.workers_per_split <= 2:
        raise SystemExit("Evaluator HTTP worker allocation must not exceed the coordinated two per split")
    directory = ROOT / "local/evaluator-production-v4"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("An evaluator production supervisor is already running")
    splits = {"calibration": 1000, "test": 2000}
    atomic_json(directory / "process.json", {"pid": os.getpid(), "started_at": utc_now(),
                "workers_per_split": args.workers_per_split, "targets": splits,
                "student_test_inference_allowed": False})
    coordinator = AccountCoordinator()
    children = {}
    restarts, next_restart, audits = Counter(), {split: 0.0 for split in splits}, {}
    started = time.monotonic()
    initial = {split: progress(split)["accepted_with_all_current_gates"] for split in splits}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for child, _ in children.values():
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        elapsed = max(1.0, time.monotonic() - started)
        account = coordinator.status()
        status = {"updated_at": utc_now(), "elapsed_seconds": round(elapsed, 1), "splits": {},
                  "student_test_inference_allowed": False, "provider_account": account}
        for split, target in splits.items():
            current = progress(split)
            child, log = children.get(split, (None, None))
            exit_code = child.poll() if child else None
            gained = current["accepted_with_all_current_gates"] - initial[split]
            rate = gained / elapsed if gained > 0 else 0.0
            data = HELDOUT / (split + ".jsonl")
            current.update(target=target, child_pid=child.pid if child else None, child_exit_code=exit_code,
                           supervisor_restarts=restarts[split], accepted_per_hour=round(rate * 3600, 2),
                           estimated_remaining_seconds=round((target - current["accepted_with_all_current_gates"]) / rate) if rate else None)
            if data.is_file():
                report = ROOT / "reports" / ("data-" + split + "-audit.json")
                if split not in audits:
                    if report.is_file():
                        audits[split] = {"status": "passed" if json.loads(report.read_text()).get("passed") else "failed", "report": str(report.relative_to(ROOT))}
                    else:
                        result = subprocess.run([sys.executable, "-m", "evaluations.audit_data", "--data", str(data),
                                                 "--output", str(report)], cwd=ROOT, capture_output=True, text=True)
                        with (directory / "audit.log").open("a") as handle:
                            handle.write(json.dumps({"split": split, "time": utc_now(), "exit_code": result.returncode,
                                                     "stdout": result.stdout, "stderr": result.stderr}, ensure_ascii=False) + "\n")
                        audits[split] = {"status": "passed" if result.returncode == 0 else "failed_requires_evaluator", "report": str(report.relative_to(ROOT))}
                current["data_status"] = "frozen"
                current["audit"] = audits[split]
            elif child is None or exit_code is not None:
                if current["accepted_with_all_current_gates"] >= target:
                    current["data_status"] = "freeze_validation_failed_requires_evaluator"
                elif account["paused"]:
                    current["data_status"] = "provider_paused_until_external_change" if account["blocked_until_external_change"] else "provider_cooldown"
                elif time.monotonic() >= next_restart[split]:
                    if log:
                        log.close()
                        restarts[split] += 1
                    next_restart[split] = time.monotonic() + min(900, 30 * 2 ** min(restarts[split], 5))
                    children[split] = launch(split, args.workers_per_split, directory)
                    current["data_status"] = "resuming_unfilled_slots"
            else:
                current["data_status"] = "generating_or_waiting_for_shared_provider_cooldown"
            status["splits"][split] = current
        atomic_json(directory / "progress.json", status)
        with (directory / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(status, ensure_ascii=False) + "\n")
        print(json.dumps({"updated_at": status["updated_at"], **{split: value["accepted_with_all_current_gates"] for split, value in status["splits"].items()},
                          "audits": audits}, ensure_ascii=False), flush=True)
        if len(audits) == 2 and all(value["status"] == "passed" for value in audits.values()):
            atomic_json(directory / "complete.json", {"completed_at": utc_now(), "targets": splits,
                        "audits": audits, "student_test_inference_run": False})
            break
        time.sleep(min(60, max(5, args.monitor_seconds)))
    for child, log in children.values():
        if child.poll() is None:
            child.terminate()
        log.close()


if __name__ == "__main__":
    main()
