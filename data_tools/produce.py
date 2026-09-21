"""Persistently supervise the full Train/Dev quotas and immutable snapshots.

This process never opens Calibration/Test data. Child generators retain accepted
slots, request audits and repair counters. Restarting this supervisor resumes the
same production run instead of shrinking its targets or relabeling old examples.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from data_tools.generate import PARTITION_PATH, PROMPT_VERSION, ROOT
from data_tools.rate_limit import AccountCoordinator
from data_tools.teacher import atomic_json, utc_now


class Progress:
    def __init__(self, split):
        self.split = split
        self.audit_cache = {}

    def read(self):
        episodes, rejections, batch_count = {}, 0, 0
        directory = ROOT / "local/generated" / self.split / "main-v3"
        for path in directory.glob("*.json"):
            if path.name == "failures.json":
                continue
            value = json.loads(path.read_text())
            batch_count += not path.name.endswith(".partial.json")
            rejections += len(value.get("rejected", []))
            for episode in value.get("episodes", []):
                episodes[episode["id"]] = episode
        for path in (ROOT / "local/teacher" / self.split).rglob("*.json"):
            stamp = path.stat().st_mtime_ns
            if path in self.audit_cache and self.audit_cache[path][0] == stamp:
                continue
            value = json.loads(path.read_text())
            response = value.get("response", {})
            self.audit_cache[path] = (stamp, {"status": value.get("status"), "usage": response.get("usage", {}), "phase": value.get("phase"), "attempts": len(value.get("attempts", []))})
        usage = Counter()
        statuses = Counter()
        for _, audit in self.audit_cache.values():
            statuses[audit["status"]] += 1
            usage.update({key: audit["usage"].get(key, 0) for key in ("prompt_tokens", "completion_tokens", "total_tokens")})
        labels = Counter(episode["label"]["decision"] if episode["label"]["decision"] == "select" else episode["label"]["abstain_reason"] for episode in episodes.values())
        return {"accepted_including_partial_batches": len(episodes), "completed_batches": batch_count, "rejected_attempt_events": rejections, "labels": dict(labels), "families_with_accepted_data": len({episode["family_id"] for episode in episodes.values()}), "teacher_requests_including_unreleased_attempts": len(self.audit_cache), "teacher_request_statuses": dict(statuses), "teacher_usage_including_unreleased_attempts": dict(usage)}


def launch(split, workers, run_dir):
    output = (run_dir / f"{split}.log").open("a", buffering=1)
    command = [sys.executable, "-m", "data_tools.generate", "--split", split, "--workers", str(workers)]
    output.write(json.dumps({"event": "launch", "time": utc_now(), "command": command, "prompt_version": PROMPT_VERSION}) + "\n")
    process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    return process, output


def freeze_if_ready(split, count, filename, run_dir):
    destination = ROOT / "data/frozen" / filename
    if destination.is_file():
        return {"status": "already_frozen", "path": str(destination.relative_to(ROOT))}
    manifest_path = ROOT / "data" / f"{split}.manifest.json"
    if not manifest_path.is_file():
        return {"status": "waiting"}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("prompt_version") != PROMPT_VERSION or manifest.get("episodes", 0) < count:
        return {"status": "waiting", "committed_rows": manifest.get("episodes", 0)}
    result = subprocess.run([sys.executable, "-m", "data_tools.freeze", "--split", split, "--count", str(count), "--output", str(destination.relative_to(ROOT))], cwd=ROOT, capture_output=True, text=True)
    with (run_dir / "freeze.log").open("a") as stream:
        stream.write(json.dumps({"time": utc_now(), "split": split, "count": count, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, ensure_ascii=False) + "\n")
    return {"status": "frozen" if result.returncode == 0 else "validation_failed", "path": str(destination.relative_to(ROOT)), "detail": result.stderr[-1000:] if result.returncode else ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--dev-workers", type=int, default=2)
    parser.add_argument("--monitor-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.train_workers <= 8 or not 1 <= args.dev_workers <= 4:
        raise SystemExit("Worker allocation must remain within the measured concurrency limits")
    run_dir = ROOT / "local/production-v3"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A Train/Dev production supervisor is already running")
    atomic_json(run_dir / "process.json", {"pid": os.getpid(), "started_at": utc_now(), "prompt_version": PROMPT_VERSION, "train_workers": args.train_workers, "dev_workers": args.dev_workers, "targets": {"train": 20000, "dev": 1000}})
    workers = {"train": args.train_workers, "dev": args.dev_workers}
    targets = {"train": 20000, "dev": 1000}
    trackers = {split: Progress(split) for split in workers}
    coordinator = AccountCoordinator()
    account_status = coordinator.status()
    children = {split: launch(split, amount, run_dir) if not account_status["paused"] else (None, None) for split, amount in workers.items()}
    restarts = Counter()
    next_restart = {split: 0.0 for split in workers}
    started = time.monotonic()
    initial_counts = {split: trackers[split].read()["accepted_including_partial_batches"] for split in workers}
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for process, _ in children.values():
            if process is not None and process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        elapsed = max(1.0, time.monotonic() - started)
        account_status = coordinator.status()
        status = {"updated_at": utc_now(), "prompt_version": PROMPT_VERSION, "elapsed_seconds": round(elapsed, 1), "account_rate_state": account_status, "splits": {}}
        for split, tracker in trackers.items():
            progress = tracker.read()
            gained = progress["accepted_including_partial_batches"] - initial_counts[split]
            rate = gained / elapsed if gained > 0 else 0.0
            remaining = max(0, targets[split] - progress["accepted_including_partial_batches"])
            process, output = children[split]
            code = process.poll() if process is not None else -1
            progress.update(target=targets[split], child_pid=process.pid if process is not None else None, child_exit_code=code, supervisor_restarts=restarts[split], accepted_episodes_per_hour=round(rate * 3600, 2), estimated_remaining_seconds=round(remaining / rate) if rate else None)
            status["splits"][split] = progress
            if code is not None and remaining > 0 and not account_status["paused"] and time.monotonic() >= next_restart[split]:
                if output is not None:
                    output.close()
                restarts[split] += 1
                # Accepted slots and attempt counters persist; retries do not
                # reuse a permanently failed prompt or conceal incomplete work.
                next_restart[split] = time.monotonic() + min(900, 30 * (2 ** min(restarts[split], 5)))
                children[split] = launch(split, workers[split], run_dir)
        snapshots = {
            "pilot": freeze_if_ready("train", 5000, "pilot-train-5000.jsonl", run_dir),
            "train": freeze_if_ready("train", 20000, "train-20000.jsonl", run_dir),
            "dev": freeze_if_ready("dev", 1000, "dev.jsonl", run_dir),
        }
        status["snapshots"] = snapshots
        atomic_json(run_dir / "progress.json", status)
        with (run_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(status, ensure_ascii=False) + "\n")
        print(json.dumps({"time": status["updated_at"], "train_accepted": status["splits"]["train"]["accepted_including_partial_batches"], "dev_accepted": status["splits"]["dev"]["accepted_including_partial_batches"], "account_rate_state": account_status, "snapshots": {key: value["status"] for key, value in snapshots.items()}}, ensure_ascii=False), flush=True)
        if all(value["status"] in ("frozen", "already_frozen") for value in snapshots.values()):
            for split in workers:
                subprocess.run([sys.executable, "-m", "data_tools.provenance", "--split", split, "--write"], cwd=ROOT, check=True)
            atomic_json(run_dir / "complete.json", {"completed_at": utc_now(), "snapshots": snapshots, "targets": targets})
            break
        time.sleep(min(60, max(5, args.monitor_seconds)))
    for process, output in children.values():
        if process is not None and process.poll() is None:
            process.terminate()
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
