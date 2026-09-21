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

from data_tools.generate import CACHE_VERSION, PARTITION_PATH, PROMPT_VERSION, ROOT
from data_tools.deployment import placement_issue
from data_tools.freeze import publish_bytes
from data_tools.rate_limit import AccountCoordinator
from data_tools.teacher import atomic_json, canonical_bytes, sha256, utc_now
from run_contract import action_quotas, family_quotas, load_run_plan


def production_gate():
    path = ROOT / "local/kimi-account-rate/production-ready.json"
    if not path.is_file():
        return {"ready": False, "reason": "waiting_for_train_only_reasoning_effort_probe"}
    gate = json.loads(path.read_text())
    report = ROOT / gate.get("report_path", "")
    inside = report.resolve().is_relative_to(ROOT.resolve())
    ready = gate.get("ready") is True and gate.get("reasoning_effort") in ("high", "max") and inside and report.is_file() and sha256(report.read_bytes()) == gate.get("report_sha256")
    return {**gate, "ready": ready}


class Progress:
    def __init__(self, split, run_plan=None):
        self.split = split
        self.run_plan = run_plan
        self.audit_cache = {}
        self.episodes = {}

    def read(self):
        episodes, rejections, batch_count = {}, 0, 0
        directory = ROOT / "local/generated" / self.split / ("main-" + CACHE_VERSION)
        if self.run_plan:
            directory /= self.run_plan.run_id
        for path in directory.glob("*.json"):
            if path.name == "failures.json":
                continue
            try:
                value = json.loads(path.read_text())
            except FileNotFoundError:
                continue  # A completed batch atomically replaced its partial.
            batch_count += not path.name.endswith(".partial.json")
            rejections += len(value.get("rejected", []))
            for episode in value.get("episodes", []):
                episodes[episode["id"]] = episode
        teacher_passed = len(episodes)
        self.episodes = {identifier: episode for identifier, episode in episodes.items() if not placement_issue(episode)}
        local_rejections = teacher_passed - len(self.episodes)
        episodes = self.episodes
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
        return {"accepted_including_partial_batches": len(episodes), "teacher_passed_before_local_release_gate": teacher_passed, "local_release_rejections": local_rejections, "completed_batches": batch_count, "rejected_attempt_events": rejections, "labels": dict(labels), "families_with_accepted_data": len({episode["family_id"] for episode in episodes.values()}), "teacher_requests_including_unreleased_attempts": len(self.audit_cache), "teacher_request_statuses": dict(statuses), "teacher_usage_including_unreleased_attempts": dict(usage)}

    def publish_pool(self):
        """Publish already-reviewed slots without waiting for their batch peers."""
        directory = ROOT / "local/data-production" / self.run_plan.run_id if self.run_plan else ROOT / "data"
        output = directory / f"{self.split}.accepted.jsonl"
        manifest_path = output.with_suffix(".manifest.json")
        episodes = sorted(self.episodes.values(), key=lambda episode: episode["id"])
        payload = b"".join(canonical_bytes(episode) + b"\n" for episode in episodes)
        digest = sha256(payload)
        if manifest_path.is_file() and json.loads(manifest_path.read_text()).get("sha256") == digest and output.is_file():
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        labels = Counter(episode["label"]["decision"] if episode["label"]["decision"] == "select" else episode["label"]["abstain_reason"] for episode in episodes)
        atomic_json(manifest_path, {**(self.run_plan.binding() if self.run_plan else {}), "split": self.split, "episodes": len(episodes), "sha256": digest, "prompt_version": PROMPT_VERSION, "labels": dict(labels), "families": dict(Counter(episode["family_id"] for episode in episodes)), "family_action_counts": {family: dict(Counter("select" if row["label"]["decision"] == "select" else "no_match" if row["label"]["abstain_reason"] == "no_match" else "missing_intent" for row in episodes if row["family_id"] == family)) for family in {row["family_id"] for row in episodes}}, "partial_batch_slots_included": True, "created_at": utc_now(), "human_validated": False})
        publish_bytes(output, payload)


def launch(split, workers, run_dir, batch_size, run_plan=None):
    output = (run_dir / f"{split}.log").open("a", buffering=1)
    command = [sys.executable, "-m", "data_tools.generate", "--split", split, "--workers", str(workers), "--batch-size", str(batch_size)]
    if run_plan:
        command += ["--run-plan", str(run_plan.path)]
    output.write(json.dumps({"event": "launch", "time": utc_now(), "command": command, "prompt_version": PROMPT_VERSION}) + "\n")
    process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    return process, output


def freeze_if_ready(split, count, filename, run_dir, run_plan=None, stage=None):
    destination = ROOT / run_plan.data_path(stage) if run_plan else ROOT / "data/frozen" / filename
    if destination.is_file():
        if run_plan:
            frozen_manifest = json.loads(destination.with_suffix(".manifest.json").read_text())
            if any(frozen_manifest.get(key) != value for key, value in run_plan.binding().items()) or frozen_manifest.get("sha256") != sha256(destination.read_bytes()):
                raise ValueError("Existing frozen snapshot differs from this registered run")
        return {"status": "already_frozen", "path": str(destination.relative_to(ROOT))}
    pool_directory = ROOT / "local/data-production" / run_plan.run_id if run_plan else ROOT / "data"
    manifest_path = pool_directory / f"{split}.accepted.manifest.json"
    if not manifest_path.is_file():
        return {"status": "waiting"}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("prompt_version") != PROMPT_VERSION or manifest.get("episodes", 0) < count:
        return {"status": "waiting", "committed_rows": manifest.get("episodes", 0)}
    if run_plan:
        if any(manifest.get(key) != value for key, value in run_plan.binding().items()):
            raise ValueError("Rolling pool binding differs from the registered run")
        quotas = action_quotas(family_quotas(json.loads(PARTITION_PATH.read_text()), split, count))
        if any(manifest.get("family_action_counts", {}).get(family, {}).get(bucket, 0) < amount for family, buckets in quotas.items() for bucket, amount in buckets.items()):
            return {"status": "waiting_for_registered_strata", "committed_rows": manifest["episodes"]}
    elif split == "train" and count == 5000:
        labels = manifest.get("labels", {})
        if labels.get("select", 0) < 3500 or labels.get("no_match", 0) < 1000 or labels.get("ambiguous", 0) + labels.get("insufficient_context", 0) < 500 or len(manifest.get("families", {})) < 40:
            return {"status": "waiting_for_pilot_strata", "committed_rows": manifest["episodes"], "labels": labels}
    command = [sys.executable, "-m", "data_tools.freeze", "--split", split, "--count", str(count), "--source", str((pool_directory / f"{split}.accepted.jsonl").relative_to(ROOT)), "--output", str(destination.relative_to(ROOT))]
    if run_plan:
        command += ["--run-plan", str(run_plan.path), "--stage", stage]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    with (run_dir / "freeze.log").open("a") as stream:
        stream.write(json.dumps({"time": utc_now(), "split": split, "count": count, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}, ensure_ascii=False) + "\n")
    return {"status": "frozen" if result.returncode == 0 else "validation_failed", "path": str(destination.relative_to(ROOT)), "detail": result.stderr[-1000:] if result.returncode else ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-workers", type=int, default=4)
    parser.add_argument("--dev-workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--monitor-seconds", type=float, default=30)
    parser.add_argument("--run-plan")
    args = parser.parse_args()
    run_plan = load_run_plan(args.run_plan) if args.run_plan else None
    if run_plan and run_plan.document["teacher_contract_version"] != PROMPT_VERSION:
        raise SystemExit("Registered teacher contract differs from the generator")
    if not 1 <= args.train_workers <= 8 or not 1 <= args.dev_workers <= 4 or not 1 <= args.batch_size <= 20:
        raise SystemExit("Worker allocation must remain within the measured concurrency limits")
    run_dir = ROOT / ("local/production-" + CACHE_VERSION)
    if run_plan:
        run_dir /= run_plan.run_id
    targets = {split: run_plan.target(split) for split in ("train", "dev")} if run_plan else {"train": 20000, "dev": 1000}
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".supervisor.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A Train/Dev production supervisor is already running")
    atomic_json(run_dir / "process.json", {**(run_plan.binding() if run_plan else {}), "pid": os.getpid(), "started_at": utc_now(), "prompt_version": PROMPT_VERSION, "train_workers": args.train_workers, "dev_workers": args.dev_workers, "batch_size": args.batch_size, "targets": targets})
    workers = {"train": args.train_workers, "dev": args.dev_workers}
    trackers = {split: Progress(split, run_plan) for split in workers}
    coordinator = AccountCoordinator()
    account_status = coordinator.status()
    gate = production_gate()
    children = {split: launch(split, amount, run_dir, args.batch_size, run_plan) if not account_status["paused"] and gate["ready"] else (None, None) for split, amount in workers.items()}
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
        if run_plan:
            run_plan.verify_unchanged()
        elapsed = max(1.0, time.monotonic() - started)
        account_status = coordinator.status()
        gate = production_gate()
        status = {**(run_plan.binding() if run_plan else {}), "updated_at": utc_now(), "prompt_version": PROMPT_VERSION, "elapsed_seconds": round(elapsed, 1), "account_rate_state": account_status, "production_gate": gate, "splits": {}}
        for split, tracker in trackers.items():
            progress = tracker.read()
            tracker.publish_pool()
            gained = progress["accepted_including_partial_batches"] - initial_counts[split]
            rate = gained / elapsed if gained > 0 else 0.0
            remaining = max(0, targets[split] - progress["accepted_including_partial_batches"])
            process, output = children[split]
            code = process.poll() if process is not None else -1
            progress.update(target=targets[split], child_pid=process.pid if process is not None else None, child_exit_code=code, supervisor_restarts=restarts[split], accepted_episodes_per_hour=round(rate * 3600, 2), estimated_remaining_seconds=round(remaining / rate) if rate else None)
            status["splits"][split] = progress
            if code is not None and remaining > 0 and not account_status["paused"] and gate["ready"] and time.monotonic() >= next_restart[split]:
                if output is not None:
                    output.close()
                restarts[split] += 1
                # Accepted slots and attempt counters persist; retries do not
                # reuse a permanently failed prompt or conceal incomplete work.
                next_restart[split] = time.monotonic() + min(900, 30 * (2 ** min(restarts[split], 5)))
                children[split] = launch(split, workers[split], run_dir, args.batch_size, run_plan)
        snapshots = {
            "pilot": freeze_if_ready("train", run_plan.document["pilot_episodes"] if run_plan else 5000, "pilot-train-5000.jsonl", run_dir, run_plan, "pilot"),
            "train": freeze_if_ready("train", targets["train"], "train-20000.jsonl", run_dir, run_plan, "train"),
            "dev": freeze_if_ready("dev", targets["dev"], "dev.jsonl", run_dir, run_plan, "dev"),
        }
        status["snapshots"] = snapshots
        atomic_json(run_dir / "progress.json", status)
        with (run_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(status, ensure_ascii=False) + "\n")
        print(json.dumps({"time": status["updated_at"], "train_accepted": status["splits"]["train"]["accepted_including_partial_batches"], "dev_accepted": status["splits"]["dev"]["accepted_including_partial_batches"], "account_rate_state": account_status, "snapshots": {key: value["status"] for key, value in snapshots.items()}}, ensure_ascii=False), flush=True)
        if all(value["status"] in ("frozen", "already_frozen") for value in snapshots.values()):
            for split in workers:
                command = [sys.executable, "-m", "data_tools.provenance", "--split", split, "--write"]
                if run_plan:
                    command += ["--run-plan", str(run_plan.path)]
                subprocess.run(command, cwd=ROOT, check=True)
            atomic_json(run_dir / "complete.json", {**(run_plan.binding() if run_plan else {}), "completed_at": utc_now(), "snapshots": snapshots, "targets": targets})
            break
        time.sleep(min(60, max(5, args.monitor_seconds)))
    for process, output in children.values():
        if process is not None and process.poll() is None:
            process.terminate()
        if output is not None:
            output.close()


if __name__ == "__main__":
    main()
