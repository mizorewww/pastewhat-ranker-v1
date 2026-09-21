"""Monitor the authorized v7 Train/Dev/hardening chain without hot quota restarts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from data_tools.rate_limit import AccountCoordinator
from data_tools.teacher import atomic_json, utc_now
from data_tools.v7 import ROOT
from run_contract import load_run_plan


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-plan", required=True)
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    base = ROOT / "local/v7" / plan.run_id
    base.mkdir(parents=True, exist_ok=True)
    lock = (base / ".production-supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = {
        "train": [sys.executable, "-m", "data_tools.generate_v7", "--run-plan", str(plan.path), "--split", "train", "--workers", "3"],
        "dev": [sys.executable, "-m", "data_tools.generate_v7", "--run-plan", str(plan.path), "--split", "dev", "--workers", "1"],
        "hardening": [sys.executable, "-m", "data_tools.hardening_v7", "--run-plan", str(plan.path), "--workers", "2"],
    }
    processes = {}
    children = {}
    for name in jobs:
        launch = base / f"{name}.bulk-launch.json"
        if launch.exists():
            processes[name] = json.loads(launch.read_text())
    coordinator = AccountCoordinator()
    while True:
        plan.verify_unchanged()
        account = coordinator.status()
        states = {}
        for name, command in jobs.items():
            if (ROOT / plan.data_path(name)).exists():
                states[name] = {"state": "frozen"}
                continue
            process = processes.get(name, {})
            if name in children:
                children[name].poll()
            if alive(process.get("pid")):
                states[name] = {"state": "running", "pid": process["pid"]}
                continue
            completion_path = base / f"{name}.run-completion.json"
            completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
            hard_insufficient = base / "hardening/insufficient-confirmed-new.json"
            if completion.get("status") == "finite_backfill_exhausted" or name == "hardening" and hard_insufficient.exists():
                states[name] = {"state": "finite_source_budget_exhausted", "note": "Keep actual deficits visible; no label rewriting or repeated preferred-answer search"}
                continue
            if account["paused"]:
                states[name] = {"state": "waiting_for_account", "reason": account["reason"]}
                continue
            if time.time() - process.get("started_epoch", 0) < 60:
                states[name] = {"state": "restart_backoff"}
                continue
            log = (base / f"{name}.bulk.log").open("ab")
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
            process = {"pid": child.pid, "started_epoch": time.time(), "launched_at": utc_now(), "command": command}
            processes[name] = process
            children[name] = child
            atomic_json(base / f"{name}.bulk-launch.json", process)
            states[name] = {"state": "running", "pid": child.pid}
        atomic_json(base / "production-status.json", {**plan.binding(), "updated_at": utc_now(), "account": account, "jobs": states})
        if all(state["state"] == "frozen" for state in states.values()):
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
