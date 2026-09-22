"""Install user LaunchAgents that resume the registered run after macOS login.

Installation does not bootstrap the agents into the current login session. The
already-running producers and waiters therefore continue untouched. At the
next login, each job starts once; its own run lock and durable state decide
what remains to be done. Normal completion does not cause a restart loop.
"""
from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path

from run_contract import load_run_plan

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/run_plan_efficient.json"
EXPECTED_RUN_ID = "ranker-v1-efficient-20260921"
EXPECTED_PLAN_SHA256 = "e5491476a01f3cd3d1b3f3778ac90f4e1dbeadca0e001a078003633266b7963f"
PYTHON = ROOT / ".venv/bin/python"
LABEL_PREFIX = "com.mizore.pastewhat.ranker-v1"


def jobs():
    plan = load_run_plan(PLAN)
    if plan.run_id != EXPECTED_RUN_ID or plan.sha256 != EXPECTED_PLAN_SHA256:
        raise ValueError("Login recovery is bound to the registered run")
    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)
    common = ["--run-plan", str(PLAN)]
    return {
        "training-data": ([str(PYTHON), "-m", "data_tools.produce_v7", *common], "local/v7/" + plan.run_id),
        "heldout-data": ([str(PYTHON), "-m", "evaluations.supervise_v7"], "local/evaluator-v7/" + plan.run_id + "/supervisor-v7"),
        "training": ([str(PYTHON), "-u", "scripts/train_pipeline.py", *common,
                      "--throughput-snapshot", "data/frozen/" + plan.run_id + "/throughput-train.jsonl"],
                     "local/pipeline/" + plan.run_id),
        "evaluation": ([str(PYTHON), "-u", "-m", "evaluations.release_pipeline", *common],
                       "local/evaluator-release/" + plan.run_id),
        "publication": ([str(PYTHON), "-u", "-m", "tools.publish_pipeline", *common],
                        "local/publication/" + plan.run_id),
        "monitor": ([str(PYTHON), "-u", "-m", "tools.production_watch", *common],
                    "local/monitor/" + plan.run_id),
    }


def main():
    target = Path.home() / "Library/LaunchAgents"
    target.mkdir(parents=True, exist_ok=True)
    for name, (command, log_directory) in jobs().items():
        logs = ROOT / log_directory
        logs.mkdir(parents=True, exist_ok=True)
        label = LABEL_PREFIX + "." + name
        destination = target / (label + ".plist")
        document = {
            "Label": label,
            "ProgramArguments": command,
            "WorkingDirectory": str(ROOT),
            "EnvironmentVariables": {
                "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            "RunAtLoad": True,
            "KeepAlive": False,
            "StandardOutPath": str(logs / "login-recovery.log"),
            "StandardErrorPath": str(logs / "login-recovery.log"),
        }
        payload = plistlib.dumps(document, sort_keys=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        subprocess.run(["plutil", "-lint", str(destination)], check=True, capture_output=True)
        print(destination)


if __name__ == "__main__":
    main()
