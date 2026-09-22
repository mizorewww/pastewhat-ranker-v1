"""Run one registered login job, skipping verified completed handoffs."""
from __future__ import annotations

import argparse
import json
import subprocess

from run_contract import load_run_plan
from tools.install_login_recovery import PLAN, ROOT, jobs


def completed(job: str) -> bool:
    plan = load_run_plan(PLAN)
    paths = {
        "training": ROOT / "local/pipeline" / plan.run_id / "ready-for-calibration.json",
        "evaluation": ROOT / "local/evaluator-release" / plan.run_id / "ready-for-publication.json",
        "publication": ROOT / "local/publication" / plan.run_id / "completed.json",
        "monitor": ROOT / "local/publication" / plan.run_id / "completed.json",
    }
    path = paths.get(job)
    if path is None or not path.is_file():
        return False
    record = json.loads(path.read_text())
    if any(record.get(key) != expected for key, expected in plan.binding().items()):
        raise ValueError("Login recovery completion belongs to another run")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=jobs())
    args = parser.parse_args()
    if completed(args.job):
        print(args.job + ": already completed")
        return
    command = jobs()[args.job][0]
    raise SystemExit(subprocess.run(command, cwd=ROOT, check=False).returncode)


if __name__ == "__main__":
    main()
