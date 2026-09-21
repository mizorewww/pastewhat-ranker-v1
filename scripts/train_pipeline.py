"""Persistent staged trainer. Only immutable Train/Dev snapshots are observed.

Run under a persistent shell or launchd. Progress/checkpoints survive process
restarts. This script stops at the calibration handoff and never opens Test or
Calibration. Hard-example data must be produced independently after ranker-v0.
"""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from pipeline_data_guards import (freeze_experiment_data, verify_frozen_data,
                                  verify_hardening_mix, verify_pilot_subset)
from pastewhat_ranker.model import sha256_file
from pastewhat_ranker.train import atomic_json, read_allowed_data, verify_completed_run

STATE_PATH = None


def wait_for_snapshot(path, count, state_path, phase, expected_split="train"):
    path = Path(path)
    while not path.exists():
        atomic_json(state_path, {"phase": phase, "status": "waiting_for_frozen_train_dev_data",
                                 "required_path": str(path), "required_count": count, "updated_unix": time.time()})
        time.sleep(30)
    episodes = read_allowed_data(path, expected_split=expected_split)
    if len(episodes) != count:
        raise ValueError(f"Frozen snapshot {path} has {len(episodes)} episodes, expected {count}")
    return sha256_file(path)


def train_stage(stage, template, output, changes, state_path, local):
    output = Path(output)
    summary = output / "training_summary.json"
    config = yaml.safe_load(Path(template).read_text())
    config.update(changes)
    if summary.exists():
        return verify_completed_run(output, config)
    config_path = local / (stage + ".yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_json(state_path, {"phase": stage, "status": "training", "output": str(output),
                             "config": str(config_path), "updated_unix": time.time()})
    command = [sys.executable, "-m", "pastewhat_ranker.train", "--config", str(config_path), "--output", str(output)]
    if (output / "latest" / "progress.json").exists():
        command.append("--resume")
    with (local / (stage + ".log")).open("a", buffering=1) as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(summary.read_text())


def profile_formal_train(train_path, local, state_path):
    report_path = Path("reports/training/pilot-throughput.json")
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if (report.get("training_data_sha256") != sha256_file(train_path)
                or report.get("reference_model_sha256") != sha256_file("checkpoints/initial/model.safetensors")
                or report.get("selection") != "workload_quantiles"):
            raise ValueError("Formal throughput report does not match the frozen training data/model")
        return report["selected_micro_batch"]
    atomic_json(state_path, {"phase": "pilot_throughput", "status": "measuring",
                             "selection": "Train workload quantiles only", "updated_unix": time.time()})
    with (local / "pilot-throughput.log").open("a", buffering=1) as log:
        subprocess.run([sys.executable, "scripts/measure_training_throughput.py", "--train", train_path,
                        "--selection", "workload_quantiles", "--output", str(report_path)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(report_path.read_text())["selected_micro_batch"]


def main():
    global STATE_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-state", default="local/pipeline")
    args = parser.parse_args()
    local = Path(args.local_state)
    local.mkdir(parents=True, exist_ok=True)
    run_lock = (local / "pipeline.lock").open("w")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (local / "pipeline.pid").write_text(str(os.getpid()) + "\n")
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        watcher = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (local / "caffeinate.pid").write_text(str(watcher.pid) + "\n")
    state_path = local / "status.json"
    STATE_PATH = state_path
    wait_for_snapshot("data/frozen/overfit-train-32.jsonl", 32, state_path, "overfit_preparation")
    throughput_path = Path("reports/training/throughput.json")
    if not throughput_path.exists():
        atomic_json(state_path, {"phase": "training_throughput", "status": "measuring", "updated_unix": time.time()})
        with (local / "throughput.log").open("a", buffering=1) as log:
            subprocess.run([sys.executable, "scripts/measure_training_throughput.py", "--train",
                            "data/frozen/overfit-train-32.jsonl", "--output", str(throughput_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    micro = json.loads(throughput_path.read_text())["selected_micro_batch"]
    overfit = train_stage("overfit", "configs/overfit.yaml", "checkpoints/overfit", {"micro_batch_episodes": micro}, state_path, local)
    overfit_metrics = json.loads(Path(overfit["best_checkpoint"], "dev_metrics.json").read_text())
    if overfit_metrics["decision_accuracy"] < 0.99:
        raise RuntimeError("32-episode overfit gate did not reach 99%; diagnose training before scaling")
    wait_for_snapshot("data/frozen/pilot-train-5000.jsonl", 5000, state_path, "pilot_preparation")
    micro = profile_formal_train("data/frozen/pilot-train-5000.jsonl", local, state_path)
    wait_for_snapshot("data/frozen/dev.jsonl", 1000, state_path, "pilot_preparation", expected_split="dev")
    pilot = train_stage("pilot", "configs/pilot.yaml", "checkpoints/pilot", {"micro_batch_episodes": micro}, state_path, local)
    pilot_metrics = json.loads(Path(pilot["best_checkpoint"], "dev_metrics.json").read_text())
    if pilot_metrics["coverage"] == 0 or pilot_metrics["answerable_top1"] == 0:
        raise RuntimeError("Pilot learned no usable candidate selections; diagnose before scaling")
    wait_for_snapshot("data/frozen/train-20000.jsonl", 20000, state_path, "main_preparation")
    experiment_data = local / "frozen-experiment-data.json"
    freeze_experiment_data(experiment_data, {"pilot_train": "data/frozen/pilot-train-5000.jsonl",
                                            "main_train": "data/frozen/train-20000.jsonl",
                                            "dev": "data/frozen/dev.jsonl"})
    atomic_json(local / "pilot-subset-verification.json",
                verify_pilot_subset("data/frozen/pilot-train-5000.jsonl", "data/frozen/train-20000.jsonl"))
    seed_report = Path("reports/training/seed-initializations.json")
    initialization_command = [sys.executable, "scripts/freeze_seed_initializations.py", "--output", str(seed_report)]
    local_source = Path("../laya-mlx/models/laya-multilingual")
    if (local_source / "model.safetensors").exists():
        initialization_command += ["--source", str(local_source)]
    subprocess.run(initialization_command, check=True)
    initializations = {row["seed"]: row for row in json.loads(seed_report.read_text())["seeds"]}
    runs = []
    for seed in (42, 43, 44):
        verify_frozen_data(experiment_data)
        run = train_stage(f"main-seed-{seed}", "configs/main.yaml", f"checkpoints/main-seed-{seed}",
                          {"seed": seed, "initial_model": initializations[seed]["initial_model"],
                           "micro_batch_episodes": micro}, state_path, local)
        runs.append({"seed": seed, **run})
    selected = max(runs, key=lambda run: run["best_dev_key"])
    v0 = Path("checkpoints/ranker-v0-selected")
    shutil.copytree(selected["best_checkpoint"], v0, dirs_exist_ok=True)
    atomic_json(local / "ranker-v0-ready.json", {"checkpoint": str(v0), "selected_seed": selected["seed"],
                                                "selected_by": "Dev only", "runs": runs,
                                                "weight_sha256": sha256_file(v0 / "model.safetensors")})
    # Teacher/data agent mines only a new training pool, not final Test failures.
    wait_for_snapshot("data/frozen/hardening-train-10000.jsonl", 10000, state_path, "hardening_preparation")
    verify_frozen_data(experiment_data)
    atomic_json(local / "hardening-data-mixture.json",
                verify_hardening_mix("data/frozen/train-20000.jsonl", "data/frozen/hardening-train-10000.jsonl"))
    hardened = train_stage("hardening", "configs/hardening.yaml", "checkpoints/hardening",
                           {"micro_batch_episodes": micro, "seed": selected["seed"]}, state_path, local)
    previous = json.loads((v0 / "dev_metrics.json").read_text())
    current = json.loads(Path(hardened["best_checkpoint"], "dev_metrics.json").read_text())
    group_regressions = []
    for name, old in previous.get("groups", {}).items():
        new = current.get("groups", {}).get(name)
        if old["episodes"] >= 20 and new and new["decision_accuracy"] < old["decision_accuracy"] - .05:
            group_regressions.append(name)
    accept = current["selection_metric"] > previous["selection_metric"] and not group_regressions
    release = Path("checkpoints/ranker-v1-candidate")
    shutil.copytree(hardened["best_checkpoint"] if accept else v0, release, dirs_exist_ok=True)
    atomic_json(local / "hardening-selection.json", {"accepted": accept, "group_regressions": group_regressions,
                                                    "v0_dev": previous, "hardening_dev": current,
                                                    "selection_rule": "Dev balanced accuracy improves; no >5pp regression in group n>=20"})
    subprocess.run([sys.executable, "-m", "pastewhat_ranker.export", "--model", str(release), "--output", str(release / "mlx")], check=True)
    atomic_json(local / "ready-for-calibration.json", {"reference_checkpoint": str(release),
                                                     "deployment_checkpoint": str(release / "mlx"),
                                                     "weight_sha256": sha256_file(release / "mlx" / "model.safetensors"),
                                                     "selected_seed": selected["seed"], "hardening_accepted": accept,
                                                     "status": "conversion_requires_independent_parity_and_calibration"})
    atomic_json(state_path, {"phase": "calibration_handoff", "status": "training_stages_complete",
                             "updated_unix": time.time()})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if STATE_PATH is not None:
            previous = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
            atomic_json(STATE_PATH, {**previous, "status": "failed_requires_diagnosis",
                                     "error": f"{type(error).__name__}: {error}",
                                     "updated_unix": time.time()})
        raise
