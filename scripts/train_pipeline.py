"""Persistent staged trainer. Only immutable Train/Dev snapshots are observed.

Run under a persistent shell or launchd. Progress/checkpoints survive process
restarts. This script stops at the calibration handoff and never opens Test or
Calibration. Hard-example data must be produced independently after ranker-v0.
"""

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

# Direct script execution puts scripts/, not the repository root, on sys.path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_contract import load_run_plan
from pipeline_data_guards import (freeze_experiment_data, freeze_run_plan, verify_binding,
                                  verify_frozen_data, verify_hardening_mix, verify_pilot_subset,
                                  verify_run_plan, verify_snapshot_binding)
from pastewhat_ranker.model import sha256_file
from pastewhat_ranker.train import atomic_json, read_allowed_data, verify_completed_run

STATE_PATH = None


def write_record(path, values, run_plan=None):
    verify_run_plan(run_plan)
    atomic_json(path, {**values, **(run_plan.binding() if run_plan else {})})


def stage_layout(run_plan=None, local_state=None):
    if run_plan is None:
        return {"local": Path(local_state or "local/pipeline"),
                "checkpoints": Path("checkpoints"), "reports": Path("reports/training"),
                "paths": {"pilot": Path("data/frozen/pilot-train-5000.jsonl"),
                          "train": Path("data/frozen/train-20000.jsonl"),
                          "dev": Path("data/frozen/dev.jsonl"),
                          "hardening": Path("data/frozen/hardening-train-10000.jsonl")},
                "counts": {"pilot": 5000, "train": 20000, "dev": 1000, "hardening": 10000}}
    if local_state is not None and Path(local_state).resolve() != run_plan.pipeline_directory.resolve():
        raise ValueError("A registered run must use its own pipeline state directory")
    document = run_plan.document
    return {"local": run_plan.pipeline_directory, "checkpoints": run_plan.checkpoint_directory,
            "reports": run_plan.report_directory,
            "paths": {stage: run_plan.data_path(stage) for stage in ("pilot", "train", "dev", "hardening")},
            "counts": {"pilot": document["pilot_episodes"], "train": run_plan.target("train"),
                       "dev": run_plan.target("dev"),
                       "hardening": document["hardening"]["accepted_new"] + document["hardening"]["retained_original"]}}


def formal_stage_changes(stage, micro, *, seed=42, initial_model=None, run_plan=None):
    """Generate formal overrides; the engineering configuration never uses this."""
    if stage not in ("pilot", "main", "hardening") or (stage == "main" and initial_model is None):
        raise ValueError("A formal stage needs its declared role and original main initialization")
    changes = {"micro_batch_episodes": micro}
    if stage != "pilot":
        changes["seed"] = seed
    if stage == "main":
        changes["initial_model"] = str(initial_model)
    if run_plan is None:
        return changes
    document = run_plan.document
    data_stage = "train" if stage == "main" else stage
    count = stage_layout(run_plan)["counts"][data_stage]
    model = (initial_model if initial_model is not None else
             run_plan.checkpoint_directory / "ranker-v0-selected" if stage == "hardening" else "checkpoints/initial")
    changes.update({**run_plan.binding(), "initial_model": str(model), "seed": seed,
                    "train_data": str(run_plan.data_path(data_stage)), "train_limit": count,
                    "dev_data": str(run_plan.data_path("dev")), "epochs": document["epochs"][stage],
                    "head_warmup_steps": 0 if stage == "hardening" else document["head_warmup_steps"],
                    "effective_batch_episodes": document["effective_batch_episodes"]})
    return changes


def wait_for_snapshot(path, count, state_path, phase, expected_split="train", *, run_plan=None):
    path = Path(path)
    while not path.exists():
        write_record(state_path, {"phase": phase, "status": "waiting_for_frozen_train_dev_data",
                                  "required_path": str(path), "required_count": count, "updated_unix": time.time()}, run_plan)
        time.sleep(30)
    verify_snapshot_binding(path, count, expected_split, run_plan)
    episodes = read_allowed_data(path, expected_split=expected_split)
    if len(episodes) != count:
        raise ValueError(f"Frozen snapshot {path} has {len(episodes)} episodes, expected {count}")
    return sha256_file(path)


def train_stage(stage, template, output, changes, state_path, local, *, run_plan=None):
    verify_run_plan(run_plan)
    output = Path(output)
    summary = output / "training_summary.json"
    config = yaml.safe_load(Path(template).read_text())
    config.update(changes)
    if stage != "overfit":
        verify_binding(config, run_plan, "Formal training configuration")
    if summary.exists():
        result = verify_completed_run(output, config)
        verify_run_plan(run_plan)
        return result
    config_path = local / (stage + ".yaml")
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    write_record(state_path, {"phase": stage, "status": "training", "output": str(output),
                              "config": str(config_path), "updated_unix": time.time()}, run_plan)
    command = [sys.executable, "-m", "pastewhat_ranker.train", "--config", str(config_path), "--output", str(output)]
    if (output / "latest" / "progress.json").exists():
        command.append("--resume")
    with (local / (stage + ".log")).open("a", buffering=1) as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    verify_run_plan(run_plan)
    return json.loads(summary.read_text())


def profile_formal_train(train_path, local, state_path, *, run_plan=None):
    verify_run_plan(run_plan)
    report_path = stage_layout(run_plan)["reports"] / "pilot-throughput.json"

    def validated(report):
        if (report.get("training_data_sha256") != sha256_file(train_path)
                or report.get("reference_model_sha256") != sha256_file("checkpoints/initial/model.safetensors")
                or report.get("selection") != "workload_quantiles"
                or report.get("selected_micro_batch") not in (1, 2, 4)):
            raise ValueError("Formal throughput report does not match the frozen training data/model")
        return report["selected_micro_batch"]

    if report_path.exists():
        report = json.loads(report_path.read_text())
        verify_binding(report, run_plan, "Formal throughput report")
        return validated(report)
    # A completed raw measurement can survive an interruption before publication.
    # It stays under already-bound run state until the final report is atomic.
    raw_path = local / "pilot-throughput-result.json" if run_plan else report_path
    if not raw_path.exists():
        write_record(state_path, {"phase": "pilot_throughput", "status": "measuring",
                                  "selection": "Train workload quantiles only", "updated_unix": time.time()}, run_plan)
        with (local / "pilot-throughput.log").open("a", buffering=1) as log:
            subprocess.run([sys.executable, "scripts/measure_training_throughput.py", "--train", str(train_path),
                            "--selection", "workload_quantiles", "--output", str(raw_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    report = json.loads(raw_path.read_text())
    micro = validated(report)
    if run_plan:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_record(report_path, report, run_plan)
    return micro


def seed_initializations(run_plan=None):
    """Reuse frozen untrained snapshots; no run starts from pilot/overfit weights."""
    source_report = Path("reports/training/seed-initializations.json")
    if run_plan is None or not source_report.exists():
        command = [sys.executable, "scripts/freeze_seed_initializations.py", "--output", str(source_report)]
        local_source = Path("../laya-mlx/models/laya-multilingual")
        if (local_source / "model.safetensors").exists():
            command += ["--source", str(local_source)]
        subprocess.run(command, check=True)
    report = json.loads(source_report.read_text())
    records = {row["seed"]: row for row in report["seeds"]}
    seeds = run_plan.document["training_seeds"] if run_plan else [42, 43, 44]
    if report.get("status") != "passed" or set(records) != set(seeds):
        raise ValueError("The frozen initialization report does not cover the registered seeds")
    for seed, row in records.items():
        expected = Path("checkpoints/initial" if seed == 42 else f"checkpoints/initial-seed-{seed}")
        if (Path(row["initial_model"]).resolve() != expected.resolve()
                or row.get("initialization_seed") != seed or row.get("encoder_tensor_equality_to_seed42") is not True
                or row["model_sha256"] != sha256_file(expected / "model.safetensors")):
            raise ValueError("A frozen original-encoder seed initialization changed")
    if run_plan:
        path = run_plan.report_directory / source_report.name
        bound = {**report, **run_plan.binding(), "source_report_sha256": sha256_file(source_report)}
        if path.exists() and json.loads(path.read_text()) != bound:
            raise ValueError("This run's original initialization binding changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        write_record(path, bound, run_plan)
    return records


def artifact_record(path):
    """A small aggregate evidence file, never an embedded training example."""
    path = Path(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def completed_stage_evidence(output, expected_config, train_count, dev_count, *, run_plan=None):
    """Require the whole registered schedule, not merely a reusable checkpoint."""
    output = Path(output)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("config") != expected_config or expected_config.get("engineering_overfit")
            or expected_config.get("max_steps")
            or manifest.get("train_episodes") != train_count or manifest.get("dev_episodes") != dev_count):
        raise ValueError("Training completion evidence differs from the full registered stage")
    verify_binding(expected_config, run_plan, "Training completion configuration")
    summary = verify_completed_run(output, expected_config)
    effective = expected_config["effective_batch_episodes"]
    epochs = expected_config["epochs"]
    warmup = expected_config["head_warmup_steps"]
    updates_per_epoch = math.ceil(train_count / effective)
    expected_steps = warmup + updates_per_epoch * epochs
    # Warmup cycles the complete dataset, including each final short batch.
    cycles, remaining = divmod(warmup, updates_per_epoch)
    expected_seen = cycles * train_count + min(train_count, remaining * effective) + train_count * epochs
    if (summary.get("engineering_overfit") is not False
            or summary.get("global_steps") != expected_steps
            or summary.get("seen_episodes_including_head_warmup") != expected_seen
            or Path(summary["manifest"]).resolve() != manifest_path.resolve()
            or Path(summary["best_checkpoint"]).resolve() != (output / "best").resolve()):
        raise ValueError("Training completion summary does not prove every registered optimizer step")
    return {"status": "completed", "summary": artifact_record(output / "training_summary.json"),
            "run_manifest": artifact_record(manifest_path),
            "train_config": artifact_record(output / "train_config.yaml"),
            "best_checkpoint": summary["best_checkpoint"], "best_weight_sha256": summary["best_weight_sha256"],
            "best_dev_key": summary["best_dev_key"], "seed": expected_config["seed"],
            "train_episodes": train_count, "dev_episodes": dev_count, "epochs": epochs,
            "head_warmup_steps": warmup, "effective_batch_episodes": effective,
            "global_steps": summary["global_steps"], "seen_episodes_including_head_warmup": expected_seen,
            "initial_weight_sha256": manifest["initial_weight_sha256"]}


def training_handoff(layout, micro, initializations, selected_seed, hardening_accepted, *, run_plan=None):
    """Public aggregate proof of pilot, every main seed, and executed hardening."""
    verify_run_plan(run_plan)
    local, checkpoints, counts = layout["local"], layout["checkpoints"], layout["counts"]
    seeds = list(run_plan.document["training_seeds"] if run_plan else (42, 43, 44))
    stages = {}
    for name, role, seed, initial in [("pilot", "pilot", 42, None),
                                    *[(f"main_seed_{seed}", "main", seed, initializations[seed]["initial_model"]) for seed in seeds],
                                    ("hardening", "hardening", selected_seed, None)]:
        config = yaml.safe_load(Path(f"configs/{role}.yaml").read_text())
        config.update(formal_stage_changes(role, micro, seed=seed, initial_model=initial, run_plan=run_plan))
        output = checkpoints / (f"main-seed-{seed}" if role == "main" else name)
        stages[name] = completed_stage_evidence(output, config, counts["train" if role == "main" else role],
                                                counts["dev"], run_plan=run_plan)
    winner = max(seeds, key=lambda seed: stages[f"main_seed_{seed}"]["best_dev_key"])
    if winner != selected_seed:
        raise ValueError("The selected seed is not the completed main run chosen by Dev")
    evidence_paths = {"experiment_data": local / "frozen-experiment-data.json",
                      "pilot_subset": local / "pilot-subset-verification.json",
                      "seed_initializations": layout["reports"] / "seed-initializations.json",
                      "v0_selection": local / "ranker-v0-ready.json",
                      "hardening_mix": local / "hardening-data-mixture.json",
                      "hardening_selection": local / "hardening-selection.json"}
    if run_plan:
        evidence_paths.update(pilot_data=local / "frozen-pilot-data.json",
                              hardening_data=local / "frozen-hardening-data.json",
                              run_plan_binding=local / "run-plan-binding.json")
    records = {name: json.loads(path.read_text()) for name, path in evidence_paths.items()}
    for name, record in records.items():
        verify_binding(record, run_plan, "Training evidence " + name)
    for name in ("experiment_data", "pilot_data", "hardening_data"):
        if name in evidence_paths:
            verify_frozen_data(evidence_paths[name], run_plan=run_plan)
    subset, mixture, selection = records["pilot_subset"], records["hardening_mix"], records["hardening_selection"]
    expected_mix = run_plan.document["hardening"] if run_plan else {"accepted_new": 5000, "retained_original": 5000}
    if (subset.get("pilot_is_unchanged_subset") is not True or subset.get("pilot_episodes") != counts["pilot"]
            or subset.get("main_episodes") != counts["train"]
            or mixture.get("originals_preserved_exactly") is not True
            or mixture.get("all_families_in_train_partition") is not True
            or mixture.get("original_episodes") != expected_mix["retained_original"]
            or mixture.get("new_episode_ids") != expected_mix["accepted_new"]
            or records["v0_selection"].get("selected_by") != "Dev only"
            or records["v0_selection"].get("selected_seed") != selected_seed
            or selection.get("accepted") is not hardening_accepted
            or selection.get("selected_by") != "Dev only" or selection.get("hardening_executed") is not True):
        raise ValueError("Aggregate stage evidence does not prove the registered data or Dev selection")
    selected_main_sha = stages[f"main_seed_{selected_seed}"]["best_weight_sha256"]
    if (records["v0_selection"].get("weight_sha256") != selected_main_sha
            or stages["hardening"]["initial_weight_sha256"] != selected_main_sha):
        raise ValueError("Hardening did not start from the Dev-selected main checkpoint")
    for seed in seeds:
        if stages[f"main_seed_{seed}"]["initial_weight_sha256"] != initializations[seed]["model_sha256"]:
            raise ValueError("A main run did not restart from its frozen original-encoder initialization")
    if stages["pilot"]["initial_weight_sha256"] != initializations[42]["model_sha256"]:
        raise ValueError("Pilot did not start from the frozen original-encoder initialization")
    release = checkpoints / "ranker-v1-candidate"
    reference_sha = sha256_file(release / "model.safetensors")
    deployment_sha = sha256_file(release / "mlx" / "model.safetensors")
    expected_sha = stages["hardening"]["best_weight_sha256"] if hardening_accepted else selected_main_sha
    conversion_path = release / "mlx" / "conversion.json"
    conversion = json.loads(conversion_path.read_text())
    if (reference_sha != expected_sha or conversion.get("reference_weight_sha256") != reference_sha
            or conversion.get("mlx_weight_sha256") != deployment_sha or conversion.get("strict_parameter_load") is not True):
        raise ValueError("Release weights do not match the completed Dev selection and strict MLX export")
    evidence_paths["conversion"] = conversion_path
    return {"version": "pastewhat-training-handoff-v2", "reference_checkpoint": str(release),
            "deployment_checkpoint": str(release / "mlx"), "reference_weight_sha256": reference_sha,
            "deployment_weight_sha256": deployment_sha, "weight_sha256": deployment_sha,
            "selected_by": "Dev only", "registered_seeds": seeds, "completed_seeds": seeds,
            "selected_seed": selected_seed, "hardening_executed": True, "hardening_accepted": hardening_accepted,
            "training_completion": {"status": "completed", "data_counts": counts, "stages": stages,
                                    "evidence": {name: artifact_record(path) for name, path in evidence_paths.items()}},
            "status": "conversion_requires_independent_parity_and_calibration"}


def main():
    global STATE_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-state", help="Legacy state directory; planned runs use their registered run directory")
    parser.add_argument("--run-plan", type=Path, help="Immutable registered production plan; omission preserves the original route")
    args = parser.parse_args()
    run_plan = load_run_plan(args.run_plan) if args.run_plan else None
    layout = stage_layout(run_plan, args.local_state)
    local, paths, counts = layout["local"], layout["paths"], layout["counts"]
    checkpoints = layout["checkpoints"]
    local.mkdir(parents=True, exist_ok=True)
    run_lock = (local / "pipeline.lock").open("w")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze_run_plan(local, run_plan)
    (local / "pipeline.pid").write_text(str(os.getpid()) + "\n")
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        watcher = subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (local / "caffeinate.pid").write_text(str(watcher.pid) + "\n")
    state_path = local / "status.json"
    STATE_PATH = state_path
    # Engineering artifacts predate the production plan. Their exact original
    # configuration and frozen initialization remain independently reusable.
    wait_for_snapshot("data/frozen/overfit-train-32.jsonl", 32, state_path, "overfit_preparation")
    throughput_path = Path("reports/training/throughput.json")
    if not throughput_path.exists():
        atomic_json(state_path, {"phase": "training_throughput", "status": "measuring", "updated_unix": time.time()})
        with (local / "throughput.log").open("a", buffering=1) as log:
            subprocess.run([sys.executable, "scripts/measure_training_throughput.py", "--train",
                            "data/frozen/overfit-train-32.jsonl", "--output", str(throughput_path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    micro = json.loads(throughput_path.read_text())["selected_micro_batch"]
    overfit = train_stage("overfit", "configs/overfit.yaml", "checkpoints/overfit", {"micro_batch_episodes": micro}, state_path, local, run_plan=run_plan)
    overfit_metrics = json.loads(Path(overfit["best_checkpoint"], "dev_metrics.json").read_text())
    if overfit_metrics["decision_accuracy"] < 0.99:
        raise RuntimeError("32-episode overfit gate did not reach 99%; diagnose training before scaling")
    wait_for_snapshot(paths["pilot"], counts["pilot"], state_path, "pilot_preparation", run_plan=run_plan)
    micro = profile_formal_train(paths["pilot"], local, state_path, run_plan=run_plan)
    wait_for_snapshot(paths["dev"], counts["dev"], state_path, "pilot_preparation", expected_split="dev", run_plan=run_plan)
    pilot_data = local / "frozen-pilot-data.json"
    if run_plan:
        freeze_experiment_data(pilot_data, {"pilot_train": paths["pilot"], "dev": paths["dev"]}, run_plan=run_plan)
    pilot = train_stage("pilot", "configs/pilot.yaml", checkpoints / "pilot",
                        formal_stage_changes("pilot", micro, run_plan=run_plan), state_path, local, run_plan=run_plan)
    if run_plan:
        verify_frozen_data(pilot_data, run_plan=run_plan)
    pilot_metrics = json.loads(Path(pilot["best_checkpoint"], "dev_metrics.json").read_text())
    if pilot_metrics["coverage"] == 0 or pilot_metrics["answerable_top1"] == 0:
        raise RuntimeError("Pilot learned no usable candidate selections; diagnose before scaling")
    wait_for_snapshot(paths["train"], counts["train"], state_path, "main_preparation", run_plan=run_plan)
    experiment_data = local / "frozen-experiment-data.json"
    freeze_experiment_data(experiment_data, {"pilot_train": paths["pilot"], "main_train": paths["train"],
                                            "dev": paths["dev"]}, run_plan=run_plan)
    write_record(local / "pilot-subset-verification.json",
                 verify_pilot_subset(paths["pilot"], paths["train"], run_plan=run_plan), run_plan)
    initializations = seed_initializations(run_plan)
    runs = []
    for seed in (run_plan.document["training_seeds"] if run_plan else (42, 43, 44)):
        verify_frozen_data(experiment_data, run_plan=run_plan)
        run = train_stage(f"main-seed-{seed}", "configs/main.yaml", checkpoints / f"main-seed-{seed}",
                          formal_stage_changes("main", micro, seed=seed, initial_model=initializations[seed]["initial_model"],
                                               run_plan=run_plan), state_path, local, run_plan=run_plan)
        runs.append({"seed": seed, **run})
    verify_frozen_data(experiment_data, run_plan=run_plan)
    selected = max(runs, key=lambda run: run["best_dev_key"])
    v0 = checkpoints / "ranker-v0-selected"
    shutil.copytree(selected["best_checkpoint"], v0, dirs_exist_ok=True)
    write_record(local / "ranker-v0-ready.json", {"checkpoint": str(v0), "selected_seed": selected["seed"],
                                                 "selected_by": "Dev only", "runs": runs,
                                                 "weight_sha256": sha256_file(v0 / "model.safetensors")}, run_plan)
    # Teacher/data agent mines only a new training pool, not final Test failures.
    wait_for_snapshot(paths["hardening"], counts["hardening"], state_path, "hardening_preparation", run_plan=run_plan)
    verify_frozen_data(experiment_data, run_plan=run_plan)
    write_record(local / "hardening-data-mixture.json",
                 verify_hardening_mix(paths["train"], paths["hardening"], run_plan=run_plan), run_plan)
    hardening_data = local / "frozen-hardening-data.json"
    if run_plan:
        freeze_experiment_data(hardening_data, {"hardening_train": paths["hardening"], "dev": paths["dev"]}, run_plan=run_plan)
    hardened = train_stage("hardening", "configs/hardening.yaml", checkpoints / "hardening",
                           formal_stage_changes("hardening", micro, seed=selected["seed"], run_plan=run_plan),
                           state_path, local, run_plan=run_plan)
    verify_frozen_data(experiment_data, run_plan=run_plan)
    if run_plan:
        verify_frozen_data(hardening_data, run_plan=run_plan)
    previous = json.loads((v0 / "dev_metrics.json").read_text())
    current = json.loads(Path(hardened["best_checkpoint"], "dev_metrics.json").read_text())
    gates = run_plan.document["quality_gates"] if run_plan else {"minimum_dev_group_episodes": 20, "maximum_group_regression": .05}
    group_regressions = []
    for name, old in previous.get("groups", {}).items():
        new = current.get("groups", {}).get(name)
        if (old["episodes"] >= gates["minimum_dev_group_episodes"] and new
                and new["decision_accuracy"] < old["decision_accuracy"] - gates["maximum_group_regression"]):
            group_regressions.append(name)
    accept = current["selection_metric"] > previous["selection_metric"] and not group_regressions
    release = checkpoints / "ranker-v1-candidate"
    shutil.copytree(hardened["best_checkpoint"] if accept else v0, release, dirs_exist_ok=True)
    write_record(local / "hardening-selection.json", {"accepted": accept, "group_regressions": group_regressions,
                                                     "selected_by": "Dev only", "hardening_executed": True,
                                                     "v0_dev": previous, "hardening_dev": current,
                                                     "selection_rule": "Dev balanced accuracy improves; no >5pp regression in group n>=20"}, run_plan)
    verify_run_plan(run_plan)
    subprocess.run([sys.executable, "-m", "pastewhat_ranker.export", "--model", str(release), "--output", str(release / "mlx")], check=True)
    write_record(local / "ready-for-calibration.json",
                 training_handoff(layout, micro, initializations, selected["seed"], accept, run_plan=run_plan), run_plan)
    write_record(state_path, {"phase": "calibration_handoff", "status": "training_stages_complete",
                              "updated_unix": time.time()}, run_plan)


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
