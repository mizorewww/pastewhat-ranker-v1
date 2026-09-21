"""Read aggregate training proof without opening any Train/Dev examples."""
from __future__ import annotations

import json
import math
from pathlib import Path

from evaluations.common import sha256
from run_contract import RunPlan


def training_handoff_inputs(plan: RunPlan, deployment: Path) -> dict[str, Path]:
    """Validate completed training and return all evidence to bind in final freeze."""
    plan.verify_unchanged()
    path = plan.pipeline_directory / "ready-for-calibration.json"
    handoff = json.loads(path.read_text())
    document = plan.document
    seeds = document["training_seeds"]
    reference = plan.checkpoint_directory / "ranker-v1-candidate"
    if (handoff.get("version") != "pastewhat-training-handoff-v2" or
            any(handoff.get(key) != value for key, value in plan.binding().items()) or
            handoff.get("selected_by") != "Dev only" or
            handoff.get("registered_seeds") != seeds or handoff.get("completed_seeds") != seeds or
            handoff.get("hardening_executed") is not True or type(handoff.get("hardening_accepted")) is not bool or
            Path(handoff["reference_checkpoint"]).resolve() != reference.resolve() or
            Path(handoff["deployment_checkpoint"]).resolve() != deployment.resolve() or
            deployment.resolve() != (reference / "mlx").resolve()):
        raise ValueError("Training handoff does not prove the registered Dev-only production run")
    inputs = {"training_handoff": path, "training_reference_weights": reference / "model.safetensors"}

    def bound_artifact(name: str, record: dict, expected: Path, *, read_json: bool = True):
        actual = Path(record["path"])
        if actual.resolve() != expected.resolve() or sha256(actual) != record.get("sha256"):
            raise ValueError("Training evidence path/hash mismatch: " + name)
        inputs[name] = actual
        return json.loads(actual.read_text()) if read_json else None

    completion = handoff["training_completion"]
    counts = {"pilot": document["pilot_episodes"], "train": plan.target("train"), "dev": plan.target("dev"),
              "hardening": document["hardening"]["accepted_new"] + document["hardening"]["retained_original"]}
    if completion.get("status") != "completed" or completion.get("data_counts") != counts:
        raise ValueError("Training handoff counts or completion status differ from the registered plan")
    stages = completion["stages"]
    if set(stages) != {"pilot", "hardening", *(f"main_seed_{seed}" for seed in seeds)}:
        raise ValueError("Training handoff is missing a required completed stage")
    for name, stage in stages.items():
        role = "main" if name.startswith("main_seed_") else name
        seed = int(name.removeprefix("main_seed_")) if role == "main" else 42 if role == "pilot" else handoff["selected_seed"]
        stage_root = plan.checkpoint_directory / (f"main-seed-{seed}" if role == "main" else name)
        summary = bound_artifact(f"training_stage_{name}_summary", stage["summary"], stage_root / "training_summary.json")
        manifest = bound_artifact(f"training_stage_{name}_run_manifest", stage["run_manifest"], stage_root / "run_manifest.json")
        bound_artifact(f"training_stage_{name}_train_config", stage["train_config"], stage_root / "train_config.yaml", read_json=False)
        config = manifest["config"]
        count = counts["train" if role == "main" else role]
        effective, epochs = document["effective_batch_episodes"], document["epochs"][role]
        warmup = 0 if role == "hardening" else document["head_warmup_steps"]
        per_epoch = math.ceil(count / effective)
        cycles, remaining = divmod(warmup, per_epoch)
        expected = {"seed": seed, "train_episodes": count, "dev_episodes": counts["dev"],
                    "epochs": epochs, "head_warmup_steps": warmup, "effective_batch_episodes": effective,
                    "global_steps": warmup + epochs * per_epoch,
                    "seen_episodes_including_head_warmup": cycles * count + min(count, remaining * effective) + epochs * count}
        if (stage.get("status") != "completed" or any(stage.get(key) != value for key, value in expected.items()) or
                summary.get("status") != "completed" or summary.get("engineering_overfit") is not False or
                config.get("engineering_overfit") or config.get("max_steps") or
                any(config.get(key) != value for key, value in plan.binding().items()) or
                any(config.get(key) != expected[key] for key in ("seed", "epochs", "head_warmup_steps", "effective_batch_episodes")) or
                manifest.get("train_episodes") != count or manifest.get("dev_episodes") != counts["dev"] or
                stage["initial_weight_sha256"] != manifest.get("initial_weight_sha256") or
                any(summary.get(key) != stage.get(key) for key in
                    ("global_steps", "seen_episodes_including_head_warmup", "best_dev_key", "best_weight_sha256"))):
            raise ValueError("Training stage does not prove its complete registered workload: " + name)

    local = plan.pipeline_directory
    expected_evidence = {
        "pilot_data": local / "frozen-pilot-data.json", "experiment_data": local / "frozen-experiment-data.json",
        "pilot_subset": local / "pilot-subset-verification.json",
        "seed_initializations": plan.report_directory / "seed-initializations.json",
        "v0_selection": local / "ranker-v0-ready.json", "hardening_data": local / "frozen-hardening-data.json",
        "hardening_mix": local / "hardening-data-mixture.json", "hardening_selection": local / "hardening-selection.json",
        "run_plan_binding": local / "run-plan-binding.json", "conversion": deployment / "conversion.json",
    }
    if set(completion["evidence"]) != set(expected_evidence):
        raise ValueError("Training handoff does not bind every required aggregate evidence file")
    evidence = {name: bound_artifact("training_evidence_" + name, completion["evidence"][name], expected)
                for name, expected in expected_evidence.items()}
    for name, record in evidence.items():
        if name != "conversion" and any(record.get(key) != value for key, value in plan.binding().items()):
            raise ValueError("Training aggregate evidence belongs to a different registered run: " + name)
    winner = max(seeds, key=lambda seed: stages[f"main_seed_{seed}"]["best_dev_key"])
    selected = stages[f"main_seed_{winner}"]
    selection, mixture, subset = evidence["hardening_selection"], evidence["hardening_mix"], evidence["pilot_subset"]
    if (handoff.get("selected_seed") != winner or evidence["v0_selection"].get("selected_seed") != winner or
            evidence["v0_selection"].get("selected_by") != "Dev only" or selection.get("selected_by") != "Dev only" or
            selection.get("hardening_executed") is not True or selection.get("accepted") != handoff["hardening_accepted"] or
            evidence["v0_selection"].get("weight_sha256") != selected["best_weight_sha256"] or
            stages["hardening"]["initial_weight_sha256"] != selected["best_weight_sha256"] or
            subset.get("pilot_is_unchanged_subset") is not True or subset.get("pilot_episodes") != counts["pilot"] or
            subset.get("main_episodes") != counts["train"] or mixture.get("originals_preserved_exactly") is not True or
            mixture.get("all_families_in_train_partition") is not True or
            mixture.get("original_episodes") != document["hardening"]["retained_original"] or
            mixture.get("new_episode_ids") != document["hardening"]["accepted_new"]):
        raise ValueError("Training aggregate proof does not establish registered data mixing and Dev-only selection")
    initial = {row["seed"]: row for row in evidence["seed_initializations"]["seeds"]}
    if (evidence["seed_initializations"].get("status") != "passed" or set(initial) != set(seeds) or
            any(initial[seed].get("encoder_tensor_equality_to_seed42") is not True or
                stages[f"main_seed_{seed}"]["initial_weight_sha256"] != initial[seed]["model_sha256"] for seed in seeds) or
            stages["pilot"]["initial_weight_sha256"] != initial[42]["model_sha256"]):
        raise ValueError("Formal training did not restart from its frozen original-encoder initializations")
    reference_sha, deployment_sha = sha256(reference / "model.safetensors"), sha256(deployment / "model.safetensors")
    expected_sha = stages["hardening"]["best_weight_sha256"] if handoff["hardening_accepted"] else selected["best_weight_sha256"]
    conversion = evidence["conversion"]
    if (reference_sha != expected_sha or reference_sha != handoff.get("reference_weight_sha256") or
            deployment_sha != handoff.get("deployment_weight_sha256") or deployment_sha != handoff.get("weight_sha256") or
            conversion.get("reference_weight_sha256") != reference_sha or conversion.get("mlx_weight_sha256") != deployment_sha or
            conversion.get("strict_parameter_load") is not True):
        raise ValueError("Frozen deployment weights differ from the completed training selection and strict export")
    plan.verify_unchanged()
    return inputs
