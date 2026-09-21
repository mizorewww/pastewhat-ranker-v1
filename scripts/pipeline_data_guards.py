"""Immutable Train/Dev experiment and hard-example mixture guards.

This module never reads Calibration/Test. It catches accidental data replacement
between stages and guarantees the specified original/new hardening proportions.
"""

import hashlib
import json
from pathlib import Path

from pastewhat_ranker.train import atomic_json, read_allowed_data


def verify_run_plan(run_plan):
    """Recheck the plan and its source/evidence bindings without opening data."""
    if run_plan is None:
        return
    from run_contract import load_run_plan

    run_plan.verify_unchanged()
    if load_run_plan(run_plan.path).binding() != run_plan.binding():
        raise ValueError("The registered training plan binding changed")


def verify_binding(record, run_plan, description):
    if run_plan is not None and any(record.get(key) != value for key, value in run_plan.binding().items()):
        raise ValueError(f"{description} belongs to a different registered run")


def freeze_run_plan(directory, run_plan):
    """Keep the first registered identity immutable across process restarts."""
    verify_run_plan(run_plan)
    if run_plan is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "run-plan-binding.json"
    record = {**run_plan.binding(), "path": str(run_plan.path)}
    if destination.exists():
        if json.loads(destination.read_text()) != record:
            raise ValueError("Pipeline state is already bound to another run plan")
    else:
        atomic_json(destination, record)


def verify_snapshot_binding(path, count, expected_split, run_plan):
    """The data owner publishes the manifest before the atomic JSONL ready file."""
    if run_plan is None:
        return
    verify_run_plan(run_plan)
    manifest = json.loads(Path(path).with_suffix(".manifest.json").read_text())
    verify_binding(manifest, run_plan, "Frozen training snapshot")
    if (manifest.get("episodes") != count or manifest.get("split") != expected_split
            or manifest.get("sha256") != file_hash(path)
            or manifest.get("family_partition_sha256") != run_plan.document["family_partition_sha256"]):
        raise ValueError("Frozen training snapshot metadata differs from its registered content")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def semantic_record(episode):
    # Audit metadata may grow; model-visible content, labels, and mapping cannot.
    return json.dumps({key: episode[key] for key in ("id", "family_id", "context", "entries", "label")},
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def verify_pilot_subset(pilot_path, main_path, *, run_plan=None):
    if run_plan is not None:
        verify_snapshot_binding(pilot_path, run_plan.document["pilot_episodes"], "train", run_plan)
        verify_snapshot_binding(main_path, run_plan.target("train"), "train", run_plan)
    pilot = read_allowed_data(pilot_path, expected_split="train")
    main = {episode["id"]: episode for episode in read_allowed_data(main_path, expected_split="train")}
    for episode in pilot:
        if episode["id"] not in main or semantic_record(main[episode["id"]]) != semantic_record(episode):
            raise ValueError("The pilot must be an unchanged subset of main Train, including labels and candidates")
    return {**(run_plan.binding() if run_plan else {}),
            "pilot_episodes": len(pilot), "main_episodes": len(main),
            "pilot_is_unchanged_subset": True, "pilot_sha256": file_hash(pilot_path), "main_sha256": file_hash(main_path)}


def freeze_experiment_data(destination, snapshots, partition="data_tools/family_partition.json", *, run_plan=None):
    verify_run_plan(run_plan)
    destination = Path(destination)
    values = {name: {"path": str(path), "sha256": file_hash(path)} for name, path in snapshots.items()}
    values["family_partition"] = {"path": str(partition), "sha256": file_hash(partition)}
    if run_plan is not None:
        values["run_plan"] = {"path": str(run_plan.path), "sha256": run_plan.sha256}
        projection = Path("tools/context_projection/provenance.json")
        values["projection_provenance"] = {"path": str(projection), "sha256": file_hash(projection)}
        for name, path in snapshots.items():
            sidecar = Path(path).with_suffix(".manifest.json")
            metadata = json.loads(sidecar.read_text())
            verify_binding(metadata, run_plan, "Experiment snapshot manifest")
            if metadata.get("sha256") != values[name]["sha256"]:
                raise ValueError("A snapshot changed after its owning data manifest was published")
            values[name + "_manifest"] = {"path": str(sidecar), "sha256": file_hash(sidecar)}
        values = {**run_plan.binding(), "files": values}
    if destination.exists():
        if json.loads(destination.read_text()) != values:
            raise ValueError("Previously frozen experiment Train/Dev/partition files have changed")
    else:
        atomic_json(destination, values)
    return values


def verify_frozen_data(manifest, *, run_plan=None):
    verify_run_plan(run_plan)
    values = json.loads(Path(manifest).read_text())
    verify_binding(values, run_plan, "Frozen experiment manifest")
    for name, record in values.get("files", values).items():
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen experiment data changed between training stages: {name}")


def verify_hardening_mix(main_path, hardening_path, original_count=None, new_count=None, *, run_plan=None):
    if run_plan is not None:
        planned = run_plan.document["hardening"]
        if original_count not in (None, planned["retained_original"]) or new_count not in (None, planned["accepted_new"]):
            raise ValueError("Hardening mixture arguments differ from the registered plan")
        original_count, new_count = planned["retained_original"], planned["accepted_new"]
        verify_snapshot_binding(main_path, run_plan.target("train"), "train", run_plan)
        verify_snapshot_binding(hardening_path, original_count + new_count, "train", run_plan)
    else:
        original_count = 5000 if original_count is None else original_count
        new_count = 5000 if new_count is None else new_count
    original = {episode["id"]: episode for episode in read_allowed_data(main_path, expected_split="train")}
    hardening = read_allowed_data(hardening_path, expected_split="train")
    retained, novel = [], []
    for episode in hardening:
        if episode["id"] in original:
            if semantic_record(episode) != semantic_record(original[episode["id"]]):
                raise ValueError("Original hardening examples cannot change their labels or model-visible inputs")
            retained.append(episode["id"])
        else:
            novel.append(episode["id"])
    if len(retained) != original_count or len(novel) != new_count:
        raise ValueError(f"Hardening requires {original_count} unchanged originals and {new_count} new episode IDs; got {len(retained)} and {len(novel)}")
    return {**(run_plan.binding() if run_plan else {}),
            "original_episodes": len(retained), "new_episode_ids": len(novel),
            "originals_preserved_exactly": True, "all_families_in_train_partition": True,
            "main_sha256": file_hash(main_path), "hardening_sha256": file_hash(hardening_path),
            "scope": "Verifies identity/content/partition mixture; teacher review and semantic novelty are separately audited by the data producer"}
