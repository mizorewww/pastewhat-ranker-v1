"""Immutable Train/Dev experiment and hard-example mixture guards.

This module never reads Calibration/Test. It catches accidental data replacement
between stages and guarantees the specified original/new hardening proportions.
"""

import hashlib
import json
from pathlib import Path

from pastewhat_ranker.train import atomic_json, read_allowed_data


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


def verify_pilot_subset(pilot_path, main_path):
    pilot = read_allowed_data(pilot_path, expected_split="train")
    main = {episode["id"]: episode for episode in read_allowed_data(main_path, expected_split="train")}
    for episode in pilot:
        if episode["id"] not in main or semantic_record(main[episode["id"]]) != semantic_record(episode):
            raise ValueError("The pilot must be an unchanged subset of main Train, including labels and candidates")
    return {"pilot_episodes": len(pilot), "main_episodes": len(main),
            "pilot_is_unchanged_subset": True, "pilot_sha256": file_hash(pilot_path), "main_sha256": file_hash(main_path)}


def freeze_experiment_data(destination, snapshots, partition="data_tools/family_partition.json"):
    destination = Path(destination)
    values = {name: {"path": str(path), "sha256": file_hash(path)} for name, path in snapshots.items()}
    values["family_partition"] = {"path": str(partition), "sha256": file_hash(partition)}
    if destination.exists():
        if json.loads(destination.read_text()) != values:
            raise ValueError("Previously frozen experiment Train/Dev/partition files have changed")
    else:
        atomic_json(destination, values)
    return values


def verify_frozen_data(manifest):
    values = json.loads(Path(manifest).read_text())
    for name, record in values.items():
        if file_hash(record["path"]) != record["sha256"]:
            raise ValueError(f"Frozen experiment data changed between training stages: {name}")


def verify_hardening_mix(main_path, hardening_path, original_count=5000, new_count=5000):
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
    return {"original_episodes": len(retained), "new_episode_ids": len(novel),
            "originals_preserved_exactly": True, "all_families_in_train_partition": True,
            "main_sha256": file_hash(main_path), "hardening_sha256": file_hash(hardening_path),
            "scope": "Verifies identity/content/partition mixture; teacher review and semantic novelty are separately audited by the data producer"}
