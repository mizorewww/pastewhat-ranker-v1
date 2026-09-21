"""Freeze a reviewed Train/Dev snapshot and publish content-only fingerprints."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random

from data_tools.generate import PARTITION_PATH, ROOT, validate_labels
from data_tools.content import content_fingerprint
from data_tools.deployment import placement_issue
from data_tools.replay import ReplayVerifier
from data_tools.teacher import atomic_json, canonical_bytes, sha256, utc_now
from pastewhat_ranker.preprocess import Preprocessor
from run_contract import action_quotas, family_quotas, load_run_plan


def publish_bytes(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def choose_pilot(episodes, families):
    """Take exact action strata while spreading each stratum across families."""
    buckets = {"select": {}, "no_match": {}, "missing_context": {}}
    for episode in sorted(episodes, key=lambda episode: sha256(episode["id"].encode())):
        label = episode["label"]
        bucket = "select" if label["decision"] == "select" else "no_match" if label["abstain_reason"] == "no_match" else "missing_context"
        buckets[bucket].setdefault(episode["family_id"], []).append(episode)
    selected = []
    for bucket, count in (("select", 3500), ("no_match", 1000), ("missing_context", 500)):
        groups = buckets[bucket]
        remaining = count
        index = 0
        while remaining:
            added = False
            for family in sorted(families):
                candidates = groups.get(family, [])
                if index < len(candidates):
                    selected.append(candidates[index])
                    remaining -= 1
                    added = True
                    if not remaining:
                        break
            if not added:
                raise ValueError(f"Pilot needs {count} independently labeled {bucket} episodes")
            index += 1
    if {episode["family_id"] for episode in selected} != set(families):
        raise ValueError("Pilot must cover every frozen Train conceptual family")
    return selected


def choose_registered(episodes, partition, split, count, *, required_ids=()):
    """Select the pre-registered exact family/action allocation without label edits."""
    quotas = action_quotas(family_quotas(partition, split, count))
    required_ids = set(required_ids)
    grouped = {}
    for episode in sorted(episodes, key=lambda row: (row["id"] not in required_ids, sha256(row["id"].encode()))):
        label = episode["label"]
        bucket = "select" if label["decision"] == "select" else "no_match" if label["abstain_reason"] == "no_match" else "missing_intent"
        grouped.setdefault((episode["family_id"], bucket), []).append(episode)
    chosen = []
    for family, buckets in quotas.items():
        for bucket, number in buckets.items():
            candidates = grouped.get((family, bucket), [])
            if len(candidates) < number:
                raise ValueError(f"Registered stratum {family}/{bucket} needs {number}, found {len(candidates)}")
            chosen.extend(candidates[:number])
    if not required_ids <= {episode["id"] for episode in chosen}:
        raise ValueError("The registered main selection would drop immutable pilot rows")
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", help="Reviewed engineering seed or the owning split's accepted partial-batch pool")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--run-plan")
    parser.add_argument("--stage", choices=("pilot", "train", "dev"))
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "laya-mlx/models/laya-multilingual/tokenizer"))
    args = parser.parse_args()
    run_plan = load_run_plan(args.run_plan) if args.run_plan else None
    if run_plan:
        if args.overfit or args.stage is None:
            raise SystemExit("Registered formal snapshots require an explicit stage; engineering32 remains separate")
        expected_split = "dev" if args.stage == "dev" else "train"
        expected_count = run_plan.document["pilot_episodes"] if args.stage == "pilot" else run_plan.target(expected_split)
        if args.split != expected_split or args.count != expected_count:
            raise SystemExit("Requested split/count differs from the registered stage")
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "data" / "frozen"):
        raise SystemExit("Frozen snapshots must stay in data/frozen")
    if any(term in output.name.lower() for term in ("test", "calibration")):
        raise SystemExit("Train/Dev producer cannot write held-out split snapshots")
    if run_plan and output != ROOT / run_plan.data_path(args.stage):
        raise SystemExit("Snapshot path differs from the registered stage")
    pool_directory = ROOT / "local/data-production" / run_plan.run_id if run_plan else ROOT / "data"
    source = pool_directory / f"{args.split}.jsonl"
    if args.source:
        source = (ROOT / args.source).resolve()
        engineering_source = args.overfit and args.split == "train" and source.is_relative_to(ROOT / "local") and source.name.startswith("train_")
        accepted_pool = not args.overfit and source == pool_directory / f"{args.split}.accepted.jsonl"
        if not (engineering_source or accepted_pool):
            raise SystemExit("Alternate input must be a reviewed Train seed or the owning split's accepted pool")
    if run_plan:
        source_manifest = json.loads(source.with_suffix(".manifest.json").read_text())
        if any(source_manifest.get(key) != value for key, value in run_plan.binding().items()):
            raise ValueError("Source pool is not bound to this registered plan")
    payload = source.read_bytes()
    if run_plan and source_manifest.get("sha256") != sha256(payload):
        raise ValueError("Source pool does not match its atomically published manifest")
    episodes = [json.loads(line) for line in payload.splitlines()]
    if len(episodes) < args.count:
        raise SystemExit(f"Need {args.count} accepted episodes, found {len(episodes)}")
    partition = json.loads(PARTITION_PATH.read_text())
    allowed_families = {family["id"] for family in partition["families"][args.split]}
    preprocessor = Preprocessor(args.tokenizer)
    replay = None if args.overfit else ReplayVerifier(ROOT, args.split, preprocessor)
    hashes = set()
    for episode in episodes:
        if run_plan and any(episode.get("provenance", {}).get(key) != value for key, value in run_plan.binding().items()):
            raise ValueError("An episode belongs to a different registered generation run")
        if episode["family_id"] not in allowed_families:
            raise ValueError("Conceptual family crosses split ownership")
        if not args.overfit and placement_issue(episode):
            raise ValueError(f"Unusable synthetic paste placement in {episode['id']}: {placement_issue(episode)}")
        required_audits = ("generation_audit_id", "label_audit_id", "blind_label_audit_id", "family_review_audit_id")
        if any(not episode.get("provenance", {}).get(key) for key in required_audits):
            raise ValueError("An episode has not passed all teacher review gates")
        if not args.overfit and (episode["provenance"].get("family_review_protocol") != "blind-68-operation-classification" or episode["provenance"].get("observed_family_id") != episode["family_id"]):
            raise ValueError("Production data requires a blind observed-family match")
        if replay is not None:
            replay.verify(episode)
        prepared = preprocessor.prepare_episode(episode)
        if prepared["preprocessing"]["visible_sha256"] != episode["provenance"]["label_visible_sha256"]:
            raise ValueError("Teacher and student visible input differs")
        if prepared["context"] != episode["context"] or prepared["entries"] != episode["entries"]:
            raise ValueError("Prepared features are not idempotent")
        validate_labels({"labels": [{"id": episode["id"], "label": episode["label"]}]}, [episode])
        encoded = preprocessor.encode_episode(episode)
        if not 1 <= len(encoded["input_ids"]) <= 20 or any(len(row) > 1024 for row in encoded["input_ids"]):
            raise ValueError("Candidate count or pair token budget violated")
        digest = content_fingerprint(episode)
        if digest in hashes:
            raise ValueError("Duplicate visible content, ignoring candidate IDs and order")
        hashes.add(digest)
    pilot_proof = None
    pilot_path = ROOT / run_plan.data_path("pilot") if run_plan else ROOT / "data/frozen/pilot-train-5000.jsonl"
    pilot_count = run_plan.document["pilot_episodes"] if run_plan else 5000
    main_count = run_plan.target("train") if run_plan else 20000
    pilot = []
    if args.split == "train" and args.count == main_count and (not run_plan or args.stage == "train"):
        if not pilot_path.is_file():
            raise ValueError("Freeze the registered pilot before the full training set")
        pilot_bytes = pilot_path.read_bytes()
        pilot = [json.loads(line) for line in pilot_bytes.splitlines()]
        if len(pilot) != pilot_count:
            raise ValueError("Pilot snapshot has the wrong episode count")
        full_by_id = {episode["id"]: canonical_bytes(episode) for episode in episodes}
        if any(full_by_id.get(episode["id"]) != canonical_bytes(episode) for episode in pilot):
            raise ValueError("The pilot is not an unchanged subset of the full training pool")
        pilot_proof = {"path": str(pilot_path.relative_to(ROOT)), "episodes": len(pilot), "sha256": sha256(pilot_bytes), "unchanged_subset": True}
    episodes.sort(key=lambda episode: sha256(episode["id"].encode()))
    chosen = choose_registered(episodes, partition, args.split, args.count, required_ids=[row["id"] for row in pilot]) if run_plan else choose_pilot(episodes, allowed_families) if args.split == "train" and args.count == 5000 and not args.overfit else []
    if args.overfit:
        # Include a multi-positive episode and every available abstention reason,
        # then mix across families. This is training-data selection, not an eval.
        predicates = [lambda e: len(e["label"]["acceptable_ids"]) > 1]
        predicates.extend(lambda e, reason=reason: e["label"]["abstain_reason"] == reason for reason in ("no_match", "ambiguous", "insufficient_context"))
        for predicate in predicates:
            match = next((episode for episode in episodes if predicate(episode) and episode not in chosen), None)
            if match:
                chosen.append(match)
        if not any(len(episode["label"]["acceptable_ids"]) > 1 for episode in chosen) or not any(episode["label"]["decision"] == "abstain" for episode in chosen):
            raise ValueError("Overfit set needs multi-positive and abstention examples")
    for episode in episodes:
        if len(chosen) == args.count:
            break
        if episode not in chosen:
            chosen.append(episode)
    random.Random(42).shuffle(chosen)
    selected = [episode for episode in chosen if episode["label"]["decision"] == "select"]
    with_same_kind_negative = 0
    for episode in selected:
        positives = set(episode["label"]["acceptable_ids"])
        positive_kinds = {entry["kind"] for entry in episode["entries"] if entry["id"] in positives}
        with_same_kind_negative += any(entry["id"] not in positives and entry["kind"] in positive_kinds for entry in episode["entries"])
    if not args.overfit:
        if not selected or with_same_kind_negative / len(selected) < 0.5:
            raise ValueError("At least half of selectable episodes must contain a same-kind negative")
        labels = Counter(episode["label"]["decision"] if episode["label"]["decision"] == "select" else episode["label"]["abstain_reason"] for episode in chosen)
        fractions = (labels["select"] / len(chosen), labels["no_match"] / len(chosen), (labels["ambiguous"] + labels["insufficient_context"]) / len(chosen))
        if any(abs(actual - target) > 0.025 for actual, target in zip(fractions, (0.7, 0.2, 0.1), strict=True)):
            raise ValueError("Observed labels fall outside the predeclared 70/20/10 tolerance of 2.5 percentage points")
    data = b"".join(canonical_bytes(episode) + b"\n" for episode in chosen)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.read_bytes() != data:
        raise ValueError("Immutable snapshot already exists with different data")
    fingerprints = [{"id": episode["id"], "family_id": episode["family_id"], "content_sha256": content_fingerprint(episode)} for episode in chosen]
    fingerprint_path = output.with_suffix(".fingerprints.jsonl")
    publish_bytes(fingerprint_path, b"".join(canonical_bytes(item) + b"\n" for item in fingerprints))
    manifest = {**(run_plan.binding() if run_plan else {}), "split": args.split, "episodes": len(chosen), "sha256": sha256(data), "source_sha256": sha256(payload), "created_at": utc_now(), "path": str(output.relative_to(ROOT)), "family_partition_sha256": sha256(PARTITION_PATH.read_bytes()), "families": dict(Counter(episode["family_id"] for episode in chosen)), "labels": dict(Counter(episode["label"]["decision"] if episode["label"]["decision"] == "select" else episode["label"]["abstain_reason"] for episode in chosen)), "multiple_positive_episodes": sum(len(episode["label"]["acceptable_ids"]) > 1 for episode in chosen), "candidate_counts": dict(Counter(len(episode["entries"]) for episode in chosen)), "fingerprints": str(fingerprint_path.relative_to(ROOT)), "preprocessing": preprocessor.manifest(), "human_validated": False, "review": "Two blind teacher label passes plus independent family/deployment review and programmatic invariants."}
    manifest["source_path"] = str(source.relative_to(ROOT))
    manifest["select_with_same_kind_negative"] = with_same_kind_negative
    manifest["context_coverage"] = {"no_accessibility": sum(not episode["context"]["hasAccessibility"] for episode in chosen), "with_selection": sum(bool(episode["context"]["selectedText"]) for episode in chosen), "application_categories": dict(Counter(episode["context"]["applicationCategory"] for episode in chosen)), "input_surfaces": dict(Counter(episode["context"]["inputSurface"] for episode in chosen))}
    manifest["purpose"] = "engineering-overfit-check, not representative evaluation" if args.overfit else "frozen training/development data"
    if pilot_proof:
        manifest["pilot_subset"] = pilot_proof
    if run_plan:
        run_plan.verify_unchanged()
        manifest["stage"] = args.stage
    publish_bytes(output.with_suffix(".manifest.json"), json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n")
    # JSONL existence is the training pipeline's readiness signal. Its complete
    # sidecars are visible first, then the fsynced snapshot is renamed atomically.
    publish_bytes(output, data)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
