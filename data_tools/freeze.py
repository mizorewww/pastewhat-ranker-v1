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
from data_tools.teacher import atomic_json, canonical_bytes, sha256, utc_now
from pastewhat_ranker.preprocess import Preprocessor


def publish_bytes(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", help="Optional explicitly reviewed Train/Dev engineering seed")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "laya-mlx/models/laya-multilingual/tokenizer"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT / "data" / "frozen"):
        raise SystemExit("Frozen snapshots must stay in data/frozen")
    if any(term in output.name.lower() for term in ("test", "calibration")):
        raise SystemExit("Train/Dev producer cannot write held-out split snapshots")
    source = ROOT / "data" / f"{args.split}.jsonl"
    if args.source:
        source = (ROOT / args.source).resolve()
        if not args.overfit or args.split != "train" or not source.is_relative_to(ROOT / "local") or not source.name.startswith("train_"):
            raise SystemExit("Alternate input is limited to a local, reviewed Train engineering seed")
    payload = source.read_bytes()
    episodes = [json.loads(line) for line in payload.splitlines()]
    if len(episodes) < args.count:
        raise SystemExit(f"Need {args.count} accepted episodes, found {len(episodes)}")
    allowed_families = {family["id"] for family in json.loads(PARTITION_PATH.read_text())["families"][args.split]}
    preprocessor = Preprocessor(args.tokenizer)
    hashes = set()
    for episode in episodes:
        if episode["family_id"] not in allowed_families:
            raise ValueError("Conceptual family crosses split ownership")
        required_audits = ("generation_audit_id", "label_audit_id", "blind_label_audit_id", "family_review_audit_id")
        if any(not episode.get("provenance", {}).get(key) for key in required_audits):
            raise ValueError("An episode has not passed all teacher review gates")
        if not args.overfit and (episode["provenance"].get("family_review_protocol") != "blind-68-operation-classification" or episode["provenance"].get("observed_family_id") != episode["family_id"]):
            raise ValueError("Production data requires a blind observed-family match")
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
    pilot_path = ROOT / "data/frozen/pilot-train-5000.jsonl"
    if args.split == "train" and args.count == 20000:
        if not pilot_path.is_file():
            raise ValueError("Freeze the 5,000-row pilot before the full 20,000-row training set")
        pilot_bytes = pilot_path.read_bytes()
        pilot = [json.loads(line) for line in pilot_bytes.splitlines()]
        if len(pilot) != 5000:
            raise ValueError("Pilot snapshot has the wrong episode count")
        full_by_id = {episode["id"]: canonical_bytes(episode) for episode in episodes}
        if any(full_by_id.get(episode["id"]) != canonical_bytes(episode) for episode in pilot):
            raise ValueError("The pilot is not an unchanged subset of the full training pool")
        pilot_proof = {"path": str(pilot_path.relative_to(ROOT)), "episodes": len(pilot), "sha256": sha256(pilot_bytes), "unchanged_subset": True}
    episodes.sort(key=lambda episode: sha256(episode["id"].encode()))
    chosen = []
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
    manifest = {"split": args.split, "episodes": len(chosen), "sha256": sha256(data), "source_sha256": sha256(payload), "created_at": utc_now(), "path": str(output.relative_to(ROOT)), "family_partition_sha256": sha256(PARTITION_PATH.read_bytes()), "families": dict(Counter(episode["family_id"] for episode in chosen)), "labels": dict(Counter(episode["label"]["decision"] if episode["label"]["decision"] == "select" else episode["label"]["abstain_reason"] for episode in chosen)), "multiple_positive_episodes": sum(len(episode["label"]["acceptable_ids"]) > 1 for episode in chosen), "candidate_counts": dict(Counter(len(episode["entries"]) for episode in chosen)), "fingerprints": str(fingerprint_path.relative_to(ROOT)), "preprocessing": preprocessor.manifest(), "human_validated": False, "review": "Two blind teacher label passes plus independent family/deployment review and programmatic invariants."}
    manifest["source_path"] = str(source.relative_to(ROOT))
    manifest["select_with_same_kind_negative"] = with_same_kind_negative
    manifest["context_coverage"] = {"no_accessibility": sum(not episode["context"]["hasAccessibility"] for episode in chosen), "with_selection": sum(bool(episode["context"]["selectedText"]) for episode in chosen), "application_categories": dict(Counter(episode["context"]["applicationCategory"] for episode in chosen)), "input_surfaces": dict(Counter(episode["context"]["inputSurface"] for episode in chosen))}
    manifest["purpose"] = "engineering-overfit-check, not representative evaluation" if args.overfit else "frozen training/development data"
    if pilot_proof:
        manifest["pilot_subset"] = pilot_proof
    publish_bytes(output.with_suffix(".manifest.json"), json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n")
    # JSONL existence is the training pipeline's readiness signal. Its complete
    # sidecars are visible first, then the fsynced snapshot is renamed atomically.
    publish_bytes(output, data)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
