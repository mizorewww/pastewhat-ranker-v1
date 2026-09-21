"""Rank new Train-only episodes for blind teacher review after ranker-v0.

Disagreement is a review priority, never proof that the student is wrong. This
tool does not edit labels, freeze a training set or inspect held-out examples.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import time

from pastewhat_ranker.calibration import score_features
from pastewhat_ranker.model import sha256_file
from pastewhat_ranker.worker import RankerScorer
from pastewhat_ranker.train import read_allowed_data


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def content_hash(episode: dict) -> str:
    entries = [{key: value for key, value in entry.items() if key != "id"} for entry in episode["entries"]]
    entries.sort(key=canonical)
    return hashlib.sha256(canonical({"context": episode["context"], "entries": entries})).hexdigest()


def inspect_score(episode: dict, response: dict) -> dict:
    features, top_id, beats_abstain = score_features(episode, response)
    entries = {entry["id"]: entry for entry in episode["entries"]}
    prediction = top_id if beats_abstain else None
    label = episode["label"]
    positive = set(label["acceptable_ids"])
    correct = prediction in positive if label["decision"] == "select" else prediction is None
    if correct:
        category = "correct_low_margin"
    elif label["decision"] == "abstain":
        category = "false_promotion_disagreement"
    elif prediction is None:
        category = "missed_match_disagreement"
    elif entries[prediction]["kind"] in {entries[key]["kind"] for key in positive}:
        category = "same_kind_choice_disagreement"
    else:
        category = "different_kind_choice_disagreement"
    # Distance to the runner-up action, not a correctness probability.
    margin = min(features[0], features[1]) if beats_abstain else -features[0]
    return {"id": episode["id"], "family_id": episode["family_id"], "category": category,
            "raw_recommended_id": prediction, "raw_top_id": top_id,
            "agrees_with_existing_teacher_label": correct,
            "action_margin": margin, "candidate_count": len(entries),
            "same_kind_negative_present": any(entry["kind"] in {entries[key]["kind"] for key in positive}
                                              and identifier not in positive for identifier, entry in entries.items())}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--original-train", type=Path, required=True)
    parser.add_argument("--v0-ready", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=6000, help="Proposals for review; final hard-example quota is 5,000 accepted new episodes")
    parser.add_argument("--gpu-exclusive-confirmation", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("A positive review proposal count is required")
    selected = json.loads(args.v0_ready.read_text())
    if selected.get("selected_by") != "Dev only":
        raise ValueError("Hard mining requires the Dev-selected ranker-v0 handoff")
    config = json.loads((args.model / "config.json").read_text())
    conversion = json.loads((args.model / "conversion.json").read_text())
    weight_hash = sha256_file(args.model / "model.safetensors")
    if config.get("deployment_backend") != "mlx" or config.get("deployment_precision") != "float16":
        raise ValueError("Expected the explicitly exported MLX FP16 v0")
    if config.get("reference_weight_sha256") != selected["weight_sha256"] or conversion.get("mlx_weight_sha256") != weight_hash:
        raise ValueError("Mining model does not match the Dev-selected v0")
    original = read_allowed_data(args.original_train, expected_split="train")
    pool = read_allowed_data(args.pool, expected_split="train")
    old_ids, old_contents = {episode["id"] for episode in original}, {content_hash(episode) for episode in original}
    new_contents = set()
    for episode in pool:
        fingerprint = content_hash(episode)
        if episode["id"] in old_ids or fingerprint in old_contents or fingerprint in new_contents:
            raise ValueError("Mining pool contains a reused original episode or duplicate visible content")
        if episode["context"].get("isSecure") or not 1 <= len(episode["entries"]) <= 20:
            raise ValueError("Mining pool must contain actual nonsecure 1–20-candidate model inputs")
        new_contents.add(fingerprint)
    provenance = {"version": "pastewhat-train-mining-v1", "pool_sha256": sha256_file(args.pool),
                  "original_train_sha256": sha256_file(args.original_train),
                  "v0_handoff_sha256": sha256_file(args.v0_ready), "mlx_weight_sha256": weight_hash,
                  "preprocess_sha256": sha256_file(args.model / "preprocess.json"),
                  "mining_source_sha256": sha256_file(__file__), "requested_proposals": args.count}
    if args.output.exists():
        if not args.resume or json.loads((args.output / "provenance.json").read_text()) != provenance:
            raise ValueError("Existing mining output requires --resume with exactly the same inputs and code")
        if (args.output / "selection.json").exists():
            raise ValueError("This immutable mining run is already complete")
    else:
        if args.resume:
            raise ValueError("There is no mining run to resume")
        args.output.mkdir(parents=True)
        write_json(args.output / "provenance.json", provenance)
    score_path = args.output / "scored-pool.jsonl"
    prior = [json.loads(line) for line in score_path.read_text().splitlines()] if score_path.exists() else []
    if [row["id"] for row in prior] != [row["id"] for row in pool[:len(prior)]]:
        raise ValueError("Incomplete mining scores are not the exact pool prefix")
    scorer = RankerScorer(args.model, backend="mlx")
    results = list(prior)
    started = time.monotonic()
    with score_path.open("a") as stream:
        for episode in pool[len(prior):]:
            row = inspect_score(episode, scorer.score(episode))
            stream.write(canonical(row).decode() + "\n")
            stream.flush()
            results.append(row)
            if len(results) % 100 == 0 or len(results) == len(pool):
                progress = {"scored": len(results), "total": len(pool), "categories": dict(Counter(item["category"] for item in results)),
                            "elapsed_this_process_seconds": time.monotonic() - started}
                write_json(args.output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    families = defaultdict(list)
    for row in results:
        families[row["family_id"]].append(row)
    for family, rows in families.items():
        # Teacher disagreements first, with confident disagreements prioritized;
        # then difficult near-ties. Round-robin avoids one family taking all slots.
        rows.sort(key=lambda row: (row["agrees_with_existing_teacher_label"],
                                  row["action_margin"] if row["agrees_with_existing_teacher_label"] else -row["action_margin"],
                                  hashlib.sha256(row["id"].encode()).hexdigest()))
        families[family] = deque(rows)
    chosen = []
    while len(chosen) < min(args.count, len(pool)):
        for family in sorted(families):
            if families[family]:
                chosen.append(families[family].popleft())
            if len(chosen) == min(args.count, len(pool)):
                break
    mapping = {episode["id"]: episode for episode in pool}
    # Proposals preserve every teacher label verbatim. The reviewer must construct
    # blind payloads from the original observable features, excluding labels and
    # this sidecar's student disagreement/margin information.
    with (args.output / "proposals.jsonl").open("x") as stream:
        for row in chosen:
            stream.write(canonical(mapping[row["id"]]).decode() + "\n")
    selection = {"status": "requires_blind_teacher_review_not_training_ready", "proposals": len(chosen),
                 "requested_proposals": args.count, "needs_more_pool": len(chosen) < args.count,
                 "categories": dict(Counter(row["category"] for row in chosen)),
                 "families": dict(Counter(row["family_id"] for row in chosen)), "selected": chosen,
                 "label_edits": 0, "gpu_exclusive_confirmation": args.gpu_exclusive_confirmation,
                 "proposals_sha256": sha256_file(args.output / "proposals.jsonl")}
    write_json(args.output / "selection.json", selection)
    print(json.dumps({key: value for key, value in selection.items() if key != "selected"}), flush=True)


if __name__ == "__main__":
    main()
