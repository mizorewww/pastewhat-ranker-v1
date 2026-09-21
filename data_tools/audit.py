"""Review Train/Dev provenance, family adherence and model-visible invariants.

This performs teacher-assisted review, not human validation. Reviewed snapshots
exclude semantic audit failures without editing their teacher decision labels.
The evaluation agent owns analogous reviews for Calibration/Test.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random

from pastewhat_ranker.preprocess import Preprocessor
from data_tools.generate import PARTITION_PATH, ROOT, validate_labels
from data_tools.teacher import TeacherClient, atomic_json, canonical_bytes, sha256, utc_now


AUDIT_SYSTEM = """You audit synthetic clipboard episodes for a grouped machine-learning dataset.
Return only {"reviews":[{"id":"e1","within_family":true,"deployment_visible":true,
"payload_metadata_consistent":true,"reason":"short finding"}]}.
The supplied operation family is the entire allowed semantic scope. Its exclusions
reserve other operation families for held-out evaluation. All paraphrases, entity
substitutions and counterfactuals remain in their operation family. Reject an
episode whose primary decision requires an excluded or different operation. A few
unrelated clipboard distractors are fine; do not reject those alone. A vague
context intentionally testing abstention can still belong to the family when its
plausible candidate intentions are within it. Do not use teacher labels (none are
provided). Judge the whole context-and-candidate problem, not keywords alone.
deployment_visible means only realistic focused-field information, user-selected
or surrounding text, app CATEGORY, observable candidate text and actual payload
metadata are used. No synthetic hidden-goal field, answer key, or secret fact.
payload_metadata_consistent means text describing a file is not falsely given a
file payload, plain strings are not images, image summaries do not claim unseen
semantic contents. Genuine file payloads and observable image dimensions are valid.
Do not return chain-of-thought. Short reason is a finding for audit, not training.
"""


def review_group(group, family, client):
    visible = [{"id": f"e{i+1}", "context": episode["context"], "entries": episode["entries"]} for i, episode in enumerate(group)]
    result = client.complete_json(AUDIT_SYSTEM, json.dumps({"family": family, "episodes": visible}, ensure_ascii=False), phase="semantic-family-review", request_id=sha256(canonical_bytes(visible)), max_tokens=8192)
    items = result.parsed.get("reviews", [])
    by_id = {item.get("id"): item for item in items}
    if len(items) != len(group) or set(by_id) != {item["id"] for item in visible}:
        raise ValueError("Family review IDs do not match")
    output = []
    for i, episode in enumerate(group):
        review = by_id[f"e{i+1}"]
        flags = ("within_family", "deployment_visible", "payload_metadata_consistent")
        if any(type(review.get(key)) is not bool for key in flags):
            raise ValueError("Family reviewer returned nonboolean status")
        output.append({"id": episode["id"], "family_id": episode["family_id"], "visible_sha256": episode["preprocessing"]["visible_sha256"], "accepted": all(review[key] for key in flags), "review": review, "audit_id": result.audit_id, "teacher_model": result.model})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--tokenizer", default=str(ROOT.parent / "laya-mlx/models/laya-multilingual/tokenizer"))
    parser.add_argument("--snapshot", help="Optional immutable snapshot name, such as train_pilot")
    args = parser.parse_args()
    if args.snapshot and not args.snapshot.startswith(args.split + "_"):
        raise SystemExit("Snapshot name must start with the owning split")
    data_path = ROOT / "data" / f"{args.split}.jsonl"
    source_bytes = data_path.read_bytes()
    episodes = [json.loads(line) for line in source_bytes.splitlines()]
    partition = json.loads(PARTITION_PATH.read_text())
    family_map = {family["id"]: family for family in partition["families"][args.split]}
    preprocessor = Preprocessor(args.tokenizer)
    groups = {}
    for episode in episodes:
        if episode["family_id"] not in family_map:
            raise ValueError("Cross-split conceptual family detected")
        prepared = preprocessor.prepare_episode(episode)
        if canonical_bytes(prepared["context"]) != canonical_bytes(episode["context"]) or canonical_bytes(prepared["entries"]) != canonical_bytes(episode["entries"]):
            raise ValueError("Teacher-labeled view is not preprocessing-idempotent")
        if prepared["preprocessing"]["visible_sha256"] != episode["provenance"]["label_visible_sha256"]:
            raise ValueError("Label refers to a different visible input")
        if any(len(sequence) > 1024 for sequence in preprocessor.encode_episode(episode)["input_ids"]):
            raise ValueError("Pair token budget violated")
        validate_labels({"labels": [{"id": episode["id"], "label": episode["label"]}]}, [episode])
        groups.setdefault(episode["family_id"], []).append(episode)
    client = TeacherClient(ROOT / "local" / "teacher" / args.split / "review")
    tasks = []
    for family_id, group in groups.items():
        group.sort(key=lambda episode: episode["id"])
        tasks.extend((group[i:i + args.batch_size], family_map[family_id]) for i in range(0, len(group), args.batch_size))
    reviews = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(review_group, group, family, client) for group, family in tasks]
        for future in as_completed(futures):
            reviews.extend(future.result())
            print(json.dumps({"split": args.split, "reviewed": len(reviews), "total": len(episodes), "accepted": sum(review["accepted"] for review in reviews)}), flush=True)
    by_id = {review["id"]: review for review in reviews}
    payload = b"".join(canonical_bytes(review) + b"\n" for review in sorted(reviews, key=lambda item: item["id"]))
    review_path = ROOT / "data" / f"{args.split}.review.jsonl"
    review_path.write_bytes(payload)
    accepted = []
    for episode in episodes:
        if by_id[episode["id"]]["accepted"]:
            episode["provenance"]["family_review_audit_id"] = by_id[episode["id"]]["audit_id"]
            episode["provenance"]["review"] = "teacher-labeled and independently teacher-reviewed; programmatically validated; not human validated"
            accepted.append(episode)
    report = {"split": args.split, "created_at": utc_now(), "source_sha256": sha256(source_bytes), "review_sha256": sha256(payload), "source_episodes": len(episodes), "accepted_episodes": len(accepted), "rejected_episodes": len(episodes) - len(accepted), "families": dict(Counter(episode["family_id"] for episode in accepted)), "checks": ["split family ownership", "production preprocessing idempotence", "teacher/student visible hash equality", "every pair <=1024 tokens", "all 1–20 candidates retained", "label IDs and actions", "independent teacher family/deployment/payload review"], "human_validated": False}
    if args.snapshot:
        snapshot = ROOT / "data" / f"{args.snapshot}.jsonl"
        data = b"".join(canonical_bytes(episode) + b"\n" for episode in accepted)
        if snapshot.is_file() and snapshot.read_bytes() != data:
            raise ValueError("Immutable reviewed snapshot already exists with other contents")
        snapshot.write_bytes(data)
        report.update(snapshot=str(snapshot.relative_to(ROOT)), snapshot_sha256=sha256(data))
    atomic_json(ROOT / "data" / f"{args.snapshot or args.split}.review.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
