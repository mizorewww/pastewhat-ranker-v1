"""Check held-out provenance and teacher visibility without running a student."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re

from data_tools.teacher import TeacherClient, canonical_bytes
from evaluations.common import load_jsonl, sha256, validate_label, write_json
from evaluations.generate import content_fingerprint, normalize_generated
from pastewhat_ranker.preprocess import Preprocessor


def label_equal(left: dict, right: dict) -> bool:
    return (left["decision"] == right["decision"] and
            set(left["acceptable_ids"]) == set(right["acceptable_ids"]) and
            left.get("abstain_reason") == right.get("abstain_reason"))


def unique_records(items: list[dict], count: int) -> dict:
    mapping = {row["id"]: row for row in items}
    if len(mapping) != len(items) or len(mapping) != count:
        raise ValueError("Teacher response has duplicate, missing, or extra records")
    return mapping


def audit_dataset(data: Path, audit_root: Path, tokenizer: Path, partition: Path) -> dict:
    episodes = load_jsonl(data)
    split_values = {row.get("split") for row in episodes}
    if len(split_values) != 1 or not split_values <= {"calibration", "test"}:
        raise ValueError("Evaluator audit owns only Calibration or Test")
    split = next(iter(split_values))
    specification = json.loads(partition.read_text())
    families = {row["id"]: row for row in specification["families"][split]}
    partition_hash = sha256(partition)
    preprocessor = Preprocessor(tokenizer)
    audit_files = {}

    @lru_cache(maxsize=None)
    def read_audit(identifier: str):
        if not re.fullmatch(r"[0-9a-f]{64}", identifier):
            raise ValueError("Malformed teacher audit identifier")
        path = audit_root / split / (identifier + ".json")
        raw = json.loads(path.read_text())
        if raw.get("status") != "success":
            raise ValueError("An accepted row references an unsuccessful teacher response")
        expected = hashlib.sha256(canonical_bytes({"endpoint": raw["endpoint"], "body": raw["request"]})).hexdigest()
        if expected != identifier or hashlib.sha256(canonical_bytes(raw["request"])).hexdigest() != raw["request_sha256"]:
            raise ValueError("Teacher request provenance hash mismatch")
        result = TeacherClient._result(raw, cache_hit=True)
        request = json.loads(raw["request"]["messages"][1]["content"])
        audit_files[str(path)] = sha256(path)
        return request, result.parsed

    fingerprints = set()
    counts = Counter()
    for episode in episodes:
        validate_label(episode)
        if episode["family_id"] not in families:
            raise ValueError("Episode crosses the preregistered conceptual-family partition")
        metadata = episode["synthetic_metadata"]
        if metadata["family_partition_sha256"] != partition_hash:
            raise ValueError("Episode family partition changed")
        prepared = preprocessor.prepare_episode(episode)
        if prepared["context"] != episode["context"] or prepared["entries"] != episode["entries"]:
            raise ValueError("Prepared data changes under production preprocessing")
        visible_hash = prepared["preprocessing"]["visible_sha256"]
        if visible_hash != episode["teacher"]["visible_sha256"]:
            raise ValueError("Teacher and deployed model do not see identical input")
        encoded = preprocessor.encode_episode(episode)
        if not 1 <= len(encoded["input_ids"]) <= 20 or any(len(ids) > 1024 for ids in encoded["input_ids"]):
            raise ValueError("Token budget or full-candidate count violated")
        fingerprint = content_fingerprint(episode)
        if fingerprint in fingerprints:
            raise ValueError("Duplicate visible episode, ignoring candidate IDs and order")
        fingerprints.add(fingerprint)
        generator_request, generator_response = read_audit(episode["teacher"]["generation_audit_id"])
        slot = metadata["generator_spec"]["slot"]
        originals = [row for row in generator_response["episodes"] if row["slot"] == slot]
        if len(originals) != 1:
            raise ValueError("Generated slot is missing or duplicated")
        recreated = normalize_generated(json.loads(json.dumps(originals[0])), metadata["generator_spec"], split,
                                       families[episode["family_id"]], partition_hash, preprocessor)
        if recreated["preprocessing"]["visible_sha256"] != visible_hash:
            raise ValueError("Generation, native projection, and preprocessing do not reproduce labeled input")
        label_request, label_response = read_audit(episode["teacher"]["label_audit_id"])
        visible = {"context": episode["context"], "entries": episode["entries"]}
        matches = [row for row in label_request["episodes"] if {"context": row["context"], "entries": row["entries"]} == visible]
        if len(matches) != 1:
            raise ValueError("Cannot identify exact prepared input in the blind teacher request")
        opaque_id = matches[0]["id"]
        if not re.fullmatch(r"e[0-9]+", opaque_id):
            raise ValueError("A label request exposed a semantic episode identifier")
        labels = unique_records(label_response["labels"], len(label_request["episodes"]))
        if set(labels) != {row["id"] for row in label_request["episodes"]} or not label_equal(labels[opaque_id]["label"], episode["label"]):
            raise ValueError("Stored label differs from its independent teacher response")
        for row in label_request["episodes"]:
            if set(row) != {"id", "context", "entries"}:
                raise ValueError("Teacher label request contains hidden task metadata")
        blind_request, blind_response = read_audit(episode["teacher"]["blind_label_audit_id"])
        blind_inputs = unique_records(blind_request["episodes"], len(label_request["episodes"]))
        blind_labels = unique_records(blind_response["labels"], len(blind_inputs))
        shuffled = list(episode["entries"])
        random.Random(episode["id"] + ":blind-label").shuffle(shuffled)
        ids = {f"item_{index + 1}": row["id"] for index, row in enumerate(shuffled)}
        expected_blind = [{**row, "id": f"item_{index + 1}"} for index, row in enumerate(shuffled)]
        if blind_inputs[opaque_id]["entries"] != expected_blind or blind_inputs[opaque_id]["context"] != episode["context"]:
            raise ValueError("Second blind labeling did not use the same visible evidence with remapped IDs/order")
        other = dict(blind_labels[opaque_id]["label"])
        other["acceptable_ids"] = [ids[value] for value in other["acceptable_ids"]]
        if not label_equal(other, episode["label"]):
            raise ValueError("Blind teacher label passes disagree")
        audit_request, audit_response = read_audit(episode["teacher"]["review_audit_id"])
        reviews = unique_records(audit_response["reviews"], len(audit_request["episodes"]))
        review = reviews[episode["id"]]
        if not (review["family_ok"] and review["input_realistic"] and review["agrees"] and label_equal(review["label"], episode["label"])):
            raise ValueError("Accepted episode did not pass independent family/realism review")
        counts[episode["label"]["decision"] if episode["label"]["decision"] == "select" else episode["label"]["abstain_reason"]] += 1
    return {"passed": True, "split": split, "episodes": len(episodes), "data_sha256": sha256(data),
            "partition_sha256": partition_hash, "label_counts": dict(counts),
            "teacher_audit_files": len(audit_files), "teacher_audit_bundle_sha256": hashlib.sha256(canonical_bytes(audit_files)).hexdigest(),
            "checks": ["exact production preprocessing", "native Swift projection replay", "token budget and full candidate preservation",
                       "opaque label-request identifiers", "label-request metadata exclusion", "two blind labels with remapped IDs and order",
                       "independent family/realism review", "within-split order/ID-independent duplicate detection"],
            "human_validated": False, "student_inference_used": False,
            "limitations": "Structural and teacher-consistency checks do not prove every semantic label is correct."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("local/teacher"))
    parser.add_argument("--tokenizer", type=Path, default=Path("../laya-mlx/models/laya-multilingual/tokenizer"))
    parser.add_argument("--partition", type=Path, default=Path("data_tools/family_partition.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_dataset(args.data, args.audit_root, args.tokenizer, args.partition)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
