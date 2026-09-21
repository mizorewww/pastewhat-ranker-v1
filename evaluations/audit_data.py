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
from data_tools.labeling import derive_candidate_label
from evaluations.common import load_jsonl, require_plan_data_path, sha256, validate_formal_heldout_allocation, validate_label, write_json
from evaluations.generate import content_fingerprint, generation_specs, passed_current_gates, normalize_generated, LABEL_SYSTEM, FAMILY_SYSTEM, same_label, reason_provenance
from evaluations.observations import assigned_specs, load_assignment, validate_allocation as validate_observation_allocation
from pastewhat_ranker.preprocess import Preprocessor
from run_contract import RunPlan, action_quotas, family_quotas, load_run_plan


def label_equal(left: dict, right: dict) -> bool:
    return same_label(left, right)


def unique_records(items: list[dict], count: int) -> dict:
    mapping = {row["id"]: row for row in items}
    if len(mapping) != len(items) or len(mapping) != count:
        raise ValueError("Teacher response has duplicate, missing, or extra records")
    return mapping


def audit_dataset(data: Path, audit_root: Path, tokenizer: Path, partition: Path, run_plan: RunPlan | None = None) -> dict:
    episodes = load_jsonl(data)
    split_values = {row.get("split") for row in episodes}
    if len(split_values) != 1 or not split_values <= {"calibration", "test"}:
        raise ValueError("Evaluator audit owns only Calibration or Test")
    split = next(iter(split_values))
    specification = json.loads(partition.read_text())
    families = {row["id"]: row for row in specification["families"][split]}
    partition_hash = sha256(partition)
    allocation = None
    planned_specs = {}
    observation_allocation = None
    if run_plan:
        require_plan_data_path(run_plan, split, data)
        allocation = validate_formal_heldout_allocation(episodes, split, specification, run_plan)
        counts = family_quotas(specification, split, run_plan.target(split))
        actions = action_quotas(counts)
        observation_assignment = load_assignment(run_plan.binding(), split)
        observation_allocation = validate_observation_allocation(episodes, run_plan.binding(), split)
        for index, family in enumerate(specification["families"][split]):
            planned_specs[family["id"]] = assigned_specs(generation_specs(index, 0, counts[family["id"]], counts[family["id"]],
                family_id=family["id"], actions=actions[family["id"]], seed_namespace=run_plan.run_id), family["id"], observation_assignment)
    preprocessor = Preprocessor(tokenizer)
    audit_files = {}

    @lru_cache(maxsize=None)
    def read_audit(identifier: str, expected_system: str | None = None):
        if not re.fullmatch(r"[0-9a-f]{64}", identifier):
            raise ValueError("Malformed teacher audit identifier")
        path = audit_root / split / (identifier + ".json")
        raw = json.loads(path.read_text())
        if raw.get("status") != "success":
            raise ValueError("An accepted row references an unsuccessful teacher response")
        expected = hashlib.sha256(canonical_bytes({"endpoint": raw["endpoint"], "body": raw["request"]})).hexdigest()
        if expected != identifier or hashlib.sha256(canonical_bytes(raw["request"])).hexdigest() != raw["request_sha256"]:
            raise ValueError("Teacher request provenance hash mismatch")
        if expected_system is not None and raw["request"]["messages"][0]["content"] != expected_system:
            raise ValueError("Teacher response used a different labeling or deployment-review protocol")
        result = TeacherClient._result(raw, cache_hit=True)
        request = json.loads(raw["request"]["messages"][1]["content"])
        audit_files[str(path)] = sha256(path)
        return request, result.parsed

    fingerprints = set()
    counts = Counter()
    for episode in episodes:
        validate_label(episode)
        if not passed_current_gates(episode):
            raise ValueError("Episode does not pass current literal-paste, blind-family, and quota gates")
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
        if run_plan and metadata["generator_spec"] != planned_specs[episode["family_id"]][slot]:
            raise ValueError("Generated slot differs from registered independent sampling plan")
        requested_specs = [row for row in generator_request["specs"] if row["slot"] == slot]
        if requested_specs != [metadata["generator_spec"]]:
            raise ValueError("Authoring audit does not contain the exact saved sampling specification")
        originals = [row for row in generator_response["episodes"] if row["slot"] == slot]
        if len(originals) != 1:
            raise ValueError("Generated slot is missing or duplicated")
        if episode["teacher"].get("author_count_repaired_before_labels"):
            parent_id = episode["teacher"]["original_generation_audit_id"]
            parent_request, parent_response = read_audit(parent_id)
            parent_rows = [row for row in parent_response["episodes"] if row["slot"] == slot]
            if (len(parent_rows) != 1 or generator_request.get("parent_authoring_audit_id") != parent_id or
                    [row for row in parent_request["specs"] if row["slot"] == slot] != [metadata["generator_spec"]] or
                    generator_request.get("draft") != parent_rows[0] or
                    generator_request.get("required_candidate_count") != metadata["generator_spec"]["candidate_count"] or
                    any(originals[0].get(key) != parent_rows[0].get(key) for key in ("slot", "guidance", "selected"))):
                raise ValueError("Unlabelled author-count repair lineage or unchanged context cannot be replayed")
        recreated = normalize_generated(json.loads(json.dumps(originals[0])), metadata["generator_spec"], split,
                                       families[episode["family_id"]], partition_hash, preprocessor,
                                       run_plan.binding() if run_plan else None)
        if recreated["synthetic_metadata"]["raw_capture_sha256"] != metadata.get("raw_capture_sha256"):
            raise ValueError("Raw observable capture differs from its authoring audit")
        if recreated["synthetic_metadata"]["raw_candidate_payloads_sha256"] != metadata.get("raw_candidate_payloads_sha256"):
            raise ValueError("Native candidate payload fixtures differ from their authoring audit")
        if recreated["synthetic_metadata"]["candidate_projection_provenance_sha256"] != metadata.get("candidate_projection_provenance_sha256"):
            raise ValueError("Native candidate projection provenance changed")
        if recreated["synthetic_metadata"]["raw_compact_authoring_sha256"] != metadata.get("raw_compact_authoring_sha256"):
            raise ValueError("Compact literal authoring differs from its teacher generation audit")
        for key in ("observation_variant", "observation_protocol", "supplement_sha256", "assignment_sha256", "observation_adapter_sha256"):
            if recreated["synthetic_metadata"].get(key) != metadata.get(key):
                raise ValueError("Registered observation-view provenance cannot be replayed")
        if recreated["preprocessing"]["visible_sha256"] != visible_hash:
            raise ValueError("Generation, native projection, and preprocessing do not reproduce labeled input")
        label_request, label_response = read_audit(episode["teacher"]["label_audit_id"], LABEL_SYSTEM)
        visible = {"context": episode["context"], "entries": episode["entries"]}
        matches = [row for row in label_request["episodes"] if {"context": row["context"], "entries": row["entries"]} == visible]
        if len(matches) != 1:
            raise ValueError("Cannot identify exact prepared input in the blind teacher request")
        opaque_id = matches[0]["id"]
        if not re.fullmatch(r"e[0-9]+", opaque_id):
            raise ValueError("A label request exposed a semantic episode identifier")
        labels = unique_records(label_response["labels"], len(label_request["episodes"]))
        first_label = derive_candidate_label(labels[opaque_id], matches[0])
        if set(labels) != {row["id"] for row in label_request["episodes"]} or not label_equal(first_label, episode["label"]):
            raise ValueError("Stored label differs from its independent teacher response")
        for row in label_request["episodes"]:
            if set(row) != {"id", "context", "entries"}:
                raise ValueError("Teacher label request contains hidden task metadata")
        blind_request, blind_response = read_audit(episode["teacher"]["blind_label_audit_id"], LABEL_SYSTEM)
        blind_inputs = unique_records(blind_request["episodes"], len(label_request["episodes"]))
        blind_labels = unique_records(blind_response["labels"], len(blind_inputs))
        shuffled = list(episode["entries"])
        random.Random(episode["id"] + ":blind-label").shuffle(shuffled)
        ids = {f"item_{index + 1}": row["id"] for index, row in enumerate(shuffled)}
        expected_blind = [{**row, "id": f"item_{index + 1}"} for index, row in enumerate(shuffled)]
        if blind_inputs[opaque_id]["entries"] != expected_blind or blind_inputs[opaque_id]["context"] != episode["context"]:
            raise ValueError("Second blind labeling did not use the same visible evidence with remapped IDs/order")
        other = derive_candidate_label(blind_labels[opaque_id], blind_inputs[opaque_id])
        other["acceptable_ids"] = [ids[value] for value in other["acceptable_ids"]]
        if not label_equal(other, episode["label"]):
            raise ValueError("Blind teacher label passes disagree")
        expected_reasons = reason_provenance(episode["label"], first_label, other)
        if any(episode["teacher"].get(key) != value for key, value in expected_reasons.items()):
            raise ValueError("Pooled abstention reason provenance differs from the teacher responses")
        if episode["teacher"].get("review_protocol") != "blind-family-and-deployment-v1":
            audit_request, audit_response = read_audit(episode["teacher"]["review_audit_id"])
            reviews = unique_records(audit_response["reviews"], len(audit_request["episodes"]))
            review = reviews[episode["id"]]
            if not (review["family_ok"] and review["input_realistic"] and review["agrees"] and label_equal(review["label"], episode["label"])):
                raise ValueError("Legacy accepted episode did not pass its original review")
        family_request, family_response = read_audit(episode["teacher"]["family_classification_audit_id"], FAMILY_SYSTEM)
        if set(family_request) != {"taxonomy", "episodes"}:
            raise ValueError("Blind family classifier received extra target metadata")
        expected_taxonomy = [family for values in specification["families"].values() for family in values]
        if family_request["taxonomy"] != expected_taxonomy:
            raise ValueError("Family classifier used a different taxonomy")
        family_inputs = [row for row in family_request["episodes"] if {"context": row["context"], "entries": row["entries"]} == visible]
        if len(family_inputs) != 1 or not re.fullmatch(r"e[0-9]+", family_inputs[0]["id"]):
            raise ValueError("Cannot match visible input to opaque blind family classification")
        classifications = unique_records(family_response["classifications"], len(family_request["episodes"]))
        classification = classifications[family_inputs[0]["id"]]
        if (classification["observed_family_id"] != episode["family_id"] or classification.get("secondary_family_ids") or
                classification.get("input_realistic") is not True):
            raise ValueError("Blind observed-operation classification or deployment review failed")
        counts[episode["label"]["decision"] if episode["label"]["decision"] == "select" else episode["label"]["abstain_reason"]] += 1
    if run_plan:
        run_plan.verify_unchanged()
    return {"passed": True, "split": split, "episodes": len(episodes), "data_sha256": sha256(data),
            "formal_run": run_plan is not None, "registered_allocation": allocation,
            "observation_supplement": observation_allocation,
            **(run_plan.binding() if run_plan else {}),
            "partition_sha256": partition_hash, "label_counts": dict(counts),
            "pooled_ambiguous_insufficient_reason_disagreements": sum(row["teacher"].get("reason_agreement") is False for row in episodes),
            "teacher_audit_files": len(audit_files), "teacher_audit_bundle_sha256": hashlib.sha256(canonical_bytes(audit_files)).hexdigest(),
            "checks": ["exact production preprocessing", "native Swift UTF-16 selection/nearby projection replay", "native Swift candidate payload projection replay", "raw capture/payload hashes and authoring contracts", "token budget and full candidate preservation",
                       "compact authoring/profile replay", "opaque label-request identifiers", "label-request metadata exclusion", "complete independent per-candidate verdicts and exact selected-text mapping", "two blind labels with remapped IDs and order",
                       "blind observed-operation classification and deployment review", "actual-label sampling quotas",
                       "within-split order/ID-independent duplicate detection", "registered pre-label observation variants and raw author replay"],
            "human_validated": False, "student_inference_used": False,
            "limitations": "Structural and teacher-consistency checks do not prove every semantic label is correct."}


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-plan", type=Path)
    mode.add_argument("--staging", action="store_true", help="Audit an unregistered non-scored authoring probe")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=Path("local/teacher"))
    parser.add_argument("--tokenizer", type=Path, default=Path("../laya-mlx/models/laya-multilingual/tokenizer"))
    parser.add_argument("--partition", type=Path, default=Path("data_tools/family_partition.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_dataset(args.data, args.audit_root, args.tokenizer, args.partition,
                           load_run_plan(args.run_plan) if args.run_plan else None)
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
