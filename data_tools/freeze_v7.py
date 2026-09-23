"""Publish immutable v7 Train/Dev stages, preserving nested learning-curve sets."""
from __future__ import annotations

import json
from collections import Counter

from data_tools.content import content_fingerprint
from data_tools.freeze import choose_registered, publish_bytes
from data_tools.teacher import (
    TeacherClient,
    atomic_json,
    canonical_bytes,
    sha256,
    teacher_source_counts,
    utc_now,
    verify_audit_identity,
)
from data_tools.v7 import (
    PROTOCOL,
    ROOT,
    compact_author_fixture,
    prepare_author_batch,
    validate_labels,
)


def verify_author_gate(row, spec, author_audit, preprocessor):
    """Replay a versioned Train/Dev author fixture and gate without the teacher."""
    from data_tools.generate_v7 import (
        DEV_MISSING_INTENT_GATE_VERSION,
        DEV_UNOBSERVED_INTENT_GATE_VERSION,
        TRAIN_MISSING_INTENT_GATE_VERSION,
        dev_missing_intent_prelabel_gate,
    )

    if spec.get("source_constraint_version") not in {DEV_MISSING_INTENT_GATE_VERSION, DEV_UNOBSERVED_INTENT_GATE_VERSION, TRAIN_MISSING_INTENT_GATE_VERSION}:
        return
    if row["provenance"]["source_spec_sha256"] != sha256(canonical_bytes(spec)):
        raise ValueError("Accepted row differs from its registered source specification")
    request = json.loads(author_audit["request"]["messages"][1]["content"])
    if request["mother_task"] != spec["mother_task"] or request["field_profile"] != spec["profile"]:
        raise ValueError("Author request differs from its registered source")
    planned = {item["id"]: item for item in spec["plans"]}
    if any(planned.get(item["id"]) != item for item in request["plans"]):
        raise ValueError("Author request changed a registered plan")
    authored = TeacherClient._result(author_audit, cache_hit=True).parsed
    prepared, _ = prepare_author_batch(compact_author_fixture(authored, spec), request["plans"], spec["profile"], preprocessor)
    prepared, _ = dev_missing_intent_prelabel_gate(authored, spec, request["plans"], prepared)
    replayed = {item["id"]: item for item in prepared}.get(row["id"])
    if replayed is None or any(replayed[key] != row[key] for key in ("context", "entries", "preprocessing")):
        raise ValueError("Accepted row fails independent author fixture and pre-label gate replay")


def try_freeze(plan, split, rows, *, preprocessor):
    partition_path = ROOT / "data_tools/family_partition.json"
    partition = json.loads(partition_path.read_text())
    stages = [("dev", plan.target("dev"))] if split == "dev" else [("throughput", 1000), ("pilot", plan.document["pilot_episodes"]), ("diagnostic", plan.document["diagnostic_episodes"]), ("train", plan.target("train"))]
    required = []
    results = []
    by_id = {row["id"]: row for row in rows}
    gated_specs = {}
    if len(rows) >= plan.target(split):
        allowed = ({"dev-missing-intent-required-parameter-v2", "dev-missing-intent-unobserved-v3"}
                   if split == "dev" else {"train-missing-intent-required-parameter-v1"})
        for path in (ROOT / "local/v7" / plan.run_id / "batches" / split).glob("*.json"):
            record = json.loads(path.read_text())
            if record["spec"].get("source_constraint_version") in allowed:
                for accepted in record["accepted"]:
                    gated_specs[accepted["id"]] = record["spec"]
    author_audits = {}
    for stage, count in stages:
        output = (ROOT / plan.data_path("pilot")).with_name("throughput-train.jsonl") if stage == "throughput" else ROOT / plan.data_path(stage)
        if stage == "throughput" and not output.exists() and (ROOT / plan.data_path("pilot")).exists():
            # A late optional subset must not bypass the already frozen pilot.
            continue
        if output.exists():
            frozen = [json.loads(line) for line in output.read_bytes().splitlines()]
            if any(row["id"] not in by_id or canonical_bytes(row) != canonical_bytes(by_id[row["id"]]) for row in frozen):
                raise ValueError("Immutable earlier snapshot disappeared or changed")
            if not set(required).issubset(row["id"] for row in frozen):
                raise ValueError("Existing later snapshot omits an earlier frozen row")
            required = [row["id"] for row in frozen]
            continue
        if len(rows) < count:
            break
        try:
            chosen = choose_registered(rows, partition, split, count, required_ids=required)
        except ValueError:
            if stage == "throughput":
                continue
            break
        throughput_coverage = None
        if stage == "throughput":
            family_ids = {row["id"] for row in partition["families"]["train"]}
            counts = {len(row["entries"]) for row in chosen}
            if {row["family_id"] for row in chosen} != family_ids or counts != set(range(1, 21)):
                continue
            max_pair_tokens = max(len(pair) for row in chosen for pair in preprocessor.encode_episode(row)["input_ids"])
            if max_pair_tokens < 512:
                continue
            throughput_coverage = {"families": len(family_ids), "candidate_counts": sorted(counts), "max_pair_tokens": max_pair_tokens}
        fingerprints = []
        audits = set()
        for row in chosen:
            provenance = row["provenance"]
            if any(provenance.get(key) != value for key, value in plan.binding().items()) or provenance.get("teacher_contract_version") != PROTOCOL:
                raise ValueError("A snapshot row belongs to another run or teacher contract")
            prepared = preprocessor.prepare_episode(row)
            if prepared["context"] != row["context"] or prepared["entries"] != row["entries"] or prepared["preprocessing"]["visible_sha256"] != provenance["visible_sha256"]:
                raise ValueError("Frozen features differ from the teacher-visible budget")
            validate_labels({"labels": [{"id": row["id"], **row["label"]}]}, [row])
            if row["id"] in gated_specs:
                author_id = provenance["author_audit_id"]
                if author_id not in author_audits:
                    path = ROOT / "local/v7" / plan.run_id / "teacher" / split / (author_id + ".json")
                    author_audits[author_id] = json.loads(path.read_text())
                verify_author_gate(row, gated_specs[row["id"]], author_audits[author_id], preprocessor)
            fingerprint = content_fingerprint(row)
            fingerprints.append(fingerprint)
            for key in ("author_audit_id", "label_audit_id", "review_audit_id"):
                if provenance.get(key):
                    audits.add(provenance[key])
            if provenance["quality_path"] != "single_pass" and not provenance.get("review_audit_id"):
                raise ValueError("Reviewed provenance is missing its independent label audit")
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("Duplicate visible content in a frozen stage")
        audit_directory = ROOT / "local/v7" / plan.run_id / "teacher" / split
        audit_hashes = {identifier: sha256((audit_directory / (identifier + ".json")).read_bytes()) for identifier in sorted(audits)}
        evidence_hashes = {}
        for identifier in sorted(audits):
            audit = json.loads((audit_directory / (identifier + ".json")).read_text())
            verify_audit_identity(audit, identifier)
            if audit.get("transport") == "pi-cli-json":
                files = audit["source_files"] + audit["raw_event_files"] + [audit["receipt_file"], audit["bound_request_file"]]
                for evidence in files:
                    evidence_path = ROOT / evidence["path"]
                    actual = sha256(evidence_path.read_bytes())
                    if actual != evidence["sha256"]:
                        raise ValueError("Pi source/request/event evidence changed before freeze")
                    evidence_hashes[evidence["path"]] = actual
        chosen.sort(key=lambda row: sha256(row["id"].encode()))
        payload = b"".join(canonical_bytes(row) + b"\n" for row in chosen)
        output.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_path = output.with_suffix(".fingerprints.json")
        atomic_json(fingerprint_path, {**plan.binding(), "content_sha256": sorted(fingerprints)})
        manifest = {**plan.binding(), "split": split, "stage": stage, "episodes": count, "sha256": sha256(payload), "family_partition_sha256": sha256(partition_path.read_bytes()), "teacher_contract_version": PROTOCOL, "native_projection_sha256": plan.document["projection_provenance_sha256"], "preprocess_sha256": sha256(canonical_bytes(preprocessor.manifest())), "fingerprints_sha256": sha256(fingerprint_path.read_bytes()), "teacher_audit_file_sha256": audit_hashes, "contains_earlier_snapshot_ids": required, "quality_paths": dict(Counter(row["provenance"]["quality_path"] for row in chosen)), "created_at": utc_now(), "human_validated": False}
        if throughput_coverage:
            manifest["throughput_coverage"] = throughput_coverage
        manifest["teacher_sources"] = teacher_source_counts(chosen, audit_directory)
        transition = ROOT / "configs/teacher_transition_swe2.json"
        manifest["teacher_transition"] = {"path": str(transition.relative_to(ROOT)), "sha256": sha256(transition.read_bytes())}
        correction = ROOT / "configs/teacher_correction_swe2_uid.json"
        if correction.is_file():
            manifest["teacher_correction"] = {"path": str(correction.relative_to(ROOT)), "sha256": sha256(correction.read_bytes())}
        manifest["teacher_evidence_file_sha256"] = evidence_hashes
        atomic_json(output.with_suffix(".manifest.json"), manifest)
        # JSONL is the readiness marker consumed by the GPU pipeline.
        publish_bytes(output, payload)
        required = [row["id"] for row in chosen]
        results.append({"stage": stage, "episodes": count, "path": str(output), "sha256": manifest["sha256"]})
    return results
