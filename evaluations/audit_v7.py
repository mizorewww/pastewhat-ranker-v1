"""Independent replay of accepted v7 heldout batches without model inference."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from data_tools.content import content_fingerprint
from data_tools.teacher import TeacherClient, canonical_bytes
from data_tools.v7 import PROTOCOL, prepare_author_batch, same_action, validate_labels
from evaluations.common import load_jsonl, require_plan_data_path, sha256, validate_formal_heldout_allocation
from pastewhat_ranker.preprocess import Preprocessor


def _entry_key(entry):
    return canonical_bytes({key: value for key, value in entry.items() if key != "id"})


class BatchAuditor:
    def __init__(self, *, plan, split, tokenizer):
        self.plan, self.split = plan, split
        self.preprocessor = Preprocessor(tokenizer)
        self.audit_root = Path("local/teacher-v7") / plan.run_id / split
        self.bound_files = {}

    @lru_cache(maxsize=None)
    def teacher(self, identifier):
        path = self.audit_root / (identifier + ".json")
        raw = json.loads(path.read_text())
        if (raw.get("status") != "success" or
                hashlib.sha256(canonical_bytes({"endpoint": raw["endpoint"], "body": raw["request"]})).hexdigest() != identifier or
                hashlib.sha256(canonical_bytes(raw["request"])).hexdigest() != raw["request_sha256"]):
            raise ValueError("Accepted heldout data references an invalid teacher audit")
        self.bound_files[str(path)] = sha256(path)
        return json.loads(raw["request"]["messages"][1]["content"]), TeacherClient._result(raw, cache_hit=True).parsed

    def decision(self, identifier, episode):
        request, response = self.teacher(identifier)
        if set(request) != {"episodes"}:
            raise ValueError("A decision reviewer saw source metadata outside the actual student view")
        for value in request["episodes"]:
            if set(value) != {"id", "context", "entries"}:
                raise ValueError("A decision request contains hidden labels or source metadata")
        matches = [row for row in request["episodes"] if content_fingerprint(row) == content_fingerprint(episode)]
        if len(matches) != 1:
            raise ValueError("Cannot uniquely identify the exact visible episode in its decision audit")
        visible = matches[0]
        choices = [row for row in response["labels"] if row.get("id") == visible["id"]]
        label = validate_labels({"labels": choices}, [visible])[visible["id"]]
        available = defaultdict(list)
        for entry in episode["entries"]:
            available[_entry_key(entry)].append(entry["id"])
        mapping = {}
        for entry in visible["entries"]:
            candidates = available[_entry_key(entry)]
            if not candidates:
                raise ValueError("Teacher candidate content differs from the deployed candidate set")
            mapping[entry["id"]] = candidates.pop()
        if any(available.values()):
            raise ValueError("Teacher decision omitted a deployed candidate")
        return {**label, "acceptable_ids": [mapping[value] for value in label["acceptable_ids"]]}

    @lru_cache(maxsize=None)
    def batch(self, path_string):
        path = Path(path_string)
        record = json.loads(path.read_text())
        spec = record["spec"]
        if (record.get("status") != "complete" or record.get("teacher_contract_version") != PROTOCOL or
                record["spec_sha256"] != hashlib.sha256(canonical_bytes(spec)).hexdigest() or
                any(record.get(key) != value or spec["run_binding"].get(key) != value for key, value in self.plan.binding().items()) or
                record["attempts"] > (1 if spec.get("cached_author_recovery") else 2)):
            raise ValueError("The heldout batch is incomplete, changed or exceeds the bounded author route")
        self.bound_files[str(path)] = sha256(path)
        plans = {plan["id"]: plan for plan in spec["plans"]}
        prepared_by_author = {}
        plans_by_author = {}
        accepted = {}
        for episode in record["accepted"]:
            if episode["id"] in accepted or episode["id"] not in plans:
                raise ValueError("Accepted batch IDs are duplicated or outside the source plan")
            provenance = episode["provenance"]
            if (provenance.get("teacher_contract_version") != PROTOCOL or provenance.get("source_spec_sha256") != record["spec_sha256"] or
                    provenance.get("source_family") != spec["family_id"] or episode["family_id"] != spec["family_id"] or
                    any(provenance.get(key) != value for key, value in self.plan.binding().items())):
                raise ValueError("Heldout row source lineage differs from its accepted batch")
            author_id = provenance["author_audit_id"]
            if author_id not in prepared_by_author:
                request, authored = self.teacher(author_id)
                if request.get("mother_task") != spec["mother_task"] or request.get("field_profile") != spec["profile"]:
                    raise ValueError("Author mother-task/profile source cannot be replayed")
                source_plans = {plan["id"]: plan for plan in request["plans"]}
                if any(source_plans.get(identifier) != plan for identifier, plan in plans.items() if identifier in source_plans):
                    raise ValueError("Author request changed the registered source plan")
                prepared, _ = prepare_author_batch(authored, spec["plans"], spec["profile"], self.preprocessor)
                prepared_by_author[author_id] = {row["id"]: row for row in prepared}
                plans_by_author[author_id] = source_plans
            if plans_by_author[author_id].get(episode["id"]) != plans[episode["id"]]:
                raise ValueError("An accepted draft was not requested with its exact source plan")
            recreated = prepared_by_author[author_id][episode["id"]]
            if (recreated["context"] != episode["context"] or recreated["entries"] != episode["entries"] or
                    recreated["preprocessing"]["visible_sha256"] != provenance["visible_sha256"]):
                raise ValueError("Native projection and budget do not reproduce the exact labeled input")
            if provenance["native_projection_sha256"] != sha256("tools/context_projection/provenance.json"):
                raise ValueError("The native projection contract changed")
            if provenance["preprocess_sha256"] != hashlib.sha256(canonical_bytes(self.preprocessor.manifest())).hexdigest():
                raise ValueError("The student preprocessing contract changed")
            if provenance.get("observation_variant") != plans[episode["id"]].get("observation_variant", "standard"):
                raise ValueError("Heldout observation provenance differs from its source plan")
            if not same_action(self.decision(provenance["label_audit_id"], episode), episode["label"]):
                raise ValueError("The accepted label differs from the independent teacher decision")
            quality = provenance["quality_path"]
            if quality not in {"single_pass", "sampled_reviewed", "risk_reviewed"}:
                raise ValueError("Unknown heldout quality path")
            if (quality != "single_pass") != bool(provenance.get("review_audit_id")):
                raise ValueError("The claimed quality path does not match its decision-review evidence")
            if spec["audit_sample"] and quality != "sampled_reviewed":
                raise ValueError("A predetermined sampled batch skipped independent review")
            if provenance.get("review_audit_id") and not same_action(self.decision(provenance["review_audit_id"], episode), episode["label"]):
                raise ValueError("The independent sampled/risk reviewer disagrees")
            actual = len(episode["entries"])
            if not 1 <= actual <= 20:
                raise ValueError("The complete candidate set violates deployment bounds")
            for key, value in {"planned_candidate_count": plans[episode["id"]]["candidate_count"],
                               "actual_candidate_count": actual, "candidate_count_delta": actual - plans[episode["id"]]["candidate_count"]}.items():
                # Early valid v7 rows predate the count-delta metadata; their
                # full groups are still replayed rather than silently rewritten.
                if key in provenance and provenance[key] != value:
                    raise ValueError("Recorded candidate count provenance is inconsistent")
            accepted[episode["id"]] = episode
        return accepted


def registered_sources(plan, split, partition, auditor):
    from evaluations.generate_v7 import planned_batches
    directory = Path("local/evaluator-v7") / plan.run_id / split
    sampling_path = directory / "sampling.json"
    sampling = json.loads(sampling_path.read_text())
    expected = planned_batches(plan, partition, split, 10)
    if sampling.get("specs") != expected or sampling.get("max_situations") != 8:
        raise ValueError("Heldout sampling differs from its pre-score registration")
    specifications = {}
    for path in [sampling_path, *sorted(directory.glob("replacement-round-*.json"))]:
        registration = json.loads(path.read_text())
        if any(registration.get(key) != value for key, value in plan.binding().items()):
            raise ValueError("Heldout replacement lineage belongs to another run")
        if path != sampling_path and not 1 <= registration.get("round", 0) < 8:
            raise ValueError("Heldout backfill exceeded eight source situations")
        auditor.bound_files[str(path)] = sha256(path)
        for spec in registration["specs"]:
            if spec["batch_id"] in specifications:
                raise ValueError("Repeated source batch registration")
            specifications[spec["batch_id"]] = spec
    initial = {item["id"]: (spec, item) for spec in expected for item in spec["plans"]}
    sources = ("data_tools/v7.py", "data_tools/authoring.py", "data_tools/observations.py", "data_tools/content.py",
               "evaluations/authoring.py", "evaluations/authoring_v7.py", "evaluations/authoring-profiles.json",
               "evaluations/generate_v7.py", "evaluations/produce_v7.py", "tools/project_context.py", "tools/project_candidates.py")
    owner_evidence = [*sorted(directory.glob("owner-review-*.json")), *sorted((directory / "producer-revisions").glob("*.json"))]
    exclusions = Path("local/evaluator-quality-exclusions") / (split + ".json")
    if exclusions.exists():
        owner_evidence.append(exclusions)
    for path in [*(Path(value) for value in sources), directory / "owner-binding.json", *owner_evidence]:
        auditor.bound_files[str(path)] = sha256(path)
    return specifications, initial


def verify_slot(row, specifications, initial_slots, auditor):
    metadata = row["synthetic_metadata"]
    slot = metadata["quota_slot_id"]
    record = json.loads(Path(metadata["batch_record_path"]).read_text())
    spec = record["spec"]
    if "cached_author_recovery" in spec:
        recovery = spec["cached_author_recovery"]
        parent_path = Path(recovery["original_batch_path"])
        if sha256(parent_path) != recovery["original_batch_sha256"] or recovery["additional_author_calls_allowed"] is not False:
            raise ValueError("Cached author recovery changed its original rejected batch")
        parent = json.loads(parent_path.read_text())
        if (any(item.get("id") == row["id"] and "original_label" in item for item in parent["rejected"]) or
                any(item["id"] == row["id"] for item in parent["accepted"]) or
                row["provenance"]["author_audit_id"] != recovery["original_author_audit_id"]):
            raise ValueError("Cached recovery reused an accepted or semantically rejected draft")
        auditor.bound_files[str(parent_path)] = sha256(parent_path)
        registered = specifications[parent["spec"]["batch_id"]]
        if parent["spec"] != registered or any(spec.get(key) != registered[key] for key in ("family_id", "mother_task", "profile", "seed", "run_binding", "audit_sample")):
            raise ValueError("Cached recovery changed the registered author source")
    elif spec != specifications[spec["batch_id"]]:
        raise ValueError("Accepted batch differs from its registered source")
    source = next(item for item in spec["plans"] if item["id"] == row["id"])
    initial_spec, initial = initial_slots[slot]
    if (source.get("quota_slot_id", source["id"]) != slot or spec["family_id"] != initial_spec["family_id"] or
            spec["audit_sample"] != initial_spec["audit_sample"] or spec["profile"] != initial_spec["profile"] or
            any(source.get(key) != initial[key] for key in ("context_language", "scenario_type", "observation_variant", "candidate_count")) or
            metadata.get("requested_context_language") != source["context_language"]):
        raise ValueError("A replacement changed its family, review cohort or registered sampling factor")


def audit_dataset(data, *, plan, split, tokenizer, partition):
    require_plan_data_path(plan, split, data)
    episodes = load_jsonl(data)
    partition_document = json.loads(Path(partition).read_text())
    allocation = validate_formal_heldout_allocation(episodes, split, partition_document, plan)
    auditor = BatchAuditor(plan=plan, split=split, tokenizer=tokenizer)
    specifications, initial_slots = registered_sources(plan, split, partition_document, auditor)
    fingerprints = set()
    qualities, variants, counts = Counter(), Counter(), Counter()
    logical_slots = set()
    exclusion_path = Path("local/evaluator-quality-exclusions") / (split + ".json")
    exclusions = set(json.loads(exclusion_path.read_text())["content_fingerprints"]) if exclusion_path.exists() else set()
    for row in episodes:
        metadata = row["synthetic_metadata"]
        accepted = auditor.batch(metadata["batch_record_path"])
        original = accepted[row["id"]]
        if any(row.get(key) != value for key, value in original.items()):
            raise ValueError("A finalized heldout row changed after its accepted teacher batch")
        if metadata["quota_slot_id"] not in initial_slots or metadata["quota_slot_id"] in logical_slots:
            raise ValueError("Several accepted replacements filled the same registered logical slot")
        logical_slots.add(metadata["quota_slot_id"])
        verify_slot(row, specifications, initial_slots, auditor)
        fingerprint = content_fingerprint(row)
        if fingerprint in fingerprints or fingerprint in exclusions:
            raise ValueError("Duplicate or independently excluded visible content entered the heldout corpus")
        fingerprints.add(fingerprint)
        qualities[row["provenance"]["quality_path"]] += 1
        variants[row["provenance"]["observation_variant"]] += 1
        counts[len(row["entries"])] += 1
    if variants["no_accessibility"] != len(episodes) * 4 // 100 or variants["generic_field"] != len(episodes) * 2 // 100:
        raise ValueError("The initial registered observation allocation changed")
    if logical_slots != set(initial_slots):
        raise ValueError("Final heldout data does not fill every registered logical slot")
    plan.verify_unchanged()
    return {"passed": True, "formal_run": True, **plan.binding(), "split": split,
            "episodes": len(episodes), "data_sha256": sha256(data), "partition_sha256": sha256(partition),
            "teacher_contract_version": PROTOCOL, "registered_allocation": allocation,
            "quality_counts": dict(qualities), "observation_counts": dict(variants), "actual_candidate_counts": dict(counts),
            "teacher_and_batch_files": auditor.bound_files,
            "teacher_audit_bundle_sha256": hashlib.sha256(canonical_bytes(auditor.bound_files)).hexdigest(),
            "human_validated": False, "student_inference_used": False,
            "limitations": "Native/input/label replay proves provenance, not perfect teacher semantic accuracy; only the reported subset has a second blind decision review."}
