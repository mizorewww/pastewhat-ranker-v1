"""Independent replay of accepted v7 heldout batches without model inference."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from data_tools.content import content_fingerprint
from data_tools.teacher import TeacherClient, audit_source, canonical_bytes, verify_audit_identity
from data_tools.v7 import PROTOCOL, compact_author_fixture, prepare_author_batch, same_action, validate_labels
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

    def bind_file(self, record):
        path = Path(record["path"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"] or ("bytes" in record and len(content) != record["bytes"]):
            raise ValueError("A raw teacher evidence file changed")
        self.bound_files[str(path)] = record["sha256"]
        return content

    def verify_pi_evidence(self, raw):
        """Replay the final assistant event; streaming deltas never count twice."""
        from data_tools.pi_teacher import normalize_usage
        from evaluations.freeze import (TEACHER_TRANSITION, TEACHER_TRANSITION_SHA,
                                        TEACHER_CORRECTION, TEACHER_CORRECTION_SHA, SWE_MODEL_MAPPING,
                                        RESOURCE_SUPPLEMENT, RESOURCE_SUPPLEMENT_SHA, teacher_transition_inputs)
        for path in teacher_transition_inputs(self.plan).values():
            self.bound_files[str(path)] = sha256(path)
        transition = json.loads(TEACHER_TRANSITION.read_text())
        correction = json.loads(TEACHER_CORRECTION.read_text())
        corrected = raw.get("teacher_correction")
        runtime = transition["runtime"]
        if corrected:
            if (Path(corrected["path"]).resolve() != TEACHER_CORRECTION.resolve() or
                    corrected["sha256"] != TEACHER_CORRECTION_SHA):
                raise ValueError("The Pi audit claims an unregistered model-identity correction")
            self.bind_file(corrected)
            runtime = correction["runtime"]
            if raw["runtime"].get("teacher_correction_sha256") != TEACHER_CORRECTION_SHA:
                raise ValueError("The corrected Pi call lacks its runtime identity binding")
        elif datetime.fromisoformat(raw["started_at"]) > datetime.fromisoformat(correction["registered_at"]):
            raise ValueError("A new Pi call used the retired unguarded runtime")
        pins = json.loads(Path(runtime["pins_path"]).read_text())
        if raw.get("teacher_transition_sha256") != TEACHER_TRANSITION_SHA:
            raise ValueError("The Pi audit lacks the registered teacher-transition binding")
        if raw.get("resource_supplement"):
            scheduling = raw["resource_supplement"]
            if Path(scheduling["path"]).resolve() != RESOURCE_SUPPLEMENT.resolve() or scheduling["sha256"] != RESOURCE_SUPPLEMENT_SHA:
                raise ValueError("The Pi audit claims a different scheduling supplement")
            self.bind_file(scheduling)
        source_paths = set()
        for record in raw["source_files"]:
            self.bind_file(record)
            source_paths.add(Path(record["path"]).resolve())
        required = {TEACHER_TRANSITION.resolve(), Path(runtime["pins_path"]).resolve(),
                    Path(pins["teacher_extension"]["path"]).resolve()}
        if corrected:
            required.add(TEACHER_CORRECTION.resolve())
        if not required <= source_paths:
            raise ValueError("The Pi audit lacks its pinned bridge and public policy evidence")
        if (raw["provider"] != transition["provider"] or raw["request"]["model"] != transition["model"] or
                any(raw["runtime"].get(key) != runtime[key] for key in
                    ("pi_version", "pi_executable_sha256", "teacher_extension_sha256", "provider_extension_sha256")) or
                raw["runtime"].get("runtime_pins_sha256") != runtime["pins_sha256"] or
                raw["runtime"].get("teacher_transition_sha256") != TEACHER_TRANSITION_SHA):
            raise ValueError("The Pi call used an unregistered provider or runtime")
        events = []
        for record in raw["raw_event_files"]:
            events.extend(json.loads(line) for line in self.bind_file(record).splitlines() if line.strip())
        endings = [event["message"] for event in events
                   if event.get("type") == "message_end" and event.get("message", {}).get("role") == "assistant"]
        if len(endings) != 1:
            raise ValueError("Accepted Pi data requires exactly one final assistant completion")
        receipt = json.loads(self.bind_file(raw["receipt_file"]))
        request = raw["request"]
        bound = {"system": request["messages"][0]["content"], "user": request["messages"][1]["content"],
                 "max_tokens": request["max_tokens"], "thinking": request["reasoning_effort"]}
        if self.bind_file(raw["bound_request_file"]) != canonical_bytes(bound):
            raise ValueError("The Pi process received different bound prompt bytes")
        expected_receipt = {"version": "pastewhat-pi-teacher-receipt-v1", "transport": "pi-cli-json",
            "provider": raw["provider"], "requested_model": request["model"],
            "request_sha256": raw["bound_request_sha256"], "max_tokens": request["max_tokens"],
            "thinking": request["reasoning_effort"], "provider_call_count": 1, "isolated": True,
            "context_message_count": 1, "tools_count": 0,
            "usage_source": "pi-devin-provider-reported-or-unknown"}
        if corrected:
            expected_receipt["exact_uid_guard"] = correction["exact_uid_guard"]
        if receipt != raw["receipt"] or any(receipt.get(key) != value for key, value in expected_receipt.items()):
            raise ValueError("The Pi receipt does not prove the exact isolated request")
        if receipt["actual_model"] != SWE_MODEL_MAPPING.get(request["reasoning_effort"]):
            raise ValueError("The effective Pi model differs from its registered thinking mapping")
        message = endings[0]
        if (message.get("stopReason") != "stop" or message.get("provider") != raw["provider"] or message.get("model") != request["model"] or
                any(item.get("type") == "toolCall" for item in message.get("content", []))):
            raise ValueError("The final Pi message used an unexpected model or tool")
        usage, semantics = normalize_usage(message.get("usage"))
        text = "\n".join(item["text"] for item in message.get("content", []) if item.get("type") == "text")
        normalized = {"model": receipt["actual_model"], "choices": [{"finish_reason": message.get("stopReason"),
            "message": {"role": "assistant", "content": text}}], "usage": usage}
        if (normalized != raw["response"] or hashlib.sha256(canonical_bytes(normalized)).hexdigest() != raw["response_sha256"] or
                message.get("usage") != raw.get("raw_usage") or semantics != raw.get("accounting_semantics")):
            raise ValueError("The normalized Pi result or usage differs from its original final event")

    @lru_cache(maxsize=None)
    def teacher_audit(self, identifier):
        path = self.audit_root / (identifier + ".json")
        raw = json.loads(path.read_text())
        if raw.get("status") != "success":
            raise ValueError("Accepted heldout data references an invalid teacher audit")
        verify_audit_identity(raw, identifier)
        self.bound_files[str(path)] = sha256(path)
        if raw.get("transport") == "pi-cli-json":
            self.verify_pi_evidence(raw)
        elif self.plan.run_id == "ranker-v1-efficient-20260921":
            from evaluations.freeze import TEACHER_TRANSITION
            transition = json.loads(TEACHER_TRANSITION.read_text())
            if datetime.fromisoformat(raw["started_at"]) > datetime.fromisoformat(transition["registered_at"]):
                raise ValueError("A new Kimi request was dispatched after its registered retirement")
        return raw

    @lru_cache(maxsize=None)
    def teacher(self, identifier):
        raw = self.teacher_audit(identifier)
        return json.loads(raw["request"]["messages"][1]["content"]), TeacherClient._result(raw, cache_hit=True).parsed

    def sources(self, episode):
        provenance = episode["provenance"]
        return {role: audit_source(self.teacher_audit(provenance[key])) if provenance.get(key) else None
                for role, key in (("author", "author_audit_id"), ("primary", "label_audit_id"), ("review", "review_audit_id"))}

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
                prepared, _ = prepare_author_batch(compact_author_fixture(authored, spec), spec["plans"], spec["profile"], self.preprocessor)
                if spec.get("source_constraint_version") == "missing-intent-required-parameter-v2":
                    from evaluations.produce_v7 import missing_intent_prelabel_gate
                    prepared, _ = missing_intent_prelabel_gate(authored, spec, spec["plans"], prepared)
                prepared_by_author[author_id] = {row["id"]: row for row in prepared}
                plans_by_author[author_id] = source_plans
            if plans_by_author[author_id].get(episode["id"]) != plans[episode["id"]]:
                raise ValueError("An accepted draft was not requested with its exact source plan")
            if episode["id"] not in prepared_by_author[author_id]:
                raise ValueError("An accepted author fixture fails replay or its registered pre-label gate")
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
            sources = self.sources(episode)
            if sources["primary"]["response_model"] != provenance["teacher_model"]:
                raise ValueError("The recorded teacher model differs from the actual primary response")
            if "teacher_sources" in provenance and provenance["teacher_sources"] != sources:
                raise ValueError("Per-role teacher provenance differs from the actual audits")
            if any(value and value["transport"] == "pi-cli-json" for value in sources.values()) and "teacher_sources" not in provenance:
                raise ValueError("Pi-produced data requires explicit per-role teacher provenance")
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
    from evaluations.freeze import teacher_transition_inputs
    for path in teacher_transition_inputs(plan).values():
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
    from evaluations.freeze import RESOURCE_SUPPLEMENT, TEACHER_CORRECTION
    require_plan_data_path(plan, split, data)
    episodes = load_jsonl(data)
    partition_document = json.loads(Path(partition).read_text())
    allocation = validate_formal_heldout_allocation(episodes, split, partition_document, plan)
    auditor = BatchAuditor(plan=plan, split=split, tokenizer=tokenizer)
    specifications, initial_slots = registered_sources(plan, split, partition_document, auditor)
    fingerprints = set()
    qualities, variants, counts = Counter(), Counter(), Counter()
    source_counts = defaultdict(Counter)
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
        for role, source in auditor.sources(row).items():
            if source:
                source_counts[role][canonical_bytes(source).decode()] += 1
    if variants["no_accessibility"] != len(episodes) * 4 // 100 or variants["generic_field"] != len(episodes) * 2 // 100:
        raise ValueError("The initial registered observation allocation changed")
    if logical_slots != set(initial_slots):
        raise ValueError("Final heldout data does not fill every registered logical slot")
    plan.verify_unchanged()
    return {"passed": True, "formal_run": True, **plan.binding(), "split": split,
            "episodes": len(episodes), "data_sha256": sha256(data), "partition_sha256": sha256(partition),
            "teacher_contract_version": PROTOCOL, "registered_allocation": allocation,
            "resource_supplement": {"path": str(RESOURCE_SUPPLEMENT), "sha256": sha256(RESOURCE_SUPPLEMENT)},
            "teacher_correction": {"path": str(TEACHER_CORRECTION), "sha256": sha256(TEACHER_CORRECTION)},
            "quality_counts": dict(qualities), "observation_counts": dict(variants), "actual_candidate_counts": dict(counts),
            "teacher_sources": {role: [{**json.loads(source), "episodes": count} for source, count in sorted(values.items())]
                                for role, values in sorted(source_counts.items())},
            "teacher_and_batch_files": auditor.bound_files,
            "teacher_audit_bundle_sha256": hashlib.sha256(canonical_bytes(auditor.bound_files)).hexdigest(),
            "human_validated": False, "student_inference_used": False,
            "limitations": "Native/input/label replay proves provenance, not perfect teacher semantic accuracy; only the reported subset has a second blind decision review."}
