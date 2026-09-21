"""Replay Train/Dev capture and teacher audit references without API calls."""

from __future__ import annotations

import copy
import functools
import json
from pathlib import Path
import random

from data_tools.content import content_fingerprint
from data_tools.teacher import TeacherClient, canonical_bytes, sha256


class ReplayVerifier:
    def __init__(self, root, split, preprocessor):
        if split not in ("train", "dev"):
            raise ValueError("Only the Train/Dev owner may use this verifier")
        self.root = Path(root)
        self.split = split
        self.preprocessor = preprocessor
        self.projection_sha = sha256((self.root / "tools/context_projection/provenance.json").read_bytes())
        # Commit 40140cf predates only the explicit-fragment authoring adapter.
        # It used the identical FocusText native formatter and remains replayable.
        self.compatible_projection_shas = {self.projection_sha, "d9f70061a1a4348e41e2b58f9311b2011c4f8e94212c7ff8480cfd8e2867e3ab"}

    @functools.lru_cache(maxsize=256)
    def audit(self, identifier, phase="main"):
        if not identifier or any(character not in "0123456789abcdef" for character in identifier) or len(identifier) != 64:
            raise ValueError("Invalid teacher audit reference")
        if phase not in ("main", "hardcase", "hard-pool"):
            raise ValueError("Unowned teacher audit phase")
        path = self.root / "local/teacher" / self.split / phase / f"{identifier}.json"
        audit = json.loads(path.read_text())
        if audit.get("status") != "success" or sha256(canonical_bytes(audit["request"])) != audit["request_sha256"]:
            raise ValueError("Teacher audit is incomplete or its request hash differs")
        return audit, TeacherClient._result(audit, cache_hit=True).parsed

    @staticmethod
    def visible_match(audit, episode, *, ignore_order=False):
        payload = json.loads(audit["request"]["messages"][1]["content"])
        expected = content_fingerprint(episode) if ignore_order else canonical_bytes({"context": episode["context"], "entries": episode["entries"]})
        matches = [item for item in payload["episodes"] if (content_fingerprint(item) if ignore_order else canonical_bytes({"context": item["context"], "entries": item["entries"]})) == expected]
        if len(matches) != 1:
            raise ValueError("Teacher request does not contain this exact student-visible episode")
        return matches[0]

    def verify(self, episode):
        from data_tools.generate import CANDIDATE_PROJECTION_PATH, LABEL_SYSTEM, PARTITION_PATH, validate_generated, validate_labels
        from data_tools.audit import AUDIT_SYSTEM
        from data_tools.authoring import AUTHORING_PROTOCOL, compile_compact_episode, owned_profile
        from data_tools.labeling import LABEL_PROTOCOL
        from tools.project_context import project_context
        from tools.project_candidates import project_candidates

        provenance = episode["provenance"]
        if provenance.get("capture_format") != "pastewhat-focus-v1" or provenance.get("projection_provenance_sha256") not in self.compatible_projection_shas:
            raise ValueError("Formal data lacks the pinned native capture provenance")
        if provenance.get("candidate_payload_protocol") != "native-synthetic-payload-v1" or provenance.get("candidate_projection_provenance_sha256") != sha256(CANDIDATE_PROJECTION_PATH.read_bytes()):
            raise ValueError("Formal data lacks the current native candidate-payload provenance")
        phase = provenance.get("generation_phase", "main")
        generation_audit, generated = self.audit(provenance["generation_audit_id"], phase)
        if provenance.get("authoring_protocol") != AUTHORING_PROTOCOL or provenance.get("label_protocol") != LABEL_PROTOCOL:
            raise ValueError("Formal data lacks the compact/exhaustive-verdict protocol")
        if provenance.get("compact_authoring_sha256") != sha256((self.root / "data_tools/authoring.py").read_bytes()):
            raise ValueError("The authoring compiler differs from the frozen protocol")
        author_request = json.loads(generation_audit["request"]["messages"][1]["content"])
        profile = author_request.get("field_fixture")
        if profile != owned_profile(episode["family_id"]) or provenance.get("field_fixture_sha256") != sha256(canonical_bytes(profile)):
            raise ValueError("The fixed pre-label field fixture does not match its provenance")
        raw = next((copy.deepcopy(item) for item in generated["episodes"] if item.get("slot") == episode["id"]), None)
        if raw is None:
            raise ValueError("Episode is absent from its authoring response")
        raw = compile_compact_episode(raw, episode_id=episode["id"], profile=profile, candidate_count=len(episode["entries"]))
        observation = provenance.get("observation_variant")
        expected_assignment = None
        if provenance.get("run_id"):
            assigned_path = self.root / "data" / f"{self.split}.observation_assignments.{provenance['run_id']}.json"
            if assigned_path.parent.resolve() != (self.root / "data").resolve():
                raise ValueError("Invalid owned observation assignment path")
            if assigned_path.is_file():
                assigned = json.loads(assigned_path.read_text())
                expected_assignment = assigned["assignments"].get(episode["id"])
        if bool(observation) != bool(expected_assignment):
            raise ValueError("Episode does not preserve its pre-registered observation variant")
        if observation:
            from data_tools.observations import OBSERVATION_PROTOCOL, apply_observation_variant
            policy_path = self.root / "configs/observation_supplement.json"
            policy = json.loads(policy_path.read_text())
            if observation.get("protocol") != OBSERVATION_PROTOCOL or observation.get("variant") != expected_assignment["variant"] or expected_assignment["family_id"] != episode["family_id"]:
                raise ValueError("Observation variant or conceptual lineage differs from assignment")
            if any(assigned.get(key) != provenance.get(key) or policy.get(key) != provenance.get(key) for key in ("run_id", "run_plan_sha256")):
                raise ValueError("Observation supplement/assignment belongs to a different run")
            expected_hashes = {"supplement_sha256": sha256(policy_path.read_bytes()), "assignment_sha256": sha256(assigned_path.read_bytes()), "observation_adapter_sha256": sha256((self.root / "data_tools/observations.py").read_bytes())}
            if any(observation.get(key) != value for key, value in expected_hashes.items()) or assigned.get("supplement_sha256") != expected_hashes["supplement_sha256"]:
                raise ValueError("Observation supplement, assignment or adapter changed")
            raw = apply_observation_variant(raw, observation["variant"])
        raw = validate_generated({"episodes": [raw]}, [{"id": episode["id"], "candidate_count": len(episode["entries"])}])[0]
        raw["family_id"] = episode["family_id"]
        raw["context"] = project_context(raw["context"], capture=raw["capture"])
        if sha256(canonical_bytes(raw["entries"])) != provenance.get("candidate_fixture_authoring_sha256"):
            raise ValueError("Authoring payload fixture specification differs from recorded provenance")
        raw["entries"] = project_candidates(raw["entries"])
        replayed = self.preprocessor.prepare_episode(raw)
        if replayed["context"] != episode["context"] or replayed["entries"] != episode["entries"] or replayed["preprocessing"]["visible_sha256"] != provenance["label_visible_sha256"]:
            raise ValueError("Raw native capture does not replay to the labeled student input")
        if "capture" in episode:
            raise ValueError("Raw authoring capture must stay out of the prepared dataset")

        first_audit, first_output = self.audit(provenance["label_audit_id"], phase)
        if first_audit["request"]["messages"][0]["content"] != LABEL_SYSTEM:
            raise ValueError("Original first teacher label used a different annotation protocol")
        first_input = self.visible_match(first_audit, episode)
        first_annotation = next(item for item in first_output["labels"] if item["id"] == first_input["id"])
        validate_labels({"labels": [first_annotation]}, [first_input], require_quoted=True)
        if first_annotation["label"] != episode["label"]:
            raise ValueError("Stored decision label differs from the original teacher annotation")

        second_audit, second_output = self.audit(provenance["blind_label_audit_id"], phase)
        if second_audit["request"]["messages"][0]["content"] != LABEL_SYSTEM:
            raise ValueError("Original blind teacher label used a different annotation protocol")
        second_input = self.visible_match(second_audit, episode, ignore_order=True)
        second_annotation = next(item for item in second_output["labels"] if item["id"] == second_input["id"])
        validate_labels({"labels": [second_annotation]}, [second_input], require_quoted=True)
        expected_entries = copy.deepcopy(episode["entries"])
        random.Random(int(episode["preprocessing"]["visible_sha256"][:16], 16) ^ 9017).shuffle(expected_entries)
        mapping = {f"x{index+1}": entry["id"] for index, entry in enumerate(expected_entries)}
        for index, entry in enumerate(expected_entries):
            entry["id"] = f"x{index+1}"
        if expected_entries != second_input["entries"]:
            raise ValueError("Blind candidate ID/order perturbation does not replay")
        second = second_annotation["label"]
        first = episode["label"]
        if first["decision"] != second["decision"] or set(first["acceptable_ids"]) != {mapping[identifier] for identifier in second["acceptable_ids"]}:
            raise ValueError("The two teacher action sets do not agree")
        reasons = [first["abstain_reason"], second["abstain_reason"]]
        reason_agreement = reasons[0] == reasons[1]
        if not reason_agreement and set(reasons) != {"ambiguous", "insufficient_context"}:
            raise ValueError("Only the two missing-context abstain reasons may disagree")
        if provenance.get("reason_agreement") is not reason_agreement or provenance.get("observed_abstain_reasons") != reasons:
            raise ValueError("Reason-agreement provenance differs from the teacher responses")

        family_audit, family_output = self.audit(provenance["family_review_audit_id"], phase)
        family_request = json.loads(family_audit["request"]["messages"][1]["content"])
        partition = json.loads(PARTITION_PATH.read_text())
        expected_taxonomy = [item for families in partition["families"].values() for item in families]
        if family_audit["request"]["messages"][0]["content"] != AUDIT_SYSTEM or family_request.get("operation_taxonomy") != expected_taxonomy or set(family_request) != {"operation_taxonomy", "episodes"}:
            raise ValueError("Family review was not blind under the frozen complete taxonomy")
        family_input = self.visible_match(family_audit, episode)
        review = next(item for item in family_output["reviews"] if item["id"] == family_input["id"])
        if review.get("observed_family_id") != episode["family_id"] or review.get("secondary_family_ids") or review.get("deployment_visible") is not True or review.get("payload_metadata_consistent") is not True:
            raise ValueError("Blind family/deployment audit did not pass")
        return {"native_capture_replayed": True, "teacher_actions_replayed": True, "reason_agreement": reason_agreement}
