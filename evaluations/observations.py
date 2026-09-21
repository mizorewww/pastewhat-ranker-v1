"""Heldout-only assignment of the registered missing-observation supplement.

Assignments contain private slot metadata, never student features. The author
still produces the same candidate group; both labels and family review see only
the native, budgeted view after the shared observation adapter runs.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

from data_tools.observations import OBSERVATION_PROTOCOL, apply_observation_variant
from data_tools.teacher import atomic_json, utc_now
from evaluations.common import sha256

SUPPLEMENT_PATH = Path("configs/observation_supplement.json")
SUPPLEMENT_SHA256 = "2d5c7945f0a8119b454f8cc4ab6466567db5ea4e7f2b2d09ddd1fb4a6b714bf7"
ADAPTER_PATH = Path("data_tools/observations.py")


def assignment_path(binding: dict, split: str) -> Path:
    if split not in {"calibration", "test"}:
        raise ValueError("The evaluator owns only heldout observation assignments")
    return Path("local/evaluator-generation") / binding["run_id"] / ("observation-assignment-" + split + ".json")


def supplement(binding: dict) -> dict:
    if sha256(SUPPLEMENT_PATH) != SUPPLEMENT_SHA256:
        raise ValueError("Registered observation supplement changed")
    value = json.loads(SUPPLEMENT_PATH.read_text())
    if any(value.get(key) != binding[key] for key in ("run_id", "run_plan_sha256")):
        raise ValueError("Observation supplement belongs to a different run")
    return value


def load_assignment(binding: dict, split: str) -> dict:
    policy = supplement(binding)
    path = assignment_path(binding, split)
    value = json.loads(path.read_text())
    if (value.get("version") != OBSERVATION_PROTOCOL or value.get("split") != split or
            any(value.get(key) != binding[key] for key in ("run_id", "run_plan_sha256")) or
            value.get("supplement_sha256") != SUPPLEMENT_SHA256 or
            value.get("observation_adapter_sha256") != sha256(ADAPTER_PATH)):
        raise ValueError("Heldout observation assignment provenance changed")
    counts = Counter(row["variant"] for row in value["assignments"])
    identities = {(row["family_id"], row["slot"]) for row in value["assignments"]}
    dispatched = {(row["family_id"], row["slot"]) for row in value["dispatched_before_assignment"]}
    if (dict(counts) != policy["split_targets"][split] or len(identities) != len(value["assignments"]) or
            identities & dispatched):
        raise ValueError("Observation assignment quota, uniqueness or dispatch eligibility failed")
    return value


def assigned_specs(specs: list[dict], family_id: str, assignment: dict) -> list[dict]:
    variants = {row["slot"]: row["variant"] for row in assignment["assignments"] if row["family_id"] == family_id}
    result = copy.deepcopy(specs)
    for spec in result:
        if spec["slot"] in variants:
            if spec["desired_decision"] not in {"ambiguous", "insufficient_context"}:
                raise ValueError("Observation supplements may use only existing missing-intent slots")
            spec["observation_variant"] = variants[spec["slot"]]
    return result


def provenance(binding: dict, split: str) -> dict:
    load_assignment(binding, split)
    return {"observation_protocol": OBSERVATION_PROTOCOL,
            "supplement_sha256": SUPPLEMENT_SHA256,
            "assignment_sha256": sha256(assignment_path(binding, split)),
            "observation_adapter_sha256": sha256(ADAPTER_PATH)}


def project_raw(raw: dict, spec: dict, family_id: str, split: str,
                binding: dict | None) -> tuple[dict, dict]:
    variant = spec.get("observation_variant", "standard")
    if variant == "standard":
        return raw, {}
    if not binding:
        raise ValueError("Missing-observation variants require a registered formal assignment")
    assignment = load_assignment(binding, split)
    matches = [row for row in assignment["assignments"] if row["family_id"] == family_id and row["slot"] == spec["slot"]]
    if len(matches) != 1 or matches[0]["variant"] != variant:
        raise ValueError("Authoring variant differs from its frozen slot assignment")
    return apply_observation_variant(raw, variant), {"observation_variant": variant, **provenance(binding, split)}


def current_observation(episode: dict) -> bool:
    metadata = episode.get("synthetic_metadata", {})
    variant = metadata.get("generator_spec", {}).get("observation_variant", "standard")
    if variant == "standard":
        return not metadata.get("observation_variant")
    try:
        binding = {key: metadata[key] for key in ("run_id", "run_plan_sha256")}
        expected = provenance(binding, episode["split"])
        return metadata.get("observation_variant") == variant and all(metadata.get(key) == value for key, value in expected.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_allocation(episodes: list[dict], binding: dict, split: str) -> dict:
    assignment = load_assignment(binding, split)
    expected = {(row["family_id"], row["slot"]): row["variant"] for row in assignment["assignments"]}
    counts = Counter()
    observed = set()
    for episode in episodes:
        spec = episode["synthetic_metadata"]["generator_spec"]
        identity = (episode["family_id"], spec["slot"])
        variant = spec.get("observation_variant", "standard")
        if variant != expected.get(identity, "standard") or not current_observation(episode):
            raise ValueError("Episode observation differs from its frozen assignment")
        if variant != "standard":
            observed.add(identity)
        counts[variant] += 1
    if observed != set(expected):
        raise ValueError("Formal heldout data is missing registered observation variants")
    return {"counts": dict(counts), **provenance(binding, split)}


def freeze_assignment(binding: dict, split: str, planned: dict[str, list[dict]], audit_root: Path) -> dict:
    """Called by the owner during a coordinated drain, before any new request.

    Only author request metadata is read; candidates, responses and labels are
    irrelevant to assignment. Already dispatched requests count even if their
    response failed or never arrived.
    """
    path = assignment_path(binding, split)
    if path.exists():
        return load_assignment(binding, split)
    policy = supplement(binding)
    expected = {(family, spec["slot"]): spec for family, specs in planned.items() for spec in specs}
    dispatched = set()
    evidence = []
    for audit_path in sorted((audit_root / split).glob("*.json")):
        audit = json.loads(audit_path.read_text())
        if audit.get("phase") != "independent-generation":
            continue
        request = json.loads(audit["request"]["messages"][1]["content"])
        family = request.get("allowed_family", {}).get("id")
        matching = []
        for spec in request.get("specs", []):
            identity = (family, spec["slot"])
            if expected.get(identity) == spec:
                dispatched.add(identity)
                matching.append(spec["slot"])
        if matching:
            evidence.append({"audit_id": audit["audit_id"], "request_sha256": audit["request_sha256"],
                             "family_id": family, "slots": sorted(matching)})
    targets = policy["split_targets"][split]
    if targets != {"no_accessibility": 2 * len(planned), "generic_field": len(planned)}:
        raise ValueError("This registered heldout supplement expects balanced two-plus-one family assignments")
    assignments = []
    eligible_counts = []
    for family, specs in planned.items():
        eligible = [spec for spec in specs if spec["desired_decision"] in {"ambiguous", "insufficient_context"}
                    and (family, spec["slot"]) not in dispatched]
        eligible_counts.append(len(eligible))
        if len(eligible) < 3:
            raise ValueError("Too few untouched missing-intent slots for the registered balanced supplement")
        eligible.sort(key=lambda spec: hashlib.sha256(f"{SUPPLEMENT_SHA256}:{split}:{family}:{spec['slot']}".encode()).hexdigest())
        for spec, variant in zip(eligible[:3], ("no_accessibility", "no_accessibility", "generic_field"), strict=True):
            assignments.append({"family_id": family, "slot": spec["slot"], "variant": variant})
    value = {"version": OBSERVATION_PROTOCOL, **binding, "split": split, "assigned_at": utc_now(),
             "supplement_sha256": SUPPLEMENT_SHA256, "observation_adapter_sha256": sha256(ADAPTER_PATH),
             "assignments": assignments,
             "dispatched_before_assignment": [{"family_id": family, "slot": slot} for family, slot in sorted(dispatched)],
             "dispatch_request_evidence": evidence, "eligible_missing_slots": sum(eligible_counts),
             "minimum_eligible_missing_per_family": min(eligible_counts),
             "student_scores_used": False, "existing_observations_or_labels_changed": False}
    atomic_json(path, value)
    load_assignment(binding, split)
    return value
