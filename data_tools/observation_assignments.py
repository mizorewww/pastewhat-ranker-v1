"""Freeze Train/Dev observation variants only for never-authored fixed slots."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from data_tools.observations import OBSERVATION_PROTOCOL
from data_tools.teacher import atomic_json, canonical_bytes, sha256, utc_now
from run_contract import load_run_plan


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/observation_supplement.json"


def dispatched_ids(split, plan):
    if split not in {"train", "dev"}:
        raise ValueError("Only owned Train/Dev author requests may be inspected")
    identifiers = set()
    for path in (ROOT / "local/teacher" / split / "main").glob("*.json"):
        audit = json.loads(path.read_text())
        try:
            slots = json.loads(audit["request"]["messages"][1]["content"]).get("plans", [])
        except (KeyError, ValueError, IndexError):
            continue
        identifiers.update(slot["id"] for slot in slots if "-" + plan.run_id + "-" in slot.get("id", ""))
    return identifiers


def assignment_path(split, plan):
    if split not in {"train", "dev"}:
        raise ValueError("Only owned Train/Dev assignments may be opened")
    return ROOT / "data" / f"{split}.observation_assignments.{plan.run_id}.json"


def load_assignment(split, plan):
    path = assignment_path(split, plan)
    if not path.is_file():
        return None
    document = json.loads(path.read_text())
    policy = json.loads(POLICY.read_text())
    if any(document.get(key) != value or policy.get(key) != value for key, value in plan.binding().items()):
        raise ValueError("Observation assignment belongs to a different run")
    if document.get("protocol") != OBSERVATION_PROTOCOL or document.get("split") != split or document.get("supplement_sha256") != sha256(POLICY.read_bytes()):
        raise ValueError("Observation supplement or assignment protocol differs")
    observed = Counter(row["variant"] for row in document["assignments"].values())
    if dict(observed) != policy["split_targets"][split]:
        raise ValueError("Observation assignment does not meet registered counts")
    return document


def provenance_for_slot(split, plan, identifier):
    document = load_assignment(split, plan)
    assignment = document["assignments"].get(identifier) if document else None
    if assignment is None:
        return None
    path = assignment_path(split, plan)
    return {"protocol": OBSERVATION_PROTOCOL, "variant": assignment["variant"],
            "supplement_path": str(POLICY.relative_to(ROOT)), "supplement_sha256": sha256(POLICY.read_bytes()),
            "assignment_path": str(path.relative_to(ROOT)), "assignment_sha256": sha256(path.read_bytes()),
            "observation_adapter_sha256": sha256(Path(__file__).with_name("observations.py").read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--run-plan", required=True)
    parser.add_argument("--verify-undispatched", action="store_true")
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    existing = load_assignment(args.split, plan)
    issued = dispatched_ids(args.split, plan)
    path = assignment_path(args.split, plan)
    if existing:
        if args.verify_undispatched and set(existing["assignments"]) & issued:
            raise ValueError("An observation slot has already been dispatched; do not alter its source")
        print(json.dumps({"status": "registered_assignment_verified", "split": args.split, "sha256": sha256(path.read_bytes()), "counts": existing["counts"]}))
        return
    from data_tools.generate import build_plan
    batches = build_plan(args.split, plan.target(args.split), 5, "main", run_plan=plan)
    available = defaultdict(list)
    for batch in batches:
        for slot in batch["plans"]:
            if slot["id"] not in issued and slot["scenario_type"] in {"ambiguous", "insufficient_context"}:
                available[batch["family"]["id"]].append(slot)
    for rows in available.values():
        # Prefer the farthest future original slot to avoid a dispatch race while
        # the current producer finishes its already queued work before the drain.
        rows.sort(key=lambda row: (-row["variant_number"], sha256(row["id"].encode())))
    policy = json.loads(POLICY.read_text())
    if any(policy.get(key) != value for key, value in plan.binding().items()):
        raise ValueError("Observation policy belongs to a different run")
    assignments, family_counts = {}, Counter()
    for variant, target in policy["split_targets"][args.split].items():
        this_variant = Counter()
        for _ in range(target):
            options = [family for family, rows in available.items() if rows]
            if not options:
                raise ValueError("Insufficient never-dispatched missing-intent slots; do not replace existing observations")
            family = min(options, key=lambda family: (this_variant[family], family_counts[family], sha256((plan.run_id + "/" + variant + "/" + family).encode())))
            slot = available[family].pop(0)
            assignments[slot["id"]] = {"variant": variant, "family_id": family, "candidate_count": slot["candidate_count"], "context_language": slot["context_language"], "planned_scenario_type": slot["scenario_type"]}
            this_variant[family] += 1
            family_counts[family] += 1
    if set(assignments) & dispatched_ids(args.split, plan):
        raise ValueError("Selected slot was dispatched during registration; recompute without editing that request")
    document = {**plan.binding(), "protocol": OBSERVATION_PROTOCOL, "split": args.split, "registered_at": utc_now(), "supplement_sha256": sha256(POLICY.read_bytes()), "counts": dict(Counter(row["variant"] for row in assignments.values())), "family_counts": dict(family_counts), "assignments": assignments, "already_dispatched_slots_at_registration": len(issued), "dispatched_ids_sha256": sha256(canonical_bytes(sorted(issued))), "selection": "Only never-dispatched missing-intent slots; balanced family round robin and farthest future original slot first. Candidate count, language, family, action bucket and ID unchanged."}
    atomic_json(path, document)
    print(json.dumps({"status": "registered", "split": args.split, "sha256": sha256(path.read_bytes()), "counts": document["counts"]}))


if __name__ == "__main__":
    main()
