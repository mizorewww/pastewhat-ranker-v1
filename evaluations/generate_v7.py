"""Thin heldout owner for the shared efficient v7 batch producer.

The first cost check is explicitly bounded with --max-batches. Partial batches
remain private and cannot be used by the formal scoring commands.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import threading

from data_tools.content import content_fingerprint
from data_tools.teacher import TeacherClient, atomic_json, make_teacher_client
from evaluations.authoring_v7 import candidate_space, mother_task, profile_for_spec, recipe_id
from evaluations.common import sha256
from run_contract import action_quotas, family_quotas, load_run_plan

CONTRACT = "teacher-episodes-v7-batched-decisions"
RECIPE_SOURCE = Path("evaluations/authoring_v7.py")
PROFILE_SOURCE = Path("evaluations/authoring-profiles.json")
LANGUAGES = ("English", "Simplified Chinese", "Spanish", "Japanese", "French", "German")


def planned_batches(plan, partition: dict, split: str, batch_size: int) -> list[dict]:
    allocations = family_quotas(partition, split, plan.target(split))
    actions = action_quotas(allocations)
    batches = []
    for family_index, family in enumerate(partition["families"][split]):
        name = family["id"]
        count = allocations[name]
        quota = actions[name]
        decisions = ["select"] * quota["select"] + ["no_match"] * quota["no_match"] + ["insufficient_context" if i % 2 else "ambiguous" for i in range(quota["missing_intent"])]
        allowed_sizes = candidate_space(name)
        sizes = [allowed_sizes[i % len(allowed_sizes)] for i in range(count)]
        languages = [LANGUAGES[i % len(LANGUAGES)] for i in range(count)]
        for factor, values in (("decisions", decisions), ("sizes", sizes), ("languages", languages)):
            random.Random(f"{plan.run_id}:{split}:{name}:v7:{factor}").shuffle(values)
        missing = [i for i, decision in enumerate(decisions) if decision in {"ambiguous", "insufficient_context"}]
        variants = (["no_accessibility"] * (len(missing) * 4 // 10) +
                    ["generic_field"] * (len(missing) * 2 // 10))
        variants += ["standard"] * (len(missing) - len(variants))
        random.Random(f"{plan.run_id}:{split}:{name}:v7:observation").shuffle(variants)
        variant_by_slot = dict(zip(missing, variants, strict=True))
        plans = []
        for slot in range(count):
            scenario = decisions[slot]
            if scenario == "ambiguous" and sizes[slot] == 1:
                scenario = "insufficient_context"
            seed = int(hashlib.sha256(f"{plan.run_id}:{split}:{name}:{slot}:v7".encode()).hexdigest()[:12], 16)
            plans.append({"id": f"{plan.run_id}-{split}-{name}-{slot:05d}",
                          "candidate_count": sizes[slot], "context_language": languages[slot],
                          "scenario_type": scenario, "observation_variant": variant_by_slot.get(slot, "standard"),
                          "seed": seed})
        for start in range(0, count, batch_size):
            batch_id = f"{plan.run_id}-{split}-{name}-{start:05d}"
            audit_draw = int(hashlib.sha256((batch_id + ":audit-v7").encode()).hexdigest()[:8], 16) / 2**32
            batches.append({"batch_id": batch_id, "family_id": name,
                            "mother_task": {"id": recipe_id(name), "operation": family["operation"],
                                            "constraints": mother_task(name)},
                            "profile": profile_for_spec(name, {}), "plans": plans[start:start + batch_size],
                            "seed": plans[start]["seed"], "run_binding": plan.binding(),
                            "audit_sample": audit_draw < 0.1,
                            "_owner_order": (start, family_index)})
    batches.sort(key=lambda item: item["_owner_order"])
    for batch in batches:
        del batch["_owner_order"]
    return batches


def recover_cached_authors(record_path, *, client, preprocessor, claim, already_accepted):
    """Relabel previously unaccepted native-valid drafts without another author.

    Original records and labels remain unchanged. A semantic rejection is never
    rescued by this structural recovery path.
    """
    from data_tools.v7 import prepare_author_batch, produce_batch
    original = json.loads(record_path.read_text())
    semantic_rejections = {row["id"] for row in original["rejected"] if "id" in row and "original_label" in row}
    pending = {p["id"]: p for p in original["spec"]["plans"] if p["id"] not in already_accepted | semantic_rejections}
    recovered = []
    for identifier in original["audit_ids"]:
        if not pending:
            break
        audit_path = client.audit_dir / (identifier + ".json")
        audit = json.loads(audit_path.read_text())
        if audit.get("phase") != "v7-author" or audit.get("status") != "success":
            continue
        author = TeacherClient._result(audit, cache_hit=True)
        prepared, _ = prepare_author_batch(author.parsed, list(pending.values()), original["spec"]["profile"], preprocessor)
        eligible = {row["id"] for row in prepared}
        if not eligible:
            continue
        spec = {**original["spec"], "batch_id": original["spec"]["batch_id"] + "-cached-" + identifier[:12],
                "plans": [p for p in pending.values() if p["id"] in eligible],
                "cached_author_recovery": {"original_batch_path": str(record_path), "original_batch_sha256": sha256(record_path),
                    "original_author_audit_id": identifier, "original_author_request_sha256": audit["request_sha256"],
                    "original_labels_changed": False, "additional_author_calls_allowed": False}}
        destination = record_path.parent / "recovery" / (spec["batch_id"] + ".json")
        result = produce_batch(spec, client=client, preprocessor=preprocessor, destination=destination,
                               claim=claim, cached_author=author)
        recovered.append(result)
        already_accepted.update(row["id"] for row in result["accepted"])
        # This draft group gets one recovery label path. Do not try another
        # author's version after a semantic disagreement or format failure.
        for key in eligible:
            pending.pop(key, None)
    return recovered


def original_author_cache(spec, client):
    cache = {}
    expected_ids = {plan["id"]: plan for plan in spec["plans"]}
    for path in sorted(client.audit_dir.glob("*.json")):
        audit = json.loads(path.read_text())
        if audit.get("phase") != "v7-author" or audit.get("status") != "success":
            continue
        for attempt in (0, 1):
            if audit.get("request_id") != spec["batch_id"] + f"-a{attempt}":
                continue
            request = json.loads(audit["request"]["messages"][1]["content"])
            if (request.get("mother_task") != spec["mother_task"] or request.get("field_profile") != spec["profile"] or
                    any(expected_ids.get(plan["id"]) != plan for plan in request.get("plans", []))):
                raise ValueError("Cached author source does not match its original heldout plan")
            cache.setdefault(attempt, TeacherClient._result(audit, cache_hit=True))
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--max-batches", type=int, required=True, help="Explicit cost-check work limit; this command does not freeze a partial dataset")
    parser.add_argument("--batch-size", type=int, default=10, choices=(10, 20))
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2))
    parser.add_argument("--recover-cached", action="store_true", help="Reuse unaccepted author drafts after the authorized native count/label-format fix; no additional author calls")
    parser.add_argument("--tokenizer", type=Path, default=Path("../laya-mlx/models/laya-multilingual/tokenizer"))
    args = parser.parse_args()
    if args.max_batches < 1:
        parser.error("A positive bounded batch count is required")
    plan = load_run_plan(args.run_plan)
    if plan.document["teacher_contract_version"] != CONTRACT:
        parser.error("The supplied plan is not the registered v7 production run")
    from data_tools.v7 import produce_batch
    from pastewhat_ranker.preprocess import Preprocessor
    partition = json.loads(Path("data_tools/family_partition.json").read_text())
    batches = planned_batches(plan, partition, args.split, args.batch_size)[:args.max_batches]
    directory = Path("local/evaluator-v7") / plan.run_id / args.split
    directory.mkdir(parents=True, exist_ok=True)
    source_binding = {**plan.binding(), "teacher_contract_version": CONTRACT,
                      "recipe_sha256": sha256(RECIPE_SOURCE), "profile_sha256": sha256(PROFILE_SOURCE),
                      "shared_producer_sha256": sha256("data_tools/v7.py"),
                      "family_partition_sha256": sha256("data_tools/family_partition.json")}
    binding_path = directory / "owner-binding.json"
    if binding_path.exists():
        previous = json.loads(binding_path.read_text())
        if ({key: value for key, value in previous.items() if key != "shared_producer_sha256"} !=
                {key: value for key, value in source_binding.items() if key != "shared_producer_sha256"}):
            raise SystemExit("Heldout v7 owner sources changed after authoring began")
        if previous != source_binding:
            atomic_json(directory / "producer-revisions" / (previous["shared_producer_sha256"] + ".json"), previous)
    atomic_json(binding_path, source_binding)
    client = make_teacher_client(Path("local/teacher-v7") / plan.run_id / args.split)
    preprocessor = Preprocessor(args.tokenizer)
    content_ids = {}
    already_accepted = set()
    for path in directory.rglob(plan.run_id + "-*.json"):
        for row in json.loads(path.read_text()).get("accepted", []):
            content_ids.setdefault(content_fingerprint(row), row["id"])
            already_accepted.add(row["id"])
    exclusion_path = Path("local/evaluator-quality-exclusions") / (args.split + ".json")
    excluded = set(json.loads(exclusion_path.read_text())["content_fingerprints"]) if exclusion_path.is_file() else set()
    content_lock = threading.Lock()

    def claim(row):
        fingerprint = content_fingerprint(row)
        with content_lock:
            if fingerprint in excluded:
                return "Previously excluded visible content cannot re-enter the corpus"
            previous = content_ids.setdefault(fingerprint, row["id"])
            return None if previous == row["id"] else "Duplicate visible content across heldout batches"

    totals = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(produce_batch, batch, client=client, preprocessor=preprocessor,
                                   destination=directory / (batch["batch_id"] + ".json"), claim=claim,
                                   author_cache=original_author_cache(batch, client)) for batch in batches]
        for future in as_completed(futures):
            result = future.result()
            totals["completed_batches"] += 1
            totals["accepted"] += len(result.get("accepted", []))
            totals["rejected"] += len(result.get("rejected", []))
            for name, value in result.get("quality_counts", {}).items():
                if isinstance(value, int):
                    totals["quality_" + name] += value
            report = {**source_binding, "split": args.split, "status": "bounded_cost_check_partial",
                      "planned_batches": len(batches), "planned_episodes": sum(len(b["plans"]) for b in batches),
                      "counts": dict(totals), "student_scoring_allowed": False}
            atomic_json(directory / "progress.json", report)
            print(json.dumps(report), flush=True)
    if args.recover_cached:
        for batch in batches:
            path = directory / (batch["batch_id"] + ".json")
            already_accepted.update(row["id"] for row in json.loads(path.read_text())["accepted"])
        for batch in batches:
            recover_cached_authors(directory / (batch["batch_id"] + ".json"), client=client,
                                   preprocessor=preprocessor, claim=claim, already_accepted=already_accepted)
        atomic_json(directory / "cached-recovery-progress.json", {**source_binding, "split": args.split,
                    "retained_unique": len(already_accepted), "student_scoring_allowed": False,
                    "recovery_additional_author_calls": 0})
    plan.verify_unchanged()


if __name__ == "__main__":
    main()
