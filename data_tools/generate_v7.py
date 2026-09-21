"""Owned Train/Dev stream, with a bounded initial cost check and durable batches."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import time

from data_tools.authoring import candidate_space, owned_profile
from data_tools.content import ContentRegistry, content_fingerprint
from data_tools.freeze import publish_bytes
from data_tools.teacher import TeacherClient, atomic_json, canonical_bytes, sha256, utc_now
from data_tools.rate_limit import AccountCoordinator
from data_tools.v7 import PROTOCOL, produce_batch
from pastewhat_ranker.preprocess import Preprocessor
from run_contract import action_quotas, family_quotas, load_run_plan

ROOT = Path(__file__).resolve().parents[1]
PARTITION = ROOT / "data_tools/family_partition.json"


def integer_seed(*values):
    return int(sha256(canonical_bytes(values))[:12], 16)


def make_specs(plan, split, batch_size=10):
    partition = json.loads(PARTITION.read_text())
    quotas = action_quotas(family_quotas(partition, split, plan.target(split)))
    grouped = []
    for family in partition["families"][split]:
        family_id = family["id"]
        buckets = quotas[family_id]
        # Evenly interleave each action so the early stream can fill the pilot.
        actions = []
        for action, count in buckets.items():
            actions.extend((index / count, action, index) for index in range(count))
        actions.sort()
        missing_count = buckets["missing_intent"]
        observations = ["no_accessibility"] * round(missing_count * .4) + ["generic_field"] * round(missing_count * .2)
        observations += ["standard"] * (missing_count - len(observations))
        random.Random(integer_seed(plan.run_id, family_id, "observation")).shuffle(observations)
        lower, upper = candidate_space(family_id)
        plans = []
        for index, (_, action, action_index) in enumerate(actions):
            seed = integer_seed(plan.run_id, split, family_id, index)
            scenario = action if action != "missing_intent" else ("ambiguous" if action_index % 2 else "insufficient_context")
            variant = observations[action_index] if action == "missing_intent" else "standard"
            plans.append({"id": f"{split}-v7-{family_id}-{index:05d}", "candidate_count": random.Random(seed ^ 37).randint(lower, upper), "context_language": random.Random(seed ^ 101).choice(["English", "简体中文", "English with Chinese UI text"]), "scenario_type": scenario, "observation_variant": variant, "seed": seed})
        family_specs = []
        for offset in range(0, len(plans), batch_size):
            number = offset // batch_size
            seed = integer_seed(plan.run_id, family_id, "mother", number)
            mother = {"id": f"{plan.run_id}:{split}:{family_id}:mother-{number:04d}", "operation": family["operation"], "data_seed": seed, "constraints": "Invent concrete synthetic task facts and operands from this seed; every answerable goal and distinguishing constraint must be in actual visible helper text. Vary operations, boundary conditions, output constraints and equivalent forms within this source family. Do not merely replace nouns in one template. A missing preference never permits selecting arbitrary valid options."}
            batch_id = f"{split}-{family_id}-{number:04d}"
            family_specs.append({"batch_id": batch_id, "family_id": family_id, "mother_task": mother, "profile": owned_profile(family_id), "plans": plans[offset:offset + batch_size], "seed": seed, "run_binding": plan.binding(), "audit_sample": integer_seed(plan.run_id, batch_id, "review") % 10 == 0})
        grouped.append(family_specs)
    # Cover all semantic families before advancing a family's next source batch.
    return [specs[index] for index in range(max(map(len, grouped))) for specs in grouped if index < len(specs)]


def usage_summary(directory):
    totals, phases, statuses, models = Counter(), {}, Counter(), Counter()
    transport_unknown, http_without_usage, started = 0, 0, Counter()
    with AccountCoordinator()._state() as state:
        leases = dict(state["leases"])
    for path in directory.glob("*.json"):
        audit = json.loads(path.read_text())
        statuses[audit.get("status", "unknown")] += 1
        usage = (audit.get("response") or {}).get("usage") or {}
        if usage:
            phase = phases.setdefault(audit.get("phase", "unknown"), Counter())
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                totals[key] += usage.get(key, 0)
                phase[key] += usage.get(key, 0)
            reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            totals["reported_reasoning_tokens"] += reasoning
            phase["reported_reasoning_tokens"] += reasoning
            models[(audit.get("response") or {}).get("model", "unknown")] += 1
        transport_unknown += sum("error_type" in attempt and not attempt.get("http_status") for attempt in audit.get("attempts", []))
        http_without_usage += sum(bool(attempt.get("http_status")) for attempt in audit.get("attempts", []))
        if audit.get("status") == "request_started":
            if audit.get("lease_id") in leases:
                started["in_flight_observed"] += 1
            else:
                try:
                    os.kill(audit.get("pid", -1), 0)
                    alive = True
                except (ProcessLookupError, PermissionError):
                    alive = False
                started["live_process_completion_pending" if alive else "orphaned_started_unknown_usage"] += 1
    return {"request_records": sum(statuses.values()), "known_usage": dict(totals), "by_phase": {key: dict(value) for key, value in phases.items()}, "statuses": dict(statuses), "response_models": dict(models), "transport_attempts_without_usage": transport_unknown, "http_error_attempts_without_usage": http_without_usage, "unfinished_started_requests": dict(started)}


def publish_pool(base, split, plan, started):
    records = [json.loads(path.read_text()) for path in (base / "batches" / split).glob("*.json")]
    rows = [row for record in records for row in record["accepted"]]
    rows.sort(key=lambda row: row["id"])
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    usage = usage_summary(base / "teacher" / split)
    result = {**plan.binding(), "teacher_contract_version": PROTOCOL, "split": split, "episodes": len(rows), "sha256": sha256(payload), "family_partition_sha256": sha256(PARTITION.read_bytes()), "batch_records": len(records), "completed_batches": sum(record["status"] == "complete" for record in records), "planned_slots_in_batches": sum(len(record["spec"]["plans"]) for record in records), "unfilled_slots": sum(len(record.get("unfilled_ids", [])) for record in records), "quality_paths": dict(Counter(row["provenance"]["quality_path"] for row in rows)), "actual_actions": dict(Counter(row["label"]["decision"] if row["label"]["decision"] == "select" else row["label"]["abstain_reason"] for row in rows)), "observation_variants": dict(Counter(row["provenance"]["observation_variant"] for row in rows)), "teacher_usage": usage, "known_tokens_per_accepted": usage["known_usage"].get("total_tokens", 0) / len(rows) if rows else None, "process_elapsed_seconds": round(time.monotonic() - started, 2), "updated_at": utc_now()}
    atomic_json(base / f"{split}.manifest.json", result)
    publish_bytes(base / f"{split}.jsonl", payload)
    atomic_json(base / f"{split}.fingerprints.json", {"content_sha256": [content_fingerprint(row) for row in rows], **plan.binding()})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-plan", required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--max-batches", type=int, help="Cap newly scheduled source batches for the initial real cost check")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    if plan.document["teacher_contract_version"] != PROTOCOL:
        raise SystemExit("This producer only runs the registered v7 protocol")
    base = ROOT / "local/v7" / plan.run_id
    base.mkdir(parents=True, exist_ok=True)
    specs = make_specs(plan, args.split)
    sampling = {**plan.binding(), "teacher_contract_version": PROTOCOL, "specs": specs}
    sampling_path = base / f"{args.split}.sampling.json"
    if sampling_path.exists() and json.loads(sampling_path.read_text()) != sampling:
        raise ValueError("Pre-registered source sampling cannot change in place")
    atomic_json(sampling_path, sampling)
    if args.plan_only:
        print(json.dumps({"split": args.split, "episodes": sum(len(spec["plans"]) for spec in specs), "batches": len(specs), "sampling_sha256": sha256(sampling_path.read_bytes())}))
        return
    batch_dir = base / "batches" / args.split
    batch_dir.mkdir(parents=True, exist_ok=True)
    pending = [spec for spec in specs if not (batch_dir / (spec["batch_id"] + ".json")).is_file() or json.loads((batch_dir / (spec["batch_id"] + ".json")).read_text())["status"] != "complete"]
    if args.max_batches:
        pending = pending[:args.max_batches]
    client = TeacherClient(base / "teacher" / args.split)
    preprocessor = Preprocessor(str(ROOT.parent / "laya-mlx/models/laya-multilingual/tokenizer"))
    registry = ContentRegistry(base / "content.sqlite3")
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(produce_batch, spec, client=client, preprocessor=preprocessor, destination=batch_dir / (spec["batch_id"] + ".json"), claim=registry.claim): spec["batch_id"] for spec in pending}
        for future in as_completed(futures):
            future.result()
            plan.verify_unchanged()
            result = publish_pool(base, args.split, plan, started)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    atomic_json(base / f"{args.split}.run-completion.json", {**plan.binding(), "updated_at": utc_now(), "scheduled_batches_this_process": len(pending), "cost_check_cap": args.max_batches, "status": "bounded_cost_check_finished" if args.max_batches else "base_sources_finished"})


if __name__ == "__main__":
    main()
