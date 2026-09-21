"""Owned Train/Dev stream, with a bounded initial cost check and durable batches."""
from __future__ import annotations

import argparse
import copy
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
from data_tools.freeze_v7 import try_freeze
from data_tools.teacher import TeacherClient, atomic_json, canonical_bytes, sha256, utc_now
from data_tools.rate_limit import AccountCoordinator
from data_tools.v7 import PROTOCOL, produce_batch
from pastewhat_ranker.preprocess import Preprocessor
from run_contract import action_quotas, family_quotas, load_run_plan

ROOT = Path(__file__).resolve().parents[1]
PARTITION = ROOT / "data_tools/family_partition.json"


def integer_seed(*values):
    return int(sha256(canonical_bytes(values))[:12], 16)


def make_specs(plan, split, batch_size=10, *, target=None, namespace=""):
    partition = json.loads(PARTITION.read_text())
    quotas = action_quotas(family_quotas(partition, split, target or plan.target(split)))
    seed_identity = plan.run_id + (":" + namespace if namespace else "")
    id_namespace = namespace + "-" if namespace else ""
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
        random.Random(integer_seed(seed_identity, family_id, "observation")).shuffle(observations)
        lower, upper = candidate_space(family_id)
        plans = []
        for index, (_, action, action_index) in enumerate(actions):
            seed = integer_seed(seed_identity, split, family_id, index)
            scenario = action if action != "missing_intent" else ("ambiguous" if action_index % 2 else "insufficient_context")
            variant = observations[action_index] if action == "missing_intent" else "standard"
            plans.append({"id": f"{split}-v7-{id_namespace}{family_id}-{index:05d}", "candidate_count": random.Random(seed ^ 37).randint(lower, upper), "context_language": random.Random(seed ^ 101).choice(["English", "简体中文", "English with Chinese UI text"]), "scenario_type": scenario, "observation_variant": variant, "seed": seed})
        family_specs = []
        for offset in range(0, len(plans), batch_size):
            number = offset // batch_size
            seed = integer_seed(seed_identity, family_id, "mother", number)
            mother = {"id": f"{seed_identity}:{split}:{family_id}:mother-{number:04d}", "operation": family["operation"], "data_seed": seed, "constraints": "Invent concrete synthetic task facts and operands from this seed; every answerable goal and distinguishing constraint must be in actual visible helper text. Vary operations, boundary conditions, output constraints and equivalent forms within this source family. Do not merely replace nouns in one template. A missing preference never permits selecting arbitrary valid options."}
            if namespace:
                mother["constraints"] += " New hard-candidate source: emphasize same-type alternatives that differ by a stated boundary, parameter, number, negation or path. Create a new situation rather than changing IDs on a previous task."
            batch_id = f"{split}-{id_namespace}{family_id}-{number:04d}"
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


def replacement_specs(originals, rows, round_number):
    """New source situations fill retired logical slots; accepted slots stay fixed."""
    fulfilled = {row["provenance"].get("quota_slot_id", row["id"]) for row in rows}
    result = []
    for original in originals:
        missing = [item for item in original["plans"] if item["id"] not in fulfilled]
        if not missing:
            continue
        spec = copy.deepcopy(original)
        spec["batch_id"] += f"-replacement-{round_number:02d}"
        spec["seed"] = integer_seed(original["seed"], "new-situation", round_number)
        spec["mother_task"]["id"] += f":new-situation-{round_number:02d}"
        spec["mother_task"]["data_seed"] = spec["seed"]
        spec["mother_task"]["constraints"] += " This replaces a retired source task. Create a genuinely different within-operation task: change the requested suboperation or boundary conditions, situation and data together, not just identifiers. Prior drafts and labels are not supplied."
        spec["plans"] = [{**item, "quota_slot_id": item["id"], "id": item["id"] + f"-replacement-{round_number:02d}", "seed": integer_seed(item["seed"], round_number, "replacement-data")} for item in missing]
        # Preserve the original preselected review cohort, action and observation
        # assignments, so failures cannot escape independent review by replacement.
        result.append(spec)
    return result


def publish_pool(base, split, plan, started, *, target=None):
    records = [json.loads(path.read_text()) for path in (base / "batches" / split).glob("*.json")]
    rows = [row for record in records for row in record["accepted"]]
    exclusion_path = base / "excluded.json"
    excluded = json.loads(exclusion_path.read_text()) if exclusion_path.exists() else []
    excluded_hashes = {row["content_sha256"] for row in excluded}
    excluded_count = sum(content_fingerprint(row) in excluded_hashes for row in rows)
    rows = [row for row in rows if content_fingerprint(row) not in excluded_hashes]
    rows.sort(key=lambda row: row["id"])
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Accepted pool repeats an episode identity")
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    usage = usage_summary(base / "teacher" / split)
    result = {**plan.binding(), "teacher_contract_version": PROTOCOL, "split": split, "episodes": len(rows), "sha256": sha256(payload), "family_partition_sha256": sha256(PARTITION.read_bytes()), "batch_records": len(records), "completed_batches": sum(record["status"] == "complete" for record in records), "planned_slots_in_batches": sum(len(record["spec"]["plans"]) for record in records), "unfilled_slots": sum(len(record.get("unfilled_ids", [])) for record in records), "quality_paths": dict(Counter(row["provenance"]["quality_path"] for row in rows)), "actual_actions": dict(Counter(row["label"]["decision"] if row["label"]["decision"] == "select" else row["label"]["abstain_reason"] for row in rows)), "observation_variants": dict(Counter(row["provenance"]["observation_variant"] for row in rows)), "teacher_usage": usage, "known_tokens_per_accepted": usage["known_usage"].get("total_tokens", 0) / len(rows) if rows else None, "process_elapsed_seconds": round(time.monotonic() - started, 2), "updated_at": utc_now()}
    planned = {item["id"]: item for record in records for item in record["spec"]["plans"]}
    result["actual_candidate_counts"] = dict(Counter(len(row["entries"]) for row in rows))
    result["candidate_counts_by_action"] = {action: dict(Counter(len(row["entries"]) for row in rows if row["label"]["decision"] == action)) for action in ("select", "abstain")}
    languages = sorted({planned[row["id"]]["context_language"] for row in rows})
    result["candidate_counts_by_language"] = {language: dict(Counter(len(row["entries"]) for row in rows if planned[row["id"]]["context_language"] == language)) for language in languages}
    result["selected_nonempty"] = sum(bool(row["context"]["selectedText"]) for row in rows)
    result["selected_exact_positive"] = sum(bool(row["context"]["selectedText"]) and any(entry["text"] == row["context"]["selectedText"] and entry["id"] in row["label"]["acceptable_ids"] for entry in row["entries"]) for row in rows)
    result["families"] = dict(Counter(row["family_id"] for row in rows))
    result["excluded_after_independent_review"] = excluded_count
    selects = [row for row in rows if row["label"]["decision"] == "select"]
    result["select_all_candidates_positive"] = sum(len(row["label"]["acceptable_ids"]) == len(row["entries"]) for row in selects)
    result["select_mean_acceptable_fraction"] = sum(len(row["label"]["acceptable_ids"]) / len(row["entries"]) for row in selects) / len(selects) if selects else None
    quality_slots = {row["provenance"].get("quota_slot_id", row["id"]) for row in rows}
    if len(quality_slots) != len(rows):
        raise ValueError("More than one accepted situation fills the same logical quota slot")
    result["unique_fulfilled_quota_slots"] = len(quality_slots)
    result["target_episodes"] = target or plan.target(split)
    scenarios = {item["id"]: item["scenario_type"] for record in records for item in record["spec"]["plans"]}
    result["accepted_by_planned_scenario"] = dict(Counter(scenarios[row["id"]] for row in rows))
    result["attempted_slots_by_planned_scenario"] = dict(Counter(scenarios.values()))
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
    parser.add_argument("--backfill-rounds", type=int, default=7, help="At most8 mother situations: the original plus7 new-situation replacements")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--hard-pool", action="store_true", help="New Train-only source pool for Dev-selected v0 mining")
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    if plan.document["teacher_contract_version"] != PROTOCOL:
        raise SystemExit("This producer only runs the registered v7 protocol")
    if args.hard_pool and args.split != "train":
        raise SystemExit("The new hard pool is owned Train data only")
    target = plan.document["hardening"]["pool_episodes"] if args.hard_pool else plan.target(args.split)
    main_base = ROOT / "local/v7" / plan.run_id
    base = main_base / "hard-pool" if args.hard_pool else main_base
    base.mkdir(parents=True, exist_ok=True)
    specs = make_specs(plan, args.split, target=target, namespace="hard-pool" if args.hard_pool else "")
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
    original_specs = specs[:args.max_batches] if args.max_batches else specs
    client = TeacherClient(base / "teacher" / args.split)
    author_cache = {}
    for path in client.audit_dir.glob("*.json"):
        audit = json.loads(path.read_text())
        if audit.get("phase") == "v7-author" and audit.get("status") == "success":
            request_id = audit["request_id"]
            batch_id, attempt = request_id.rsplit("-a", 1)
            author_cache.setdefault(batch_id, {})[int(attempt)] = client._result(audit, cache_hit=True)
    preprocessor = Preprocessor(str(ROOT.parent / "laya-mlx/models/laya-multilingual/tokenizer"))
    registry = ContentRegistry(main_base / "content.sqlite3")
    started = time.monotonic()
    scheduled = 0

    def run_sources(sources):
        nonlocal scheduled
        pending = [spec for spec in sources if not (batch_dir / (spec["batch_id"] + ".json")).is_file() or json.loads((batch_dir / (spec["batch_id"] + ".json")).read_text())["status"] != "complete"]
        scheduled += len(pending)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(produce_batch, spec, client=client, preprocessor=preprocessor, destination=batch_dir / (spec["batch_id"] + ".json"), claim=registry.claim, author_cache=author_cache.get(spec["batch_id"])): spec["batch_id"] for spec in pending}
            try:
                for future in as_completed(futures):
                    future.result()
                    plan.verify_unchanged()
                    result = publish_pool(base, args.split, plan, started, target=target)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    if not args.hard_pool and result["episodes"] >= (plan.target("dev") if args.split == "dev" else plan.document["pilot_episodes"]):
                        rows = [json.loads(line) for line in (base / f"{args.split}.jsonl").read_bytes().splitlines()]
                        for frozen in try_freeze(plan, args.split, rows, preprocessor=preprocessor):
                            print(json.dumps({"frozen": frozen}), flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    run_sources(original_specs)
    if not args.max_batches:
        for round_number in range(1, args.backfill_rounds + 1):
            rows = [json.loads(line) for line in (base / f"{args.split}.jsonl").read_bytes().splitlines()]
            registration = base / f"{args.split}.replacement-round-{round_number:02d}.json"
            if registration.exists():
                replacement = json.loads(registration.read_text())["specs"]
            else:
                replacement = replacement_specs(original_specs, rows, round_number)
                atomic_json(registration, {**plan.binding(), "created_at": utc_now(), "round": round_number, "specs": replacement})
            if not replacement:
                break
            run_sources(replacement)
    final = publish_pool(base, args.split, plan, started, target=target)
    status = "bounded_cost_check_finished" if args.max_batches else "complete" if final["episodes"] == target else "finite_backfill_exhausted"
    atomic_json(base / f"{args.split}.run-completion.json", {**plan.binding(), "updated_at": utc_now(), "scheduled_batches_this_process": scheduled, "cost_check_cap": args.max_batches, "status": status, "episodes": final["episodes"], "target": target})


if __name__ == "__main__":
    main()
