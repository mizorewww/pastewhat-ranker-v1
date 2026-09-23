"""Owned Train/Dev stream, with a bounded initial cost check and durable batches."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data_tools.authoring import candidate_space, owned_profile
from data_tools.content import ContentRegistry, content_fingerprint
from data_tools.freeze import publish_bytes
from data_tools.freeze_v7 import try_freeze
from data_tools.pi_teacher import pi_coordinator
from data_tools.rate_limit import AccountCoordinator
from data_tools.resources import bounded_futures, load_resources, worker_budget
from data_tools.teacher import (
    atomic_json,
    audit_source,
    canonical_bytes,
    make_teacher_client,
    observed_responses,
    sha256,
    teacher_source_counts,
    utc_now,
)
from data_tools.v7 import PROTOCOL, produce_batch
from pastewhat_ranker.preprocess import Preprocessor
from run_contract import action_quotas, family_quotas, load_run_plan

ROOT = Path(__file__).resolve().parents[1]
PARTITION = ROOT / "data_tools/family_partition.json"
DEV_MISSING_INTENT_GATE_VERSION = "dev-missing-intent-required-parameter-v2"
DEV_UNOBSERVED_INTENT_GATE_VERSION = "dev-missing-intent-unobserved-v3"
TRAIN_MISSING_INTENT_GATE_VERSION = "train-missing-intent-required-parameter-v1"


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
    totals, phases, statuses, models, providers = Counter(), {}, Counter(), Counter(), {}
    transport_unknown, http_without_usage, started = 0, 0, Counter()
    observed_count, prior_count = 0, 0
    with AccountCoordinator()._state() as state:
        leases = dict(state["leases"])
    with pi_coordinator()._state() as state:
        leases.update(state["leases"])
    for path in directory.glob("*.json"):
        audit = json.loads(path.read_text())
        statuses[audit.get("status", "unknown")] += 1
        prior_count += len(audit.get("prior_observed_responses", []))
        for observed in observed_responses(audit):
            observed_count += 1
            usage = (observed.get("response") or {}).get("usage") or {}
            source = audit_source(observed)
            provider = providers.setdefault(source["provider"] + "/" + source["response_model"], {"known_usage": Counter(), "observed_responses": 0, "missing_usage_fields": Counter()})
            provider["observed_responses"] += 1
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if key in usage:
                    provider["known_usage"][key] += usage[key]
                else:
                    provider["missing_usage_fields"][key] += 1
            if usage:
                phase = phases.setdefault(observed.get("phase", "unknown"), Counter())
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    totals[key] += usage.get(key, 0)
                    phase[key] += usage.get(key, 0)
                reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                if reasoning is not None:
                    totals["reported_reasoning_tokens"] += reasoning
                    phase["reported_reasoning_tokens"] += reasoning
                models[(observed.get("response") or {}).get("model", "unknown")] += 1
        transport_unknown += sum("error_type" in attempt and not attempt.get("http_status") and attempt.get("usage_known") is not True for attempt in audit.get("attempts", []))
        http_without_usage += sum(bool(attempt.get("http_status")) and attempt.get("usage_known") is not True for attempt in audit.get("attempts", []))
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
    return {"request_records": sum(statuses.values()), "observed_response_records": observed_count, "prior_observed_response_records": prior_count, "known_usage": dict(totals), "by_phase": {key: dict(value) for key, value in phases.items()}, "by_provider": providers, "statuses": dict(statuses), "response_models": dict(models), "transport_attempts_without_usage": transport_unknown, "http_error_attempts_without_usage": http_without_usage, "unfinished_started_requests": dict(started)}


def replacement_specs(originals, rows, round_number, *, split="train"):
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
        if split == "train" and round_number >= 2 and any(item["scenario_type"] in ("ambiguous", "insufficient_context") for item in missing):
            spec["source_constraint_version"] = TRAIN_MISSING_INTENT_GATE_VERSION
            spec["mother_task"]["constraints"] += (
                " For ambiguous or insufficient_context plans with standard observation,"
                " visible guidance must state that a required exact decision parameter"
                " (such as target, scope, recipient, or format) must be known before"
                " pasting, but its value is absent from every visible field. Include"
                " plausible complete candidates for mutually exclusive parameter"
                " values; none is justified until the missing value is supplied."
                " Do not substitute an optional preference, and do not make every"
                " candidate invalid. If only one candidate is registered, leave"
                " another plausible value unrepresented. For each such standard"
                " episode add an author-only decision_gate with exact keys parameter,"
                " visible_requirement_quote, possible_values, candidate_value_indices."
                " The quote must appear verbatim in guidance after observation trimming;"
                " possible_values lists at least two distinct short values; mapping"
                " items have {index: zero-based candidate position, value: one possible"
                " value}. With multiple candidates map at least two distinct values."
                " For no_accessibility or generic_field missing-intent plans instead"
                " emit guidance=[] and selected='' exactly, keep the registered"
                " candidate count and directly pasteable alternatives, and emit no"
                " decision_gate. Do not add an intent, answer hint, or label."
                " Preserve every other plan's registered action scenario."
            )
        if split == "dev" and round_number >= 2 and any(item["scenario_type"] in ("ambiguous", "insufficient_context") for item in missing):
            unobserved = round_number >= 8 and all(item["observation_variant"] != "standard" for item in missing)
            if unobserved:
                spec["source_constraint_version"] = DEV_UNOBSERVED_INTENT_GATE_VERSION
                spec["mother_task"]["constraints"] += (
                    " These registered observation variants remove all authored guidance and"
                    " selection before the teacher and student see the fixture. Emit guidance=[]"
                    " and selected='' exactly. Keep the registered candidate count, and make"
                    " each candidate a complete pasteable value from this family. Do not put"
                    " an intent, an answer hint, a decision_gate, or a label in the fixture."
                    " The independent labeler must judge the resulting information-poor view."
                )
            elif round_number >= 3:
                spec["source_constraint_version"] = DEV_MISSING_INTENT_GATE_VERSION
                spec["mother_task"]["constraints"] += (
                    " For ambiguous or insufficient_context plans, visible guidance must state"
                    " that the pasted result must match a required exact decision parameter,"
                    " such as recipient policy, target variant, scope or format. Its value is"
                    " absent from every visible field; no candidate can be adopted until it is"
                    " supplied. Construct at least two plausible, mutually exclusive parameter"
                    " values and candidates for different values. Do not treat optional style"
                    " or several interchangeable answers to an already complete request as"
                    " missing intent. A one-candidate insufficient_context plan must leave"
                    " another plausible value unrepresented. Preserve every other plan's own"
                    " registered action scenario. Add a non-visible decision_gate object to"
                    " each such episode with exact keys parameter, visible_requirement_quote,"
                    " possible_values, candidate_value_indices. The quote must occur verbatim"
                    " in guidance after observation trimming; possible_values lists at least"
                    " two distinct short values; candidate_value_indices lists {index:"
                    " zero-based candidate position, value: one possible value}. With two or"
                    " more candidates, map at least two to distinct values. This is author"
                    " metadata only; never put it, labels or rationale into the paste fixture."
                )
            else:
                spec["source_constraint_version"] = "dev-missing-intent-observable-alternatives-v1"
                spec["mother_task"]["constraints"] += (
                    " For ambiguous or insufficient_context plans, make at least one candidate"
                    " directly usable for a plausible goal within this operation, but do not"
                    " state the decisive goal, fact or preference in visible guidance or selected"
                    " text. For ambiguous plans with at least two candidates, include different"
                    " directly usable candidates for at least two plausible goals; the visible"
                    " information must not select between those goals. For insufficient_context"
                    " plans, omit the decisive fact needed to choose a usable candidate rather"
                    " than making every candidate invalid. Preserve each other plan's own"
                    " registered action scenario. Do not put labels or explanations in the fixture."
                )
        # Preserve the original preselected review cohort, action and observation
        # assignments, so failures cannot escape independent review by replacement.
        result.append(spec)
    return result


def dev_missing_intent_prelabel_gate(raw, spec, pending, prepared):
    """Admit only structurally checkable versioned drafts to blind labeling."""
    version = spec.get("source_constraint_version")
    if version not in {DEV_MISSING_INTENT_GATE_VERSION, DEV_UNOBSERVED_INTENT_GATE_VERSION, TRAIN_MISSING_INTENT_GATE_VERSION}:
        return prepared, []
    drafts = {row.get("slot"): row for row in raw.get("episodes", []) if isinstance(row, dict)} if isinstance(raw, dict) else {}
    plans = {plan["id"]: plan for plan in pending}
    kept, errors = [], []
    for row in prepared:
        plan = plans[row["id"]]
        if plan["scenario_type"] not in ("ambiguous", "insufficient_context"):
            kept.append(row)
            continue
        draft = drafts.get(row["id"], {})
        if version == TRAIN_MISSING_INTENT_GATE_VERSION and plan["observation_variant"] != "standard":
            context = row["context"]
            if (not isinstance(draft.get("candidates"), list)
                    or len(draft["candidates"]) != plan["candidate_count"]
                    or draft.get("guidance") != [] or draft.get("selected") != ""
                    or "decision_gate" in draft
                    or context["selectedText"] or context["surroundingText"]
                    or context["isSecure"]):
                errors.append({"id": row["id"], "reason": "Train unobserved-intent fixture retained guidance or changed its registered candidates"})
            else:
                kept.append(row)
            continue
        if version == DEV_UNOBSERVED_INTENT_GATE_VERSION:
            context = row["context"]
            if (plan["observation_variant"] == "standard"
                    or plan["candidate_count"] != 1
                    or not isinstance(draft.get("candidates"), list)
                    or len(draft["candidates"]) != 1
                    or draft.get("guidance") != [] or draft.get("selected") != ""
                    or context["selectedText"] or context["surroundingText"]
                    or context["isSecure"]):
                errors.append({"id": row["id"], "reason": "Unobserved-intent fixture retained task evidence or changed its registered candidate count"})
            else:
                kept.append(row)
            continue
        gate = draft.get("decision_gate")
        problem = None
        if not isinstance(gate, dict) or set(gate) != {"parameter", "visible_requirement_quote", "possible_values", "candidate_value_indices"}:
            problem = "Missing required-parameter gate metadata"
        else:
            parameter, quote = gate["parameter"], gate["visible_requirement_quote"]
            values, mapping = gate["possible_values"], gate["candidate_value_indices"]
            candidates = draft.get("candidates")
            visible = json.dumps(row["context"], ensure_ascii=False)
            if not isinstance(parameter, str) or not 3 <= len(parameter.strip()) <= 80:
                problem = "Invalid required-parameter name"
            elif not isinstance(quote, str) or not 10 <= len(quote.strip()) <= 180 or quote not in visible:
                problem = "Required-parameter statement is not student-visible"
            elif (version == TRAIN_MISSING_INTENT_GATE_VERSION
                  and (not isinstance(draft.get("guidance"), list)
                       or not any(isinstance(line, str) and quote in line for line in draft["guidance"]))):
                problem = "Train required-parameter statement is absent from authored guidance"
            elif not isinstance(candidates, list) or len(candidates) != plan["candidate_count"]:
                problem = "Required-parameter candidate count differs from plan"
            elif not isinstance(values, list) or not 2 <= len(values) <= 8 or any(not isinstance(value, str) or not value.strip() or len(value) > 80 for value in values) or len({value.casefold().strip() for value in values}) != len(values):
                problem = "Required-parameter values are not distinct"
            elif not isinstance(mapping, list) or not mapping or any(not isinstance(item, dict) or set(item) != {"index", "value"} or type(item["index"]) is not int or item["index"] < 0 or item["index"] >= len(candidates) or item["value"] not in values for item in mapping):
                problem = "Required-parameter candidate mapping is invalid"
            elif len({item["index"] for item in mapping}) != len(mapping) or (len(candidates) >= 2 and len({item["value"] for item in mapping}) < 2):
                problem = "Required-parameter alternatives are not mapped"
        if problem:
            errors.append({"id": row["id"], "reason": problem})
        else:
            kept.append(row)
    return kept, errors


def prioritize_pilot_sources(pending, rows, partition, pilot_count, batch_dir):
    """Cover missing frozen Pilot strata with already registered original sources.

    This changes only submission order. Batch specifications and their identities
    remain those in the immutable sampling registration.
    """
    quotas = action_quotas(family_quotas(partition, "train", pilot_count))
    missing = {(family, action): count for family, buckets in quotas.items()
               for action, count in buckets.items()}
    for row in rows:
        label = row["label"]
        action = "select" if label["decision"] == "select" else "no_match" if label["abstain_reason"] == "no_match" else "missing_intent"
        key = (row["family_id"], action)
        missing[key] = max(0, missing[key] - 1)
    if not any(missing.values()):
        return pending

    priority, remaining = [], []
    for spec in pending:
        path = batch_dir / (spec["batch_id"] + ".json")
        record = json.loads(path.read_text()) if path.is_file() else {}
        done = {row["id"] for row in record.get("accepted", [])}
        done.update(record.get("unrecoverable_label_ids", []))
        targets_missing_stratum = any(
            plan["id"] not in done and missing.get((spec["family_id"], "missing_intent" if plan["scenario_type"] in ("ambiguous", "insufficient_context") else plan["scenario_type"]), 0)
            for plan in spec["plans"]
        )
        (priority if targets_missing_stratum else remaining).append(spec)
    # A planned slot is only an opportunity: author rejection or independent
    # labeling may leave it empty. Keep every backup source ahead of unrelated
    # work until the actual accepted pool closes the stratum.
    return priority + remaining


def prioritize_diagnostic_sources(pending, rows, partition, diagnostic_count, batch_dir):
    """Advance registered 10k strata, retaining backups for authoring failures."""
    quotas = action_quotas(family_quotas(partition, "train", diagnostic_count))
    missing = {(family, action): count for family, buckets in quotas.items()
               for action, count in buckets.items()}
    for row in rows:
        label = row["label"]
        action = "select" if label["decision"] == "select" else "no_match" if label["abstain_reason"] == "no_match" else "missing_intent"
        key = (row["family_id"], action)
        missing[key] = max(0, missing[key] - 1)
    targets = {key for key, count in missing.items() if count}
    if not targets:
        return pending

    supplies = []
    for index, spec in enumerate(pending):
        path = batch_dir / (spec["batch_id"] + ".json")
        record = json.loads(path.read_text()) if path.is_file() else {}
        done = {row["id"] for row in record.get("accepted", [])}
        done.update(record.get("unrecoverable_label_ids", []))
        supply = Counter((spec["family_id"], "missing_intent" if plan["scenario_type"] in ("ambiguous", "insufficient_context") else plan["scenario_type"])
                         for plan in spec["plans"] if plan["id"] not in done)
        supplies.append((index, spec, supply))

    primary = []
    available = supplies.copy()
    while available and any(missing.values()):
        chosen = max(range(len(available)), key=lambda position: (
            sum(min(missing.get(key, 0), count) for key, count in available[position][2].items()),
            -available[position][0],
        ))
        index, spec, supply = available.pop(chosen)
        gain = sum(min(missing.get(key, 0), count) for key, count in supply.items())
        if not gain:
            available.insert(chosen, (index, spec, supply))
            break
        primary.append(spec)
        for key, count in supply.items():
            missing[key] = max(0, missing.get(key, 0) - count)
    # Hypothetical coverage is not accepted coverage. Keep every remaining
    # registered source for initially missing strata before unrelated batches.
    backups = [spec for _, spec, supply in available if targets.intersection(supply)]
    rest = [spec for _, spec, supply in available if not targets.intersection(supply)]
    return primary + backups + rest


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
    result["teacher_sources"] = teacher_source_counts(rows, base / "teacher" / split)
    transition = ROOT / "configs/teacher_transition_swe2.json"
    result["teacher_transition"] = {"path": str(transition.relative_to(ROOT)), "sha256": sha256(transition.read_bytes())}
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
    parser.add_argument("--backfill-rounds", type=int, default=7, help="Maximum registered replacement situations for each missing original slot")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--hard-pool", action="store_true", help="New Train-only source pool for Dev-selected v0 mining")
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    resources = load_resources(plan)
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
    client = make_teacher_client(base / "teacher" / args.split)
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

    def refresh():
        # Batch records are the durable source of truth. A restart may find
        # every batch complete even if the previous process exited before its
        # rolling-pool publication or final snapshot rename.
        result = publish_pool(base, args.split, plan, started, target=target)
        result["scheduler"] = {"submitted_task_limit": worker_budget(plan, args.split, args.workers, hard_pool=args.hard_pool), "executor_max_workers": args.workers, "resource_supplement": resources[1] if resources else None}
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not args.hard_pool and result["episodes"] >= (plan.target("dev") if args.split == "dev" else 1000):
            rows = [json.loads(line) for line in (base / f"{args.split}.jsonl").read_bytes().splitlines()]
            for frozen in try_freeze(plan, args.split, rows, preprocessor=preprocessor):
                print(json.dumps({"frozen": frozen}), flush=True)
        return result

    def run_sources(sources):
        nonlocal scheduled
        pending = [spec for spec in sources if not (batch_dir / (spec["batch_id"] + ".json")).is_file() or json.loads((batch_dir / (spec["batch_id"] + ".json")).read_text())["status"] != "complete"]
        if sources is original_specs and args.split == "train" and not args.hard_pool:
            rows = [json.loads(line) for line in (base / "train.jsonl").read_bytes().splitlines()]
            partition = json.loads(PARTITION.read_text())
            if not (ROOT / plan.data_path("pilot")).is_file():
                pending = prioritize_pilot_sources(pending, rows, partition, plan.document["pilot_episodes"], batch_dir)
            elif not (ROOT / plan.data_path("diagnostic")).is_file():
                pending = prioritize_diagnostic_sources(pending, rows, partition, plan.document["diagnostic_episodes"], batch_dir)
        scheduled += len(pending)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            submit = lambda pool, spec: pool.submit(produce_batch, spec, client=client, preprocessor=preprocessor, destination=batch_dir / (spec["batch_id"] + ".json"), claim=registry.claim, author_cache=author_cache.get(spec["batch_id"]), prelabel_gate=dev_missing_intent_prelabel_gate)
            capacity = lambda: worker_budget(plan, args.split, args.workers, hard_pool=args.hard_pool)
            for future in bounded_futures(executor, pending, submit, capacity):
                future.result()
                plan.verify_unchanged()
                refresh()

    # Publish already accepted durable work before planning replacements or
    # sending a request. This also completes a crash-interrupted final freeze.
    refresh()
    run_sources(original_specs)
    if not args.max_batches:
        for round_number in range(1, args.backfill_rounds + 1):
            rows = [json.loads(line) for line in (base / f"{args.split}.jsonl").read_bytes().splitlines()]
            registration = base / f"{args.split}.replacement-round-{round_number:02d}.json"
            if registration.exists():
                replacement = json.loads(registration.read_text())["specs"]
            else:
                replacement = replacement_specs(original_specs, rows, round_number, split=args.split)
                atomic_json(registration, {**plan.binding(), "created_at": utc_now(), "round": round_number, "specs": replacement})
            if not replacement:
                break
            run_sources(replacement)
    final = refresh()
    status = "bounded_cost_check_finished" if args.max_batches else "complete" if final["episodes"] == target else "finite_backfill_exhausted"
    atomic_json(base / f"{args.split}.run-completion.json", {**plan.binding(), "updated_at": utc_now(), "scheduled_batches_this_process": scheduled, "cost_check_cap": args.max_batches, "backfill_rounds": args.backfill_rounds, "status": status, "episodes": final["episodes"], "target": target})


if __name__ == "__main__":
    main()
