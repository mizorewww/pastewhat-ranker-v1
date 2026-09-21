"""Resume heldout v7 production with finite new-source backfill and data freezing.

This process never loads a student, fits calibration, or scores final Test.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from data_tools.content import ContentRegistry, content_fingerprint
from data_tools.teacher import TeacherClient, TeacherError, atomic_json, audit_source, canonical_bytes, make_teacher_client, observed_responses, utc_now
from data_tools.v7 import produce_batch
from evaluations.common import sha256, write_json
from evaluations.generate_v7 import (
    CONTRACT, PROFILE_SOURCE, RECIPE_SOURCE, planned_batches,
)
from pastewhat_ranker.preprocess import Preprocessor
from run_contract import load_run_plan


def records(directory, run_id):
    return [(path, json.loads(path.read_text())) for path in sorted(directory.rglob(run_id + "-*.json"))]


def retained(directory, plan, split):
    exclusion_path = Path("local/evaluator-quality-exclusions") / (split + ".json")
    exclusions = set(json.loads(exclusion_path.read_text())["content_fingerprints"]) if exclusion_path.exists() else set()
    rows, slots, ids = [], set(), set()
    for path, record in records(directory, plan.run_id):
        if record.get("status") != "complete":
            continue
        spec = record["spec"]
        source_plans = {item["id"]: item for item in spec["plans"]}
        for original in record["accepted"]:
            if content_fingerprint(original) in exclusions:
                continue
            source = source_plans[original["id"]]
            slot = source.get("quota_slot_id", source["id"])
            if slot in slots or original["id"] in ids:
                raise ValueError("Two accepted heldout situations fill the same logical slot")
            slots.add(slot)
            ids.add(original["id"])
            row = copy.deepcopy(original)
            row.update(split=split, group=row["family_id"], synthetic_metadata={
                **plan.binding(), "teacher_contract_version": CONTRACT,
                "family_partition_sha256": sha256("data_tools/family_partition.json"),
                "batch_record_path": str(path), "quota_slot_id": slot,
                "requested_context_language": source["context_language"],
                "planned_candidate_count": source["candidate_count"],
            })
            rows.append(row)
    return sorted(rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())


def usage_summary(directory):
    totals, phases, statuses, models, unknown = Counter(), {}, Counter(), Counter(), Counter()
    provider_usage, failures = {}, Counter()
    elapsed, starts, ends = [], [], []
    authored = 0
    for path in directory.glob("*.json"):
        raw = json.loads(path.read_text())
        statuses[raw.get("status", "unknown")] += 1
        phase = phases.setdefault(raw.get("phase", "unknown"), Counter())
        phase["request_records"] += 1
        for observed in observed_responses(raw):
            response = observed.get("response") or {}
            usage = response.get("usage") or {}
            source = audit_source(observed)
            source_key = canonical_bytes({key: source[key] for key in ("transport", "provider", "requested_model", "response_model")}).decode()
            provider = provider_usage.setdefault(source_key, {"counts": Counter(), "semantics": Counter()})
            provider["counts"]["observed_completions"] += 1
            totals["observed_completions"] += 1
            models[response.get("model", "unspecified")] += 1
            provider["semantics"][observed.get("accounting_semantics", "Kimi-reported usage; reasoning is included in completion_tokens")] += 1
            if usage:
                phase["known_responses"] += 1
                totals["known_responses"] += 1
                provider["counts"]["responses_with_reported_usage"] += 1
                for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(name)
                    if type(value) in (int, float) and value >= 0:
                        totals[name] += value
                        phase[name] += value
                        provider["counts"][name] += value
                        provider["counts"][name + "_reported_responses"] += 1
                reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                if type(reasoning) in (int, float) and reasoning >= 0:
                    totals["reported_reasoning_tokens"] += reasoning
                    phase["reported_reasoning_tokens"] += reasoning
                    provider["counts"]["reported_reasoning_tokens"] += reasoning
                    provider["counts"]["reasoning_reported_responses"] += 1
            else:
                unknown["observed_completion_without_reported_usage"] += 1
                provider["counts"]["responses_without_reported_usage"] += 1
            for name in ("cacheRead", "cacheWrite"):
                value = (observed.get("raw_usage") or {}).get(name)
                if type(value) in (int, float) and value >= 0:
                    provider["counts"]["reported_" + name] += value
            if observed.get("elapsed_seconds") is not None:
                elapsed.append(observed["elapsed_seconds"])
                phase["elapsed_seconds"] += observed["elapsed_seconds"]
            if observed.get("started_at"):
                starts.append(observed["started_at"])
            if observed.get("completed_at"):
                ends.append(observed["completed_at"])
            if observed.get("phase") == "v7-author" and observed.get("status") == "success":
                parsed = TeacherClient._result(observed, cache_hit=True).parsed
                authored += len(parsed.get("episodes", [])) if isinstance(parsed, dict) else 0
        for attempt in raw.get("attempts", []):
            category = attempt.get("error_type") or ("HTTP_" + str(attempt["http_status"]) if attempt.get("http_status") else None)
            if category:
                failures[category] += 1
                if attempt.get("usage_known") is not True:
                    unknown[category] += 1
        if raw.get("status") == "request_started":
            unknown["request_started_unobserved_completion"] += 1
    return {"request_records": sum(statuses.values()), "statuses": dict(statuses),
            "known_usage": dict(totals), "by_phase": {key: dict(value) for key, value in phases.items()},
            "by_provider_model": [{**json.loads(source), **dict(value["counts"]), "accounting_semantics": dict(value["semantics"])}
                                  for source, value in sorted(provider_usage.items())],
            "unknown_usage_attempt_events": dict(unknown), "response_models": dict(models),
            "failure_attempt_events": dict(failures),
            "authored_draft_rows_including_repairs": authored,
            "first_request_at": min(starts, default=None), "latest_completion_at": max(ends, default=None),
            "summed_request_elapsed_seconds": sum(elapsed),
            "response_accounting": "Each distinct observed completion, including retained invalid responses; successful cache reads do not add usage. Transport/HTTP unknown attempts come only from the top-level accumulated attempt ledger.",
            "reasoning_accounting": "Only explicitly reported reasoning counters are summed; Kimi includes them in completion_tokens. Pi reasoning and cache inclusion are not inferred.",
            "token_totals_are_reported_counters_not_a_billing_estimate": True}


def teacher_source_counts(rows, audit_directory):
    sources = {}
    counts = {role: Counter() for role in ("author", "primary", "review")}
    for row in rows:
        for role, key in (("author", "author_audit_id"), ("primary", "label_audit_id"), ("review", "review_audit_id")):
            identifier = row["provenance"].get(key)
            if not identifier:
                continue
            if identifier not in sources:
                sources[identifier] = audit_source(json.loads((audit_directory / (identifier + ".json")).read_text()))
            counts[role][canonical_bytes(sources[identifier]).decode()] += 1
    return {role: [{**json.loads(source), "episodes": count} for source, count in sorted(values.items())]
            for role, values in counts.items() if values}


def progress(directory, plan, split):
    rows = retained(directory, plan, split)
    completed = [record for _, record in records(directory, plan.run_id) if record.get("status") == "complete"]
    actions = lambda row: "select" if row["label"]["decision"] == "select" else "no_match" if row["label"]["abstain_reason"] == "no_match" else "missing_intent"
    languages = {row["synthetic_metadata"]["requested_context_language"] for row in rows}
    report = {**plan.binding(), "teacher_contract_version": CONTRACT, "split": split,
        "updated_at": utc_now(), "retained_unique": len(rows), "target": plan.target(split),
        "completed_batch_records": len(completed),
        "rejection_event_counts": dict(Counter(str(row["reason"]) for record in completed for row in record["rejected"])),
        "quality_paths": dict(Counter(row["provenance"]["quality_path"] for row in rows)),
        "teacher_sources": teacher_source_counts(rows, Path("local/teacher-v7") / plan.run_id / split),
        "actual_actions": dict(Counter(actions(row) for row in rows)),
        "families": dict(Counter(row["family_id"] for row in rows)),
        "observation_variants": dict(Counter(row["provenance"]["observation_variant"] for row in rows)),
        "actual_candidate_counts": dict(Counter(len(row["entries"]) for row in rows)),
        "candidate_counts_by_action": {action: dict(Counter(len(row["entries"]) for row in rows if actions(row) == action)) for action in ("select", "no_match", "missing_intent")},
        "candidate_counts_by_requested_language": {language: dict(Counter(len(row["entries"]) for row in rows if row["synthetic_metadata"]["requested_context_language"] == language)) for language in sorted(languages)},
        "selected_nonempty": sum(bool(row["context"]["selectedText"]) for row in rows),
        "selected_exact_positive": sum(bool(row["context"]["selectedText"]) and any(entry["text"] == row["context"]["selectedText"] and entry["id"] in row["label"]["acceptable_ids"] for entry in row["entries"]) for row in rows),
        "usage": usage_summary(Path("local/teacher-v7") / plan.run_id / split),
        "student_inference_used": False, "human_validated": False}
    atomic_json(directory / "production-progress.json", report)
    return rows, report


def replacement_specs(originals, rows, round_number):
    filled = {row["synthetic_metadata"]["quota_slot_id"] for row in rows}
    sources = []
    for original in originals:
        missing = [item for item in original["plans"] if item["id"] not in filled]
        if not missing:
            continue
        spec = copy.deepcopy(original)
        spec["batch_id"] += f"-replacement-{round_number:02d}"
        spec["seed"] = int(hashlib.sha256(f"{original['seed']}:replacement:{round_number}".encode()).hexdigest()[:12], 16)
        spec["mother_task"]["id"] += f":new-situation-{round_number:02d}"
        spec["mother_task"]["data_seed"] = spec["seed"]
        spec["mother_task"]["constraints"] += " Create a new situation within this operation, changing the goal boundary and operands together. Earlier rejected drafts and labels are not provided."
        spec["plans"] = [{**item, "quota_slot_id": item["id"],
            "id": item["id"] + f"-replacement-{round_number:02d}",
            "seed": int(hashlib.sha256(f"{item['seed']}:replacement:{round_number}".encode()).hexdigest()[:12], 16)} for item in missing]
        # Review cohort and all quota factors stay fixed across replacements.
        sources.append(spec)
    return sources


def author_caches(specifications, client):
    """Scan historical response files once per resumed wave, not once per slot."""
    sources = {spec["batch_id"]: spec for spec in specifications}
    caches = {}
    for path in client.audit_dir.glob("*.json"):
        raw = json.loads(path.read_text())
        if raw.get("phase") != "v7-author" or raw.get("status") != "success":
            continue
        identity = raw.get("request_id", "").rsplit("-a", 1)
        if len(identity) != 2 or identity[0] not in sources or identity[1] not in {"0", "1"}:
            continue
        spec = sources[identity[0]]
        request = json.loads(raw["request"]["messages"][1]["content"])
        planned = {item["id"]: item for item in spec["plans"]}
        if request.get("mother_task") != spec["mother_task"] or request.get("field_profile") != spec["profile"] or any(planned.get(item["id"]) != item for item in request["plans"]):
            raise ValueError("A cached author response belongs to a different source")
        caches.setdefault(identity[0], {}).setdefault(int(identity[1]), TeacherClient._result(raw, cache_hit=True))
    return caches


def freeze_data(directory, plan, split, tokenizer):
    from evaluations.audit_v7 import audit_dataset
    rows, report = progress(directory, plan, split)
    if len(rows) != plan.target(split):
        return False
    output = plan.data_path(split)
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    if output.exists() and output.read_bytes() != payload:
        raise ValueError("The frozen heldout dataset cannot be overwritten")
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".building")
        temporary.write_bytes(payload)
        temporary.replace(output)
    audit = audit_dataset(output, plan=plan, split=split, tokenizer=tokenizer, partition="data_tools/family_partition.json")
    audit_path = Path("reports/evaluation") / plan.run_id / ("data-" + split + "-audit.json")
    write_json(audit_path, audit)
    manifest = {**plan.binding(), "split": split, "episodes": len(rows), "sha256": sha256(output),
        "teacher_contract_version": CONTRACT, "family_partition_sha256": sha256("data_tools/family_partition.json"),
        "families": report["families"], "actual_actions": report["actual_actions"],
        "quality_paths": report["quality_paths"], "observation_variants": report["observation_variants"],
        "teacher_sources": audit["teacher_sources"],
        "teacher_transition": {"path": "configs/teacher_transition_swe2.json", "sha256": sha256("configs/teacher_transition_swe2.json")},
        "resource_supplement": audit["resource_supplement"],
        "teacher_correction": audit["teacher_correction"],
        "audit_path": str(audit_path), "audit_sha256": sha256(audit_path),
        "teacher_audit_bundle_sha256": audit["teacher_audit_bundle_sha256"],
        "human_validated": False, "student_inference_used": False}
    public = Path("data/evaluator-manifests") / plan.run_id
    write_json(public / (split + ".manifest.json"), manifest)
    write_json(public / (split + ".fingerprints.json"), {**plan.binding(), "content_sha256": sorted(content_fingerprint(row) for row in rows)})
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--workers", type=int, default=1, choices=(1, 2))
    parser.add_argument("--max-situations", type=int, default=8, choices=range(1, 9))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--tokenizer", type=Path, default=Path("../laya-mlx/models/laya-multilingual/tokenizer"))
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    if plan.document["teacher_contract_version"] != CONTRACT:
        raise SystemExit("Full heldout producer requires the registered v7 run")
    directory = Path("local/evaluator-v7") / plan.run_id / args.split
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".production.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = json.loads((directory / "owner-binding.json").read_text())
    if any(binding.get(key) != value for key, value in {**plan.binding(), "recipe_sha256": sha256(RECIPE_SOURCE), "profile_sha256": sha256(PROFILE_SOURCE)}.items()):
        raise ValueError("Heldout author sources changed after their first batch")
    partition = json.loads(Path("data_tools/family_partition.json").read_text())
    originals = planned_batches(plan, partition, args.split, 10)
    sampling_path = directory / "sampling.json"
    sampling = {**plan.binding(), "teacher_contract_version": CONTRACT, "batch_size": 10, "max_situations": args.max_situations, "specs": originals}
    if sampling_path.exists() and json.loads(sampling_path.read_text()) != sampling:
        raise ValueError("The registered heldout source sampling changed")
    if not sampling_path.exists():
        atomic_json(sampling_path, sampling)
    if args.plan_only:
        print(json.dumps({"split": args.split, "episodes": sum(len(spec["plans"]) for spec in originals), "sampling_sha256": sha256(sampling_path)}))
        return
    # A previous bounded cost-check process must finish naturally before this
    # full owner starts; process.json is its explicit per-split ownership file.
    previous = directory / "process.json"
    if previous.exists():
        prior_pid = json.loads(previous.read_text())["pid"]
        try:
            os.kill(prior_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit("The bounded heldout producer is still active; wait for natural completion")
    atomic_json(directory / "production-process.json", {**plan.binding(), "pid": os.getpid(), "started_at": utc_now(), "workers": args.workers, "max_situations": args.max_situations, "student_inference_used": False})
    client = make_teacher_client(Path("local/teacher-v7") / plan.run_id / args.split)
    preprocessor = Preprocessor(args.tokenizer)
    registry = ContentRegistry(directory.parent / "heldout-content.sqlite3")
    for _, record in records(directory, plan.run_id):
        for row in record["accepted"]:
            if registry.claim(row):
                raise ValueError("Existing heldout accepted content is duplicated")
    exclusion_path = Path("local/evaluator-quality-exclusions") / (args.split + ".json")
    def claim(row):
        excluded = json.loads(exclusion_path.read_text())["content_fingerprints"] if exclusion_path.exists() else []
        return "Independently excluded content" if content_fingerprint(row) in excluded else registry.claim(row)

    def run_sources(sources):
        pending = list(sources)
        failures = 0
        while pending:
            account = client.coordinator.status()
            if account["paused"]:
                time.sleep(30)
                continue
            pending = [spec for spec in pending if not (directory / (spec["batch_id"] + ".json")).exists() or json.loads((directory / (spec["batch_id"] + ".json")).read_text()).get("status") != "complete"]
            if not pending:
                break
            cached = author_caches(pending, client)
            try:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = [executor.submit(produce_batch, spec, client=client, preprocessor=preprocessor,
                        destination=directory / (spec["batch_id"] + ".json"), claim=claim,
                        author_cache=cached.get(spec["batch_id"])) for spec in pending]
                    try:
                        for future in as_completed(futures):
                            future.result()
                            failures = 0
                            plan.verify_unchanged()
                            _, report = progress(directory, plan, args.split)
                            print(json.dumps({key: report[key] for key in ("updated_at", "split", "retained_unique", "target", "quality_paths", "actual_actions")}), flush=True)
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        raise
            except TeacherError:
                failures += 1
                if failures >= 3 and not client.coordinator.status()["paused"]:
                    raise
                time.sleep(30)

    run_sources(originals)
    for round_number in range(1, args.max_situations):
        rows, _ = progress(directory, plan, args.split)
        registration = directory / f"replacement-round-{round_number:02d}.json"
        if registration.exists():
            sources = json.loads(registration.read_text())["specs"]
        else:
            sources = replacement_specs(originals, rows, round_number)
            atomic_json(registration, {**plan.binding(), "round": round_number, "created_at": utc_now(), "specs": sources})
        if not sources:
            break
        run_sources(sources)
    frozen = freeze_data(directory, plan, args.split, args.tokenizer)
    rows, report = progress(directory, plan, args.split)
    atomic_json(directory / "production-completion.json", {**plan.binding(), "status": "data_frozen_and_audited" if frozen else "finite_backfill_exhausted", "retained_unique": len(rows), "target": plan.target(args.split), "finished_at": utc_now(), "student_inference_used": False})


if __name__ == "__main__":
    main()
