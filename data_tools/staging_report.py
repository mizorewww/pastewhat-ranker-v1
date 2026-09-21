"""Measure the actual Train/Dev native-payload staging throughput and failures."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

from data_tools.deployment import placement_issue
from data_tools.generate import CACHE_VERSION, GENERATOR_SYSTEM, ROOT
from data_tools.teacher import atomic_json, utc_now


def report_split(split):
    audits = []
    for path in (ROOT / "local/teacher" / split / "main").glob("*.json"):
        audit = json.loads(path.read_text())
        audit["_path"] = str(path.relative_to(ROOT))
        audits.append(audit)
    starts = [audit["started_at"] for audit in audits if audit.get("request", {}).get("messages", [{}])[0].get("content") == GENERATOR_SYSTEM]
    start = min(starts) if starts else None
    current = [audit for audit in audits if start and audit.get("started_at", "") >= start]
    episodes, rejections = {}, []
    for path in (ROOT / "local/generated" / split / ("main-" + CACHE_VERSION)).glob("*.json"):
        if path.name == "failures.json":
            continue
        record = json.loads(path.read_text())
        for episode in record.get("episodes", []):
            if not placement_issue(episode):
                episodes[episode["id"]] = episode
        rejections.extend(record.get("rejected", []))
    elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(start)).total_seconds() if start else 0
    usage = Counter()
    phases = defaultdict(list)
    requests = Counter()
    attempted_ids = set()
    attempted_counts = Counter()
    for audit in current:
        requests[audit.get("status", "unknown")] += 1
        usage.update({key: value for key, value in audit.get("response", {}).get("usage", {}).items() if isinstance(value, (int, float))})
        if "elapsed_seconds" in audit:
            phases[audit.get("phase", "unknown")].append(audit["elapsed_seconds"])
        if audit.get("request", {}).get("messages", [{}])[0].get("content") == GENERATOR_SYSTEM:
            author = json.loads(audit["request"]["messages"][1]["content"])
            for plan in author["plans"]:
                attempted_ids.add(plan["id"])
                attempted_counts[plan["candidate_count"]] += 1
    accepted = len(episodes)
    target = 20000 if split == "train" else 1000
    estimated = elapsed / accepted * target if accepted else None
    return {
        "split": split, "started_at": start, "elapsed_seconds": elapsed,
        "accepted_episodes": accepted, "distinct_requested_slots": len(attempted_ids),
        "accepted_fraction_of_requested_slots": accepted / len(attempted_ids) if attempted_ids else None,
        "accepted_per_hour": accepted / elapsed * 3600 if elapsed else 0,
        "known_api_tokens": dict(usage),
        "known_total_tokens_per_accepted_episode": usage["total_tokens"] / accepted if accepted else None,
        "estimated_seconds_for_full_target_at_observed_rate": estimated,
        "estimated_total_tokens_for_full_target_at_observed_cost": usage["total_tokens"] / accepted * target if accepted else None,
        "request_statuses": dict(requests),
        "request_phases": {phase: {"completed_observed": len(values), "median_seconds": statistics.median(values), "sum_seconds": sum(values)} for phase, values in phases.items()},
        "rejection_event_types": dict(Counter(item.get("type", "blind_family_or_deployment_review") for item in rejections)),
        "rejection_findings": dict(Counter(str(item.get("finding", item.get("review", {}).get("reason", item.get("type", "unspecified")))) for item in rejections)),
        "candidate_counts_per_author_attempt": dict(attempted_counts),
        "accepted_candidate_counts": dict(Counter(len(episode["entries"]) for episode in episodes.values())),
        "accepted_families": dict(Counter(episode["family_id"] for episode in episodes.values())),
        "accepted_labels": dict(Counter(episode["label"]["decision"] if episode["label"]["decision"] == "select" else episode["label"]["abstain_reason"] for episode in episodes.values())),
        "accepted_ids": sorted(episodes),
    }


def main():
    report = {"created_at": utc_now(), "dataset_protocol": "teacher-episodes-v5-native-payload", "splits": {split: report_split(split) for split in ("train", "dev")}, "limits": "Small initial staging only. Token costs include observed failed attempts; unfinished-request usage is unknown. ETA is a linear scenario at observed acceptance, not a promised completion time. No held-out examples are accessed."}
    atomic_json(ROOT / "local/production-v5-staging/report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
