"""Train-only paired reasoning-effort measurement; never formal training data.

The agent-authored expected actions are frozen for independent root review before
either teacher receives any probe. Requests contain only production-visible
context/candidates, use identical inputs for high/max, and preserve real audits.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import random
import statistics
import time

from pastewhat_ranker.preprocess import Preprocessor
from tools.project_context import project_context
from data_tools.deployment import placement_issue
from data_tools.generate import LABEL_SYSTEM, PARTITION_PATH, ROOT, validate_labels
from data_tools.rate_limit import AccountCoordinator
from data_tools.teacher import TeacherClient, atomic_json, canonical_bytes, sha256, utc_now


DIRECTORY = ROOT / "local/teacher-effort-probe"
AUTHORING = ROOT / "data_tools/probes/teacher_effort_v1.authoring.json"


def prepare():
    allowed = {family["id"] for family in json.loads(PARTITION_PATH.read_text())["families"]["train"]}
    preprocessor = Preprocessor(ROOT.parent / "laya-mlx/models/laya-multilingual/tokenizer")
    cases = []
    for source in json.loads(AUTHORING.read_text())["cases"]:
        if source["family_id"] not in allowed:
            raise ValueError("Effort probes are restricted to preassigned Train families")
        episode = copy.deepcopy(source)
        expected = episode.pop("expected_label")
        capture = episode.pop("capture")
        episode["context"] = project_context(episode["context"], capture=capture)
        episode = preprocessor.prepare_episode(episode)
        if episode["preprocessing"]["truncated"]:
            raise ValueError(f"Probe must retain all explicit evidence: {episode['id']}")
        issue = placement_issue(episode, expected)
        if issue:
            raise ValueError(f"Probe placement invalid: {episode['id']}: {issue}")
        cases.append({**episode, "expected_label": expected})
    payload = b"".join(canonical_bytes(case) + b"\n" for case in cases)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    review_path = DIRECTORY / "review_input.jsonl"
    if review_path.is_file() and review_path.read_bytes() != payload:
        raise ValueError("Existing probe is immutable; introduce a new version explicitly")
    review_path.write_bytes(payload)
    manifest = {"purpose": "Train-only reasoning-effort engineering probe, excluded from formal data", "cases": len(cases), "review_input_sha256": sha256(payload), "authoring_sha256": sha256(AUTHORING.read_bytes()), "family_partition_sha256": sha256(PARTITION_PATH.read_bytes()), "projection_sha256": sha256((ROOT / "tools/context_projection/provenance.json").read_bytes()), "label_prompt_sha256": sha256(LABEL_SYSTEM.encode()), "families": dict(Counter(case["family_id"] for case in cases)), "human_validated": False}
    atomic_json(DIRECTORY / "manifest.json", manifest)
    return cases, manifest


def same_action(actual, expected):
    if actual["decision"] != expected["decision"] or set(actual["acceptable_ids"]) != set(expected["acceptable_ids"]):
        return False
    a, b = actual["abstain_reason"], expected["abstain_reason"]
    return a == b or {a, b} <= {"ambiguous", "insufficient_context"}


def run(*, wait):
    cases, manifest = prepare()
    root_review_path = DIRECTORY / "root-review.json"
    coordinator = AccountCoordinator()
    while True:
        account = coordinator.status()
        root_review = json.loads(root_review_path.read_text()) if root_review_path.is_file() else {}
        approved = root_review.get("approved") is True and root_review.get("review_input_sha256") == manifest["review_input_sha256"]
        if approved and not account["paused"]:
            break
        if not wait:
            raise SystemExit("Waiting for matching independent Train probe review and normal quota availability")
        atomic_json(DIRECTORY / "status.json", {"updated_at": utc_now(), "state": "waiting", "review_approved": approved, "account_rate_state": account})
        time.sleep(30)
    client = TeacherClient(ROOT / "local/teacher/train/effort-probe")
    batches = [cases[start:start + 5] for start in range(0, len(cases), 5)]
    tasks = []
    for index, batch in enumerate(batches):
        # Same shuffled input in each pair; condition order alternates by batch.
        visible = []
        for position, case in enumerate(batch):
            entries = copy.deepcopy(case["entries"])
            random.Random(1403 + index * 31 + position).shuffle(entries)
            visible.append({"id": f"p{position + 1}", "context": case["context"], "entries": entries})
        user = json.dumps({"episodes": visible}, ensure_ascii=False)
        for effort in (("high", "max") if index % 2 == 0 else ("max", "high")):
            tasks.append((index, effort, batch, visible, user))

    def request(task):
        index, effort, batch, visible, user = task
        result = client.complete_json(LABEL_SYSTEM, user, max_tokens=8192, reasoning_effort=effort, phase="train-effort-probe", request_id=f"probe-v1-{index}-{effort}")
        labels = validate_labels(result.parsed, visible, require_quoted=True)
        audit = json.loads((client.audit_dir / f"{result.audit_id}.json").read_text())
        rows = [{"id": case["id"], "family_id": case["family_id"], "actual": labels[f"p{position + 1}"]["label"], "expected": case["expected_label"], "matches_reviewed_action": same_action(labels[f"p{position + 1}"]["label"], case["expected_label"])} for position, case in enumerate(batch)]
        record = {"batch": index, "reasoning_effort": effort, "audit_id": result.audit_id, "request_sha256": audit["request_sha256"], "response_sha256": result.response_sha256, "model": result.model, "usage": result.usage, "elapsed_seconds": audit["elapsed_seconds"], "rows": rows, "cache_hit": result.cache_hit}
        atomic_json(DIRECTORY / f"batch-{index}-{effort}.json", record)
        return record

    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(request, tasks))
    by_effort = {}
    for effort in ("high", "max"):
        selected = [record for record in records if record["reasoning_effort"] == effort]
        rows = [row for record in selected for row in record["rows"]]
        usage = Counter()
        for record in selected:
            usage.update({key: value for key, value in record["usage"].items() if isinstance(value, (int, float))})
        by_effort[effort] = {"cases": len(rows), "correct_reviewed_actions": sum(row["matches_reviewed_action"] for row in rows), "requests": len(selected), "total_elapsed_seconds": sum(record["elapsed_seconds"] for record in selected), "median_request_seconds": statistics.median(record["elapsed_seconds"] for record in selected), "usage": dict(usage), "rows": rows}
    high = {row["id"]: row["actual"] for row in by_effort["high"]["rows"]}
    maximum = {row["id"]: row["actual"] for row in by_effort["max"]["rows"]}
    agree = sum(same_action(high[key], maximum[key]) for key in high)
    recommended = "high" if by_effort["high"]["correct_reviewed_actions"] == len(cases) and agree == len(cases) else "max" if by_effort["max"]["correct_reviewed_actions"] == len(cases) else None
    report = {"created_at": utc_now(), "manifest": manifest, "independent_review_sha256": sha256(root_review_path.read_bytes()), "teacher": "kimi-for-coding", "teacher_is_rolling": True, "by_effort": by_effort, "paired_action_agreement": agree, "cases": len(cases), "recommended_effort_for_this_probe": recommended, "production_ready": False, "claim_boundary": "Small, agent-reviewed Train-only engineering probe. It does not establish equal generalization quality, held-out accuracy, or human validation.", "audits": [{key: record[key] for key in ("batch", "reasoning_effort", "audit_id", "request_sha256", "response_sha256", "model")} for record in records]}
    atomic_json(DIRECTORY / "report.json", report)
    summary = {key: value for key, value in report.items() if key != "by_effort"}
    summary["by_effort"] = {effort: {key: value for key, value in result.items() if key != "rows"} for effort, result in by_effort.items()}
    atomic_json(ROOT / "data/train_teacher_effort.report.json", summary)
    atomic_json(DIRECTORY / "status.json", {"updated_at": utc_now(), "state": "measured", "report_sha256": sha256((DIRECTORY / "report.json").read_bytes()), "recommended_effort": recommended})
    print(json.dumps({"cases": len(cases), "paired_agreement": agree, "correct": {effort: result["correct_reviewed_actions"] for effort, result in by_effort.items()}, "recommended_effort": recommended, "report": str(DIRECTORY / "report.json")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.run:
        run(wait=args.wait)
    else:
        _, manifest = prepare()
        print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
