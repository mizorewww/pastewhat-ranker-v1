"""Export credential-free Train/Dev teacher request provenance and progress."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from data_tools.teacher import atomic_json, sha256, utc_now


ROOT = Path(__file__).resolve().parents[1]


def summarize(split, *, write=False):
    manifest_path = ROOT / "data" / f"{split}.manifest.json"
    dataset = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    references = set()
    data_path = ROOT / "data" / f"{split}.jsonl"
    if data_path.is_file():
        for line in data_path.read_text().splitlines():
            references.update(value for key, value in json.loads(line).get("provenance", {}).items() if key.endswith("audit_id"))
    requests, phases, statuses, tokens = [], Counter(), Counter(), Counter()
    for path in sorted((ROOT / "local" / "teacher" / split).rglob("*.json")):
        audit = json.loads(path.read_text())
        response = audit.get("response", {})
        usage = response.get("usage", {})
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            tokens[key] += usage.get(key, 0)
        phases[audit.get("phase", "unknown")] += 1
        statuses[audit.get("status", "unknown")] += 1
        requests.append({
            "audit_id": audit.get("audit_id"), "phase": audit.get("phase"), "request_id": audit.get("request_id"),
            "status": audit.get("status"), "started_at": audit.get("started_at"), "completed_at": audit.get("completed_at"),
            "endpoint": audit.get("endpoint"), "request_sha256": audit.get("request_sha256"), "response_sha256": audit.get("response_sha256"),
            "requested_model": audit.get("request", {}).get("model"), "response_model": response.get("model"),
            "temperature": audit.get("request", {}).get("temperature"), "thinking": audit.get("request", {}).get("thinking", {"type": "provider-default"}),
            "usage": usage, "elapsed_seconds": audit.get("elapsed_seconds"), "attempts": len(audit.get("attempts", [])) + (audit.get("status") == "success"),
            "referenced_by_current_accepted_data": audit.get("audit_id") in references,
        })
    result = {"split": split, "updated_at": utc_now(), "accepted_episodes": dataset.get("episodes", 0), "full_target": dataset.get("planned_full_split", 20000 if split == "train" else 1000), "dataset_sha256": dataset.get("sha256"), "semantic_or_consensus_rejections_in_completed_batches": dataset.get("semantic_rejections", 0), "request_count_including_unreleased_attempts": len(requests), "request_statuses": dict(statuses), "request_phases": dict(phases), "usage_including_unreleased_attempts": dict(tokens), "rolling_teacher_cannot_be_reexecuted_as_a_pinned_revision": True, "raw_audits_location": f"local/teacher/{split}/ (ignored; no credentials)", "requests": requests}
    if write:
        atomic_json(ROOT / "data" / f"{split}.teacher_manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = summarize(args.split, write=args.write)
    print(json.dumps({key: value for key, value in result.items() if key != "requests"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
