"""Bounded Train-only low/high comparison; six calls, no authoring or retries."""
from __future__ import annotations

import json
from pathlib import Path
import statistics

from data_tools.teacher import TeacherClient, TeacherError, atomic_json, canonical_bytes, sha256, utc_now
from data_tools.v7 import LABEL_SYSTEM, ROOT, remap_labels, same_action, validate_labels, visible_batch
from run_contract import load_run_plan


def main():
    plan = load_run_plan(ROOT / "configs/run_plan_efficient.json")
    base = ROOT / "local/v7" / plan.run_id
    directory = base / "effort-probe"
    directory.mkdir(parents=True, exist_ok=True)
    engineering_path = ROOT / "local/teacher-effort-probe/review_input.jsonl"
    engineering = [json.loads(line) for line in engineering_path.read_bytes().splitlines()]
    assert len(engineering) == 20
    allowed = {row["id"] for row in json.loads((ROOT / "data_tools/family_partition.json").read_text())["families"]["train"]}
    frozen_path = directory / "inputs.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
    else:
        pool = [json.loads(line) for line in (base / "train.jsonl").read_bytes().splitlines()]
        supplemental = sorted(pool, key=lambda row: (-len(row["entries"]), row["id"]))[:10]
        frozen = {"engineering": engineering, "supplemental": supplemental, "engineering_source_sha256": sha256(engineering_path.read_bytes()), "supplemental_reference_labels_are_not_ground_truth": True}
        atomic_json(frozen_path, frozen)
    if any(row["family_id"] not in allowed for rows in (frozen["engineering"], frozen["supplemental"]) for row in rows):
        raise ValueError("Provider tuning must use Train families only")
    groups = [frozen["engineering"][:10], frozen["engineering"][10:], frozen["supplemental"]]
    client = TeacherClient(directory / "teacher", max_attempts=1)
    records = []
    for index, group in enumerate(groups):
        visible, mapping = visible_batch(group, 47913 + index, "effort-probe")
        for effort in (("low", "high") if index % 2 == 0 else ("high", "low")):
            record = {"group": index, "reasoning_effort": effort, "cases": len(group), "reference_kind": "root-reviewed engineering labels" if index < 2 else "agreement only; no independent ground truth"}
            try:
                result = client.complete_json(LABEL_SYSTEM, json.dumps({"episodes": visible}, ensure_ascii=False), max_tokens=12288, reasoning_effort=effort, response_format="json_object", phase="v7-bounded-low-high-probe", request_id=f"group-{index}-{effort}")
                labels = remap_labels(validate_labels(result.parsed, visible), mapping)
                audit = json.loads((client.audit_dir / (result.audit_id + ".json")).read_text())
                record.update(audit_id=result.audit_id, usage=result.usage, elapsed_seconds=audit["elapsed_seconds"], model=result.model, labels=labels, cache_hit=result.cache_hit)
                if index < 2:
                    record["reference_matches"] = sum(same_action(row["expected_label"], labels[row["id"]]) for row in group)
            except (TeacherError, ValueError, KeyError, TypeError) as error:
                record["error"] = str(error)
            records.append(record)
            atomic_json(directory / "progress.json", {"records": records, "http_request_ceiling": 6, "automatic_retries": 0, "updated_at": utc_now()})
    paired = []
    for index, group in enumerate(groups):
        pair = {row["reasoning_effort"]: row for row in records if row["group"] == index}
        if all("labels" in pair[effort] for effort in ("low", "high")):
            paired.append({"group": index, "cases": len(group), "action_set_agreements": sum(same_action(pair["low"]["labels"][row["id"]], pair["high"]["labels"][row["id"]]) for row in group), "reference_kind": pair["low"]["reference_kind"]})
    summary = {}
    for effort in ("low", "high"):
        rows = [row for row in records if row["reasoning_effort"] == effort]
        success = [row for row in rows if "usage" in row]
        summary[effort] = {"requests_planned": len(rows), "valid_responses": len(success), "engineering_reference_matches": sum(row.get("reference_matches", 0) for row in rows), "engineering_cases": 20, "known_tokens": sum(row["usage"].get("total_tokens", 0) for row in success), "median_request_seconds": statistics.median(row["elapsed_seconds"] for row in success) if success else None}
    report = {**plan.binding(), "created_at": utc_now(), "input_sha256": sha256(frozen_path.read_bytes()), "label_system_sha256": sha256(LABEL_SYSTEM.encode()), "http_request_ceiling": 6, "automatic_retries": 0, "by_effort": summary, "paired_agreement": paired, "records": records, "production_effort_changed": False, "limitations": "Only20 engineering cases have an independently reviewed reference. Supplemental10 supports agreement and latency observations only. No heldout data or model scores were accessed."}
    atomic_json(directory / "report.json", report)
    print(json.dumps({"by_effort": summary, "paired_agreement": paired}), flush=True)


if __name__ == "__main__":
    main()
