"""Paired final acceptance on the frozen test, without selecting any artifact."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from evaluations.common import (
    calibrated_decision, load_jsonl, paired_comparison, row_outcome, sha256,
    summarize, summarize_outcomes, write_json, write_jsonl,
)
from evaluations.freeze import verify_freeze
from pastewhat_ranker.calibration import load_calibrator


def slices(episodes: list[dict], predictions: list[dict]) -> dict:
    mapping = {row["id"]: row for row in predictions}
    groups = defaultdict(list)
    for episode in episodes:
        outcome = row_outcome(episode, mapping[episode["id"]])
        groups["label/" + (episode["label"]["abstain_reason"] or "select")].append(outcome)
        count = len(episode["entries"])
        bucket = "1" if count == 1 else "2-4" if count <= 4 else "5-10" if count <= 10 else "11-20"
        groups["candidate_count/" + bucket].append(outcome)
        metadata = episode.get("synthetic_metadata", {})
        groups["language/" + metadata.get("language", "unspecified")].append(outcome)
        if len(episode["label"]["acceptable_ids"]) > 1:
            groups["multiple_acceptable_candidates"].append(outcome)
        positives = set(episode["label"]["acceptable_ids"])
        kinds = {entry["kind"] for entry in episode["entries"] if entry["id"] in positives}
        if any(entry["kind"] in kinds and entry["id"] not in positives for entry in episode["entries"]):
            groups["same_kind_negative_present"].append(outcome)
        if metadata.get("generator_spec", {}).get("field_overrides_app_category"):
            groups["generator_requested_field_category_conflict"].append(outcome)
        if episode.get("preprocessing", {}).get("truncated"):
            groups["truncated_before_teacher_label"].append(outcome)
    return {key: summarize_outcomes(value) for key, value in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ranker-scores", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--preprocess", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite frozen Test results")
    frozen = verify_freeze(args.freeze, args.data)
    if sha256(args.calibrator) != frozen["inputs"]["calibrator"]["sha256"]:
        raise SystemExit("Report calibrator differs from final freeze")
    episodes = load_jsonl(args.data)
    if any(row.get("split") != "test" for row in episodes):
        raise SystemExit("Final acceptance only uses final Test")
    baseline, scores = load_jsonl(args.baseline), load_jsonl(args.ranker_scores)
    mapping = {row["id"]: row for row in scores}
    if len(mapping) != len(scores) or set(mapping) != {row["id"] for row in episodes}:
        raise SystemExit("Ranker scores do not exactly cover Test")
    calibrator = load_calibrator(args.calibrator, weights_path=args.weights, preprocess_path=args.preprocess)
    decisions = [calibrated_decision(row, mapping[row["id"]], calibrator) for row in episodes]
    baseline_summary, ranker_summary = summarize(episodes, baseline), summarize(episodes, decisions)
    paired = paired_comparison(episodes, baseline, decisions)
    groups = {}
    for group, current in ranker_summary["by_group"].items():
        previous = baseline_summary["by_group"][group]
        delta = None if current["answerable_top1"] is None or previous["answerable_top1"] is None else current["answerable_top1"] - previous["answerable_top1"]
        eligible = current["answerable"] >= 30
        groups[group] = {"answerable": current["answerable"], "top1_delta": delta,
                         "sufficient_for_gate": eligible,
                         "material_regression": eligible and delta is not None and delta < -0.05}
    criteria = {
        "answerable_top1_improves_five_points": paired["answerable_top1_delta"] is not None and paired["answerable_top1_delta"] >= 0.05,
        "no_material_key_group_regression": not any(row["material_regression"] for row in groups.values()),
        "calibration_observed_precision_target_met": calibrator["status"] == "observed_precision_target_met",
        "no_inference_failures": ranker_summary["overall"]["failure"] == 0 and baseline_summary["overall"]["failure"] == 0,
    }
    metrics = {
        "scope": "independently held-out synthetic conceptual families; no real-user or human-validation claim",
        "baseline": baseline_summary, "ranker": ranker_summary, "paired": paired,
        "slices": {"baseline": slices(episodes, baseline), "ranker": slices(episodes, decisions)},
        "key_group_gates": groups, "acceptance": {"passed": all(criteria.values()), "criteria": criteria},
        "provenance": {"freeze_sha256": sha256(args.freeze), "data_sha256": sha256(args.data),
                       "baseline_scores_sha256": sha256(args.baseline), "ranker_scores_sha256": sha256(args.ranker_scores),
                       "calibrator_sha256": sha256(args.calibrator), "report_code_sha256": sha256(__file__)},
    }
    write_json(args.output / "metrics.json", metrics)
    write_jsonl(args.output / "ranker-decisions.jsonl", decisions)
    verify_freeze(args.freeze, args.data)
    print(json.dumps({"acceptance": metrics["acceptance"], "baseline": baseline_summary["overall"],
                      "ranker": ranker_summary["overall"], "paired": paired}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
