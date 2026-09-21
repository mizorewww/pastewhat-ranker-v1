"""Fit recommendation correctness only on the frozen Calibration split."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from evaluations.common import (
    FEATURE_NAMES, calibrated_decision, calibrated_probability, load_jsonl,
    score_features, sha256, summarize, validate_label, verify_score_run, wilson_interval, write_json,
)
from evaluations.score import command_artifacts
from run_contract import load_run_plan


def partition_families(episodes: list[dict]) -> dict:
    families = sorted({row["family_id"] for row in episodes},
                      key=lambda value: hashlib.sha256(("pastewhat-calibration-v1:" + value).encode()).hexdigest())
    if len(families) < 2 or len(families) % 2:
        raise ValueError("Calibration requires an even number of complete conceptual families")
    midpoint = len(families) // 2
    return {"fit": families[:midpoint], "threshold": families[midpoint:]}


def fit_calibrator(episodes: list[dict], predictions: list[dict], *, minimum_recommendations: int = 25) -> tuple[dict, dict]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    mapping = {row["id"]: row for row in predictions}
    if len(mapping) != len(predictions) or set(mapping) != {row["id"] for row in episodes}:
        raise ValueError("Calibration predictions must cover every episode exactly once")
    partition = partition_families(episodes)
    prepared = []
    excluded = Counter()
    for episode in episodes:
        validate_label(episode)
        if episode["context"].get("isSecure") or not episode["entries"]:
            excluded["bypass"] += 1
            continue
        try:
            features, top_id, qualifies = score_features(episode, mapping[episode["id"]])
        except (KeyError, ValueError, TypeError):
            excluded["invalid_score_or_inference_failure"] += 1
            continue
        target = int(episode["label"]["decision"] == "select" and top_id in episode["label"]["acceptable_ids"])
        prepared.append({"episode": episode, "features": features, "top_id": top_id, "qualifies": qualifies,
                         "target": target, "role": "fit" if episode["family_id"] in partition["fit"] else "threshold"})
    fit = [row for row in prepared if row["role"] == "fit"]
    threshold = [row for row in prepared if row["role"] == "threshold"]
    if {row["target"] for row in fit} != {0, 1}:
        raise ValueError("Calibration fit targets must contain successes and failures; no constant fallback")
    scaler = StandardScaler().fit(np.array([row["features"] for row in fit], dtype=np.float64))
    classifier = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    classifier.fit(scaler.transform([row["features"] for row in fit]), [row["target"] for row in fit])
    calibrator = {
        "schema": "pastewhat-recommendation-calibration",
        "version": "pastewhat-calibrator-v1", "feature_names": FEATURE_NAMES,
        "standardizer": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "coefficients": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0]),
        "threshold": 1.0, "target_precision": 0.95,
        "minimum_threshold_recommendations": minimum_recommendations,
        "candidate_must_beat_abstain": True, "candidate_abstain_ties": "abstain",
        "missing_context_definition": "fieldLabel, selectedText and surroundingText are all blank",
        "fit_class_counts": dict(Counter(str(row["target"]) for row in fit)),
        "family_partition": partition,
    }
    # The same JSON-only inference function is used here and after export.
    for row in prepared:
        row["probability"] = calibrated_probability(row["features"], calibrator)
    sklearn_prob = classifier.predict_proba(scaler.transform([row["features"] for row in prepared]))[:, 1]
    if not np.allclose(sklearn_prob, [row["probability"] for row in prepared], atol=1e-12, rtol=0):
        raise AssertionError("Exported logistic coefficients do not reproduce sklearn probabilities")
    thresholds = sorted({row["probability"] for row in threshold if row["qualifies"]}, reverse=True)
    denominator = sum(row["family_id"] in partition["threshold"] for row in episodes)
    curve = []
    for value in thresholds:
        selected = [row for row in threshold if row["qualifies"] and row["probability"] >= value]
        correct = sum(row["target"] for row in selected)
        curve.append({"threshold": value, "recommended": len(selected), "correct": correct,
                      "precision": correct / len(selected), "coverage": len(selected) / denominator,
                      "precision_wilson95": wilson_interval(correct, len(selected))})
    eligible = [row for row in curve if row["recommended"] >= minimum_recommendations]
    accepted = [row for row in eligible if row["precision"] >= 0.95]
    if accepted:
        selected = max(accepted, key=lambda row: (row["coverage"], row["precision"], row["threshold"]))
        calibrator["status"] = "observed_precision_target_met"
    elif eligible:
        selected = max(eligible, key=lambda row: (row["precision"], row["coverage"], row["threshold"]))
        calibrator["status"] = "precision_target_not_met_diagnostic_only"
    elif curve:
        selected = max(curve, key=lambda row: (row["recommended"], row["precision"]))
        calibrator["status"] = "insufficient_recommendation_coverage_diagnostic_only"
    else:
        raise ValueError("No calibration examples beat the abstain score; an all-null release is not accepted")
    calibrator["threshold"] = selected["threshold"]
    calibrator["threshold_selection"] = selected
    outcomes = [calibrated_decision(episode, mapping[episode["id"]], calibrator) for episode in episodes]
    held = [episode for episode in episodes if episode["family_id"] in partition["threshold"]]
    fit_episodes = [episode for episode in episodes if episode["family_id"] in partition["fit"]]
    outcome_map = {row["id"]: row for row in outcomes}
    report = {
        "family_partition": partition, "excluded_from_logistic_fit": dict(excluded),
        "fit_examples": len(fit), "threshold_examples": len(threshold),
        "status": calibrator["status"], "precision_coverage_curve": curve,
        "fit_diagnostic": summarize(fit_episodes, [outcome_map[row["id"]] for row in fit_episodes]),
        "threshold_selection": summarize(held, [outcome_map[row["id"]] for row in held]),
        "release_passes_calibration": calibrator["status"] == "observed_precision_target_met",
        "scope": "synthetic calibration distribution; observed precision is not a population guarantee",
    }
    return calibrator, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="The actual final deployment precision weights")
    parser.add_argument("--preprocess", type=Path, required=True, help="The production preprocessing manifest")
    parser.add_argument("--partition", type=Path, default=Path(__file__).with_name("calibration-family-partition.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a calibration run")
    score_run = verify_score_run(args.scores, dataset=args.data, split="calibration", protocol="ranker", run_plan=plan)
    if (score_run.get("artifacts", {}).get("weights_sha256") != sha256(args.weights) or
            score_run.get("artifacts", {}).get("preprocess_sha256") != sha256(args.preprocess)):
        raise SystemExit("Calibration score run differs from the requested deployment weights or preprocessing")
    if command_artifacts(score_run["command"], "ranker", None) != score_run["artifacts"]:
        raise SystemExit("Deployment model files or runtime changed after Calibration scoring")
    episodes, predictions = load_jsonl(args.data), load_jsonl(args.scores)
    fixed_partition = json.loads(args.partition.read_text())
    if partition_families(episodes) != fixed_partition["families"]:
        raise SystemExit("Calibration families differ from their pre-scoring allocation")
    if any(row.get("split") != "calibration" for row in episodes):
        raise SystemExit("Only Calibration may fit the final calibrator")
    calibrator, report = fit_calibrator(episodes, predictions,
        minimum_recommendations=plan.document["quality_gates"]["minimum_calibration_recommendations"])
    plan.verify_unchanged()
    calibrator["weightsSHA"] = sha256(args.weights)
    calibrator["preprocessSHA"] = sha256(args.preprocess)
    provenance = {"dataset_sha256": sha256(args.data), "scores_sha256": sha256(args.scores),
                  "calibration_partition_sha256": sha256(args.partition),
                  "deployment_manifest_sha256": sha256(args.deployment_manifest), **plan.binding()}
    calibrator.update(plan.binding())
    report.update(plan.binding())
    calibrator["provenance"] = provenance
    report["provenance"] = provenance
    write_json(args.output / "calibrator.json", calibrator)
    write_json(args.output / "calibration-report.json", report)
    print({"status": calibrator["status"], "fit": report["fit_examples"], "threshold": report["threshold_examples"],
           "output": str(args.output), "calibrator_sha256": sha256(args.output / "calibrator.json")})


if __name__ == "__main__":
    main()
