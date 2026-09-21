"""Stable evaluation contracts; labels never enter inference requests."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_contract import RunPlan, action_quotas, family_quotas

from pastewhat_ranker.calibration import (
    FEATURE_NAMES, apply_calibration as calibrated_decision,
    calibrated_probability, score_features,
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    seen = set()
    with Path(path).open() as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{lineno}: expected an object with a string id")
            if row["id"] in seen:
                raise ValueError(f"{path}:{lineno}: duplicate episode id")
            seen.add(row["id"])
            rows.append(row)
    return rows


def require_plan_data_path(plan: RunPlan, split: str, path: Path) -> None:
    if path.resolve() != plan.data_path(split).resolve():
        raise ValueError("Formal heldout input path differs from its registered run")


def validate_formal_heldout_allocation(episodes: list[dict], split: str, partition: dict, plan: RunPlan) -> dict:
    """Only complete preregistered heldout allocations qualify for formal scoring."""
    if split not in {"calibration", "test"}:
        raise ValueError("Formal heldout allocation is only Calibration or Test")
    families = partition["families"][split]
    expected_count = 8 if split == "calibration" else 12
    if len(families) != expected_count:
        raise ValueError("Conceptual family partition differs from preregistered allocation")
    plan.verify_unchanged()
    expected = family_quotas(partition, split, plan.target(split))
    if any(episode.get("split") != split for episode in episodes):
        raise ValueError("Heldout rows contain a different split")
    observed = dict(Counter(episode["family_id"] for episode in episodes))
    if observed != expected or len({episode["id"] for episode in episodes}) != len(episodes):
        raise ValueError("Formal heldout data must contain the complete unique preregistered family allocation")
    buckets = {family: Counter() for family in expected}
    for episode in episodes:
        metadata = episode.get("synthetic_metadata", {})
        if any(metadata.get(key) != value for key, value in plan.binding().items()):
            raise ValueError("Heldout row belongs to a different or unregistered production plan")
        if metadata.get("teacher_contract_version") != plan.document["teacher_contract_version"]:
            raise ValueError("Heldout row uses a different registered teacher contract")
        validate_label(episode)
        label = episode["label"]
        bucket = "select" if label["decision"] == "select" else "no_match" if label["abstain_reason"] == "no_match" else "missing_intent"
        buckets[episode["family_id"]][bucket] += 1
    observed_actions = {family: {name: counts[name] for name in ("select", "no_match", "missing_intent")}
                        for family, counts in buckets.items()}
    if observed_actions != action_quotas(expected):
        raise ValueError("Formal heldout labels differ from registered global 70/20/10 action quotas")
    return {**plan.binding(), "split": split, "episodes": len(episodes), "families": observed,
            "action_quotas": observed_actions}


def verify_score_run(path: Path, *, dataset: Path, split: str, protocol: str, freeze: Path | None = None,
                     run_plan: RunPlan | None = None) -> dict:
    """Reject stale, partial, differently scoped, or unbound scoring artifacts."""
    manifest_path = path.parent / "run-manifest.json"
    completion_path = path.parent / "completion.json"
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads(completion_path.read_text())
    if split in {"calibration", "test"}:
        if run_plan is None:
            raise ValueError("Formal score verification requires its registered run plan")
        run_plan.verify_unchanged()
        require_plan_data_path(run_plan, split, dataset)
        if any(manifest.get(key) != value or completion.get(key) != value for key, value in run_plan.binding().items()):
            raise ValueError("Scoring run belongs to a different registered production plan")
    if (manifest.get("split") != split or manifest.get("protocol") != protocol or
            manifest.get("data_sha256") != sha256(dataset)):
        raise ValueError("Scoring run belongs to a different dataset, split, or model protocol")
    if manifest.get("freeze_sha256") != (sha256(freeze) if freeze else None):
        raise ValueError("Scoring run does not refer to the required artifact freeze")
    if completion.get("scores_sha256") != sha256(path):
        raise ValueError("Scores changed after the scoring run completed")
    if completion.get("episodes") != manifest.get("episodes"):
        raise ValueError("Scoring completion count differs from the run manifest")
    return manifest


def write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: list[dict], *, overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def inference_request(episode: dict) -> dict:
    """Explicit input allowlist excludes IDs' meanings, labels and explanations."""
    context_names = {
        "applicationCategory", "inputSurface", "fieldRole", "fieldLabel",
        "selectedText", "surroundingText", "hasAccessibility", "isSecure",
    }
    entry_names = {"id", "text", "kind", "capabilities", "sourceCategory"}
    return {
        "id": episode["id"],
        "context": {key: value for key, value in episode["context"].items() if key in context_names},
        "entries": [
            {key: value for key, value in entry.items() if key in entry_names}
            for entry in episode["entries"]
        ],
    }


def validate_label(episode: dict) -> None:
    label = episode["label"]
    acceptable = label["acceptable_ids"]
    ids = [entry["id"] for entry in episode["entries"]]
    if len(ids) != len(set(ids)) or not 0 <= len(ids) <= 20:
        raise ValueError("invalid candidate ids/count")
    if not isinstance(acceptable, list) or len(acceptable) != len(set(acceptable)):
        raise ValueError("acceptable_ids must be a unique list")
    if any(item not in ids for item in acceptable):
        raise ValueError("acceptable id absent from candidate set")
    if label["decision"] == "select":
        if not acceptable or label.get("abstain_reason") is not None:
            raise ValueError("select must have positives and no abstain reason")
        if episode["context"].get("isSecure"):
            raise ValueError("secure inputs must bypass recommendations")
    elif label["decision"] == "abstain":
        if acceptable or label.get("abstain_reason") not in {"no_match", "insufficient_context", "ambiguous"}:
            raise ValueError("abstain must have no positives and a recognized reason")
    else:
        raise ValueError("invalid decision label")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float] | None:
    if not count:
        return None
    probability = successes / count
    denominator = 1 + z * z / count
    center = (probability + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    left = int(index)
    right = min(len(values) - 1, left + 1)
    return values[left] + (values[right] - values[left]) * (index - left)


def row_outcome(episode: dict, prediction: dict) -> dict:
    answerable = episode["label"]["decision"] == "select"
    ids = set(episode["label"]["acceptable_ids"])
    failed = bool(prediction.get("error")) or prediction.get("decision") in {
        "inference_failure", "invalid_request", "remote_unavailable", "model_unavailable",
        "timeout", "worker_timeout", "error",
    }
    recommended = prediction.get("recommendedID")
    if recommended is not None and recommended not in {entry["id"] for entry in episode["entries"]}:
        failed = True
    promoted = recommended is not None and not failed
    correct = promoted and answerable and recommended in ids
    decision_correct = not failed and (correct or (not answerable and not promoted))
    raw_top = prediction.get("rawTopID")
    if raw_top is None and not failed:
        try:
            _, raw_top, _ = score_features(episode, prediction)
        except (KeyError, TypeError, ValueError):
            pass
    return {
        "id": episode["id"], "family_id": episode["family_id"],
        "group": episode.get("group", episode.get("domain", "unspecified")),
        "answerable": answerable, "failure": failed,
        "promoted": promoted, "correct_promotion": correct,
        "correct_decision": decision_correct, "raw_top_correct": raw_top in ids if answerable and not failed else False,
        "raw_top_available": raw_top is not None,
        "false_promotion": not answerable and promoted,
        "latencyMS": prediction.get("latencyMS"),
        "candidate_count": len(episode["entries"]),
    }


def summarize_outcomes(rows: list[dict]) -> dict:
    total = len(rows)
    count = Counter()
    for row in rows:
        for key in ("answerable", "failure", "promoted", "correct_promotion", "correct_decision", "raw_top_correct", "raw_top_available", "false_promotion"):
            count[key] += int(row[key])
    no_answer = total - count["answerable"]
    latency = [float(row["latencyMS"]) for row in rows if isinstance(row.get("latencyMS"), (int, float)) and math.isfinite(row["latencyMS"])]
    result = {
        "episodes": total, **dict(count), "unanswerable": no_answer,
        "answerable_top1": ratio(count["correct_promotion"], count["answerable"]),
        "decision_accuracy": ratio(count["correct_decision"], total),
        "recommendation_precision": ratio(count["correct_promotion"], count["promoted"]),
        "recommendation_precision_wilson95": wilson_interval(count["correct_promotion"], count["promoted"]),
        "coverage": ratio(count["promoted"], total),
        "false_promotion_rate": ratio(count["false_promotion"], no_answer),
        "raw_answerable_top1": ratio(count["raw_top_correct"], count["answerable"]) if count["raw_top_available"] else None,
        "latency_ms_p50": percentile(latency, 50), "latency_ms_p95": percentile(latency, 95),
    }
    return result


def summarize(episodes: list[dict], predictions: list[dict]) -> dict:
    mapping = {prediction["id"]: prediction for prediction in predictions}
    if len(mapping) != len(predictions) or set(mapping) != {episode["id"] for episode in episodes}:
        raise ValueError("prediction ids must exactly cover dataset ids")
    rows = [row_outcome(episode, mapping[episode["id"]]) for episode in episodes]
    groups, families = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
        families[row["family_id"]].append(row)
    return {"overall": summarize_outcomes(rows),
            "by_group": {key: summarize_outcomes(value) for key, value in sorted(groups.items())},
            "by_family": {key: summarize_outcomes(value) for key, value in sorted(families.items())}}


def paired_comparison(episodes: list[dict], baseline: list[dict], replacement: list[dict], *, bootstrap_repeats: int = 2000) -> dict:
    maps = [{prediction["id"]: prediction for prediction in items} for items in (baseline, replacement)]
    if any(len(mapping) != len(episodes) or set(mapping) != {episode["id"] for episode in episodes} for mapping in maps):
        raise ValueError("paired predictions must exactly cover the same episodes")
    pairs = [(row_outcome(episode, maps[0][episode["id"]]), row_outcome(episode, maps[1][episode["id"]])) for episode in episodes]
    families = defaultdict(list)
    for left, right in pairs:
        families[left["family_id"]].append((left, right))
    family_ids = sorted(families)
    generator = random.Random(423190)
    bootstraps = []
    for _ in range(bootstrap_repeats):
        sample = [pair for family in generator.choices(family_ids, k=len(family_ids)) for pair in families[family]]
        answerable = [(left, right) for left, right in sample if left["answerable"]]
        if answerable:
            bootstraps.append(sum(int(right["correct_promotion"]) - int(left["correct_promotion"]) for left, right in answerable) / len(answerable))
    answerable = [(left, right) for left, right in pairs if left["answerable"]]
    return {
        "answerable_top1_delta": ratio(sum(int(right["correct_promotion"]) - int(left["correct_promotion"]) for left, right in answerable), len(answerable)),
        "delta_family_bootstrap95": [percentile(bootstraps, 2.5), percentile(bootstraps, 97.5)],
        "bootstrap_unit": "family_id", "bootstrap_repeats": bootstrap_repeats,
        "baseline_wrong_replacement_correct": sum(not left["correct_decision"] and right["correct_decision"] for left, right in pairs),
        "baseline_correct_replacement_wrong": sum(left["correct_decision"] and not right["correct_decision"] for left, right in pairs),
        "answerable_gains": sum(not left["correct_promotion"] and right["correct_promotion"] for left, right in answerable),
        "answerable_regressions": sum(left["correct_promotion"] and not right["correct_promotion"] for left, right in answerable),
    }
