"""Shared JSON-only recommendation policy for evaluation and the AppKit worker.

This module has no NumPy, scikit-learn, PyTorch, or MLX dependency. Calibration
fitting belongs to the independent evaluator; deployment only reads frozen
coefficients bound to the actual FP16 weights and preprocessing manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

FEATURE_NAMES = ["top_minus_abstain", "top_minus_second", "candidate_count", "missing_context"]
VERSION = "pastewhat-calibrator-v2"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_calibrator(path: str | Path, *, weights_path: str | Path,
                    preprocess_path: str | Path, require_accepted: bool = False) -> dict:
    """Load only a schema-valid policy matched to the deployed artifact bytes."""
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "pastewhat-recommendation-calibration" or value.get("version") != VERSION:
        raise ValueError("Unsupported PasteWhat calibration schema")
    if value.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Calibration feature order does not match this policy")
    if value.get("weightsSHA") != file_sha256(weights_path):
        raise ValueError("Calibration is not bound to these deployment weights")
    if value.get("preprocessSHA") != file_sha256(preprocess_path):
        raise ValueError("Calibration is not bound to this preprocessing manifest")
    if require_accepted and value.get("status") != "observed_precision_target_met":
        raise ValueError("Calibration did not meet the preregistered observed precision target")
    if value.get("candidate_must_beat_abstain") is not True or value.get("candidate_abstain_ties") != "abstain":
        raise ValueError("Unsupported abstention decision rule")
    arrays = [value["standardizer"]["mean"], value["standardizer"]["scale"], value["coefficients"]]
    if any(not isinstance(array, list) or len(array) != 4 for array in arrays):
        raise ValueError("Calibration vectors must have four features")
    numeric = [*arrays[0], *arrays[1], *arrays[2], value["intercept"], value["threshold"]]
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item) for item in numeric):
        raise ValueError("Calibration parameters must be finite numbers")
    if any(item <= 0 for item in arrays[1]) or not 0 <= value["threshold"] <= 1:
        raise ValueError("Invalid calibration scale or threshold")
    return value


def has_semantic_context(context: dict) -> bool:
    """Inspect actual text inside a valid native focus envelope, not its keys.

    Plain text and malformed/truncated envelopes retain their literal meaning.
    The native all-empty capture already renders an empty string; this also
    handles valid external protocol inputs and captures containing whitespace.
    """
    if any(isinstance(context.get(key), str) and context[key].strip() for key in ("fieldLabel", "selectedText")):
        return True
    surrounding = context.get("surroundingText", "")
    if not isinstance(surrounding, str) or not surrounding.strip():
        return False
    try:
        focus = json.loads(surrounding)
    except (ValueError, TypeError):
        return True
    if not isinstance(focus, dict) or focus.get("format") != "pastewhat-focus-v1":
        return True
    if type(focus.get("selectionKnown")) is not bool:
        return True
    text_keys = ("beforeSelection", "afterSelection") if focus["selectionKnown"] else ("textWindow",)
    if (set(focus) != {"format", "selectionKnown", "nearbyText", *text_keys}
            or any(not isinstance(focus.get(key), str) for key in text_keys)
            or not isinstance(focus.get("nearbyText"), list)
            or any(not isinstance(value, str) for value in focus["nearbyText"])):
        return True
    return any(value.strip() for value in [*(focus[key] for key in text_keys), *focus["nearbyText"]])


def score_features(episode: dict, prediction: dict) -> tuple[list[float], str, bool]:
    """Features, raw top candidate ID, and whether it strictly beats abstention."""
    if prediction.get("error"):
        raise ValueError("inference failed")
    entries = episode["entries"]
    if not entries or episode["context"].get("isSecure"):
        raise ValueError("bypass has no score features")
    values = prediction.get("candidateScores", [])
    score_map = {item["id"]: float(item["score"]) for item in values}
    if len(score_map) != len(values) or set(score_map) != {entry["id"] for entry in entries}:
        raise ValueError("candidate scores must exactly cover all input candidates")
    abstain = float(prediction["abstainScore"])
    if not all(math.isfinite(value) for value in [*score_map.values(), abstain]):
        raise ValueError("nonfinite inference score")
    ordered = sorted(entries, key=lambda entry: score_map[entry["id"]], reverse=True)
    top_id = ordered[0]["id"]
    top = score_map[top_id]
    second = score_map[ordered[1]["id"]] if len(ordered) > 1 else abstain
    missing = not has_semantic_context(episode["context"])
    return [top - abstain, top - second, float(len(entries)), float(missing)], top_id, top > abstain


def calibrated_probability(features: list[float], calibrator: dict) -> float:
    value = float(calibrator["intercept"])
    value += sum(((x - mean) / scale) * coefficient for x, mean, scale, coefficient in zip(
        features, calibrator["standardizer"]["mean"], calibrator["standardizer"]["scale"], calibrator["coefficients"], strict=True))
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def apply_calibration(episode: dict, prediction: dict, calibrator: dict) -> dict:
    """Map raw scores to an original candidate ID or null with a frozen policy."""
    if episode["context"].get("isSecure") or not episode["entries"]:
        return {**prediction, "recommendedID": None, "decision": "bypass", "confidence": None}
    if prediction.get("error"):
        return {**prediction, "recommendedID": None, "decision": "inference_failure", "confidence": None}
    try:
        features, top_id, beats_abstain = score_features(episode, prediction)
        probability = calibrated_probability(features, calibrator)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return {**prediction, "recommendedID": None, "decision": "inference_failure", "confidence": None,
                "error": "invalid_score_or_calibration:" + type(error).__name__}
    recommended = top_id if beats_abstain and probability >= calibrator["threshold"] else None
    return {**prediction, "recommendedID": recommended, "rawTopID": top_id,
            "confidence": probability, "decision": "recommended" if recommended else "abstain"}


predict = apply_calibration
