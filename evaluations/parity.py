"""Independent fixed-input PyTorch/MLX export and inference invariants.

This is a numerical regression benchmark, not the held-out accuracy Test.
It has no labels and never selects a checkpoint. Backends run in separate
sequential processes so training/GPU coordination remains explicit.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import subprocess
import sys

from evaluations.common import calibrated_decision, load_jsonl, sha256, write_json, write_jsonl
from pastewhat_ranker.preprocess import Preprocessor

PARITY_VERSION = "pastewhat-export-regression-v1"
TOLERANCES = {"cross_backend_max_abs": 0.05, "cross_backend_mean_abs": 0.01,
              "mlx_padding_or_permutation_max_abs": 0.02,
              "torch_padding_or_permutation_max_abs": 0.001,
              "save_reload_max_abs": 0.0}


def make_regression_inputs() -> list[dict]:
    rows = []
    phrases = [("Paste the note confirming availability.", "Available for the meeting.", "Unavailable for the meeting."),
               ("粘贴确认可以参加的短消息。", "我可以参加这次会议。", "我无法参加这次会议。"),
               ("Pega la nota que confirma la disponibilidad.", "Puedo asistir a la reunión.", "No puedo asistir a la reunión."),
               ("参加できることを伝える短いメッセージを貼り付けてください。", "会議に参加できます。", "会議に参加できません。")]
    for size_index, count in enumerate((1, 5, 10, 20)):
        for length_index, repeats in enumerate((0, 6, 28, 90)):
            query, positive, negative = phrases[(size_index + length_index) % len(phrases)]
            background = " Synthetic numerical verification text with no additional task evidence." * repeats
            context = {"applicationCategory": "writing", "inputSurface": "text", "fieldRole": "AXTextArea",
                       "fieldLabel": "Text", "selectedText": "", "surroundingText": query + background,
                       "hasAccessibility": True, "isSecure": False}
            entries = [{"id": f"candidate-{index}", "text": (positive if index % 3 == 0 else negative) +
                        f" Synthetic note {index}." + background, "kind": "text", "capabilities": ["text"], "sourceCategory": "writing"}
                       for index in range(count)]
            if count > 1:
                entries[-1] = {**entries[0], "id": entries[-1]["id"]}
            rows.append({"id": f"regression-{count}-{length_index}", "context": context, "entries": entries})
    return rows


def score_values(row: dict) -> dict:
    return {**{entry["id"]: float(entry["score"]) for entry in row["candidateScores"]}, "__abstain__": float(row["abstainScore"])}


def equivalent(episode: dict, first: str | None, second: str | None) -> bool:
    if first == second:
        return True
    if first is None or second is None:
        return False
    entries = {row["id"]: {key: value for key, value in row.items() if key != "id"} for row in episode["entries"]}
    return entries.get(first) == entries.get(second) and first in entries and second in entries


def raw_action(row: dict) -> str | None:
    entries = row["candidateScores"]
    first = max(entries, key=lambda value: value["score"])
    return first["id"] if first["score"] > row["abstainScore"] else None


def run_backend(args):
    from pastewhat_ranker.worker import RankerScorer

    episodes = load_jsonl(args.inputs)
    scorer = RankerScorer(args.model, backend=args.backend, device=args.device)
    preprocessor = scorer.preprocessor
    outputs = []
    for episode in episodes:
        regular = scorer.score(episode)
        permuted = copy.deepcopy(episode)
        random.Random(episode["id"] + ":permutation").shuffle(permuted["entries"])
        reordered = scorer.score(permuted)
        encoded = preprocessor.encode_episode(episode)
        longest = max(map(len, encoded["input_ids"]))
        padding = min(64, 1024 - longest)
        if args.backend == "mlx":
            import mlx.core as mx
            from pastewhat_ranker.mlx_model import collate_mlx
            batch = collate_mlx([episode], preprocessor, extra_padding=padding)
            logits = scorer.model(batch["input_ids"], batch["attention_mask"], batch["candidate_counts"])
            mx.eval(logits)
            padded = logits[0].tolist()
        else:
            import torch
            from pastewhat_ranker.model import collate
            batch = collate([episode], preprocessor, args.device, extra_padding=padding)
            with torch.inference_mode():
                logits = scorer.model(batch["input_ids"], batch["attention_mask"], batch["candidate_counts"])
            padded = logits[0].cpu().tolist()
        padded_response = {"candidateScores": [{"id": entry["id"], "score": value} for entry, value in zip(episode["entries"], padded[:-1], strict=True)],
                           "abstainScore": padded[-1]}
        base, shuffled, extra = score_values(regular), score_values(reordered), score_values(padded_response)
        if not all(math.isfinite(value) for value in [*base.values(), *shuffled.values(), *extra.values()]):
            raise ValueError("Nonfinite regression scores")
        outputs.append({"id": episode["id"], "regular": regular, "permuted": reordered, "padded": padded_response,
                        "pair_token_lengths": [len(ids) for ids in encoded["input_ids"]], "extra_padding": padding,
                        "permutation_max_abs": max(abs(base[key] - shuffled[key]) for key in base),
                        "padding_max_abs": max(abs(base[key] - extra[key]) for key in base)})
    group = [episodes[0], episodes[4]]  # One and five valid candidates, same batch.
    if args.backend == "mlx":
        batch = collate_mlx(group, preprocessor)
        mixed = scorer.model(batch["input_ids"], batch["attention_mask"], batch["candidate_counts"])
        mx.eval(mixed)
        mixed_scores = mixed.tolist()
    else:
        batch = collate(group, preprocessor, args.device)
        with torch.inference_mode():
            mixed = scorer.model(batch["input_ids"], batch["attention_mask"], batch["candidate_counts"])
        mixed_scores = mixed.cpu().tolist()
    mixed_errors, padded_candidates_excluded = [], True
    reference_rows = {row["id"]: row for row in outputs}
    for episode, scores in zip(group, mixed_scores, strict=True):
        count = len(episode["entries"])
        regular = reference_rows[episode["id"]]["regular"]
        expected = [row["score"] for row in regular["candidateScores"]] + [regular["abstainScore"]]
        actual = scores[:count] + [scores[-1]]
        mixed_errors.extend(abs(a - b) for a, b in zip(expected, actual, strict=True))
        padded_candidates_excluded &= all(math.isinf(value) and value < 0 for value in scores[count:-1])
    write_json(args.output / (args.backend + "-batch-invariants.json"), {
        "mixed_candidate_count_max_abs": max(mixed_errors),
        "padding_candidates_are_negative_infinity": padded_candidates_excluded,
    })
    del scorer
    # Reload the same saved artifact in a new object; no re-save can change the
    # weights under comparison. Cross-process reload is additionally exercised
    # by the separate backend runs and the performance benchmark.
    reloaded = RankerScorer(args.model, backend=args.backend, device=args.device)
    for episode, output in zip(episodes, outputs, strict=True):
        restored = score_values(reloaded.score(episode))
        original = score_values(output["regular"])
        output["reload_max_abs"] = max(abs(original[key] - restored[key]) for key in original)
    write_jsonl(args.output / (args.backend + "-regression.jsonl"), outputs)


def compare(args):
    episodes = load_jsonl(args.inputs)
    reference = {row["id"]: row for row in load_jsonl(args.output / "torch-regression.jsonl")}
    deployment = {row["id"]: row for row in load_jsonl(args.output / "mlx-regression.jsonl")}
    if set(reference) != {row["id"] for row in episodes} or set(deployment) != set(reference):
        raise ValueError("Backend regression coverage mismatch")
    differences, rows = [], []
    calibrator = json.loads(args.calibrator.read_text()) if args.calibrator else None
    for episode in episodes:
        left, right = reference[episode["id"]], deployment[episode["id"]]
        left_scores, right_scores = score_values(left["regular"]), score_values(right["regular"])
        errors = [abs(left_scores[key] - right_scores[key]) for key in left_scores]
        differences.extend(errors)
        raw_match = equivalent(episode, raw_action(left["regular"]), raw_action(right["regular"]))
        current = {"id": episode["id"], "max_absolute_score_difference": max(errors),
                   "raw_action_agreement_allowing_identical_payload": raw_match,
                   "reference_raw_action": raw_action(left["regular"]), "mlx_raw_action": raw_action(right["regular"]),
                   "pair_token_lengths": right["pair_token_lengths"]}
        if calibrator:
            left_decision = calibrated_decision(episode, left["regular"], calibrator)
            right_decision = calibrated_decision(episode, right["regular"], calibrator)
            current["calibrated_decision_agreement_allowing_identical_payload"] = equivalent(
                episode, left_decision["recommendedID"], right_decision["recommendedID"])
        rows.append(current)
    invariants = {}
    for backend, mapping in (("torch", reference), ("mlx", deployment)):
        invariants[backend] = {name: max(row[name] for row in mapping.values()) for name in ("permutation_max_abs", "padding_max_abs", "reload_max_abs")}
        invariants[backend].update(json.loads((args.output / (backend + "-batch-invariants.json")).read_text()))
    checks = {"maximum_score_difference": max(differences) <= TOLERANCES["cross_backend_max_abs"],
              "mean_score_difference": sum(differences) / len(differences) <= TOLERANCES["cross_backend_mean_abs"],
              "raw_action_agreement": all(row["raw_action_agreement_allowing_identical_payload"] for row in rows)}
    for backend, values in invariants.items():
        for name in ("permutation_max_abs", "padding_max_abs"):
            checks[backend + "_" + name] = values[name] <= TOLERANCES[backend + "_padding_or_permutation_max_abs"]
        checks[backend + "_reload_exact"] = values["reload_max_abs"] <= TOLERANCES["save_reload_max_abs"]
        checks[backend + "_mixed_candidate_count"] = values["mixed_candidate_count_max_abs"] <= TOLERANCES[backend + "_padding_or_permutation_max_abs"]
        checks[backend + "_padding_candidates_excluded"] = values["padding_candidates_are_negative_infinity"]
    if calibrator:
        checks["calibrated_decision_agreement"] = all(row["calibrated_decision_agreement_allowing_identical_payload"] for row in rows)
    report = {"status": "passed" if all(checks.values()) else "failed", "version": PARITY_VERSION,
              "reference_weight_sha256": sha256(args.reference / "model.safetensors"),
              "mlx_weight_sha256": sha256(args.mlx / "model.safetensors"),
              "preprocess_sha256": sha256(args.mlx / "preprocess.json"),
              "regression_inputs_sha256": sha256(args.inputs), "calibrator_sha256": sha256(args.calibrator) if args.calibrator else None,
              "checks": checks, "tolerances": TOLERANCES, "cases": len(rows), "compared_scores": len(differences),
              "max_absolute_score_difference": max(differences), "mean_absolute_score_difference": sum(differences) / len(differences),
              "invariants": invariants, "rows": rows, "created_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Numerical export regression only; inputs are unlabeled and separate from Calibration/Test.",
              "gpu_exclusive_confirmation": args.gpu_exclusive_confirmation}
    destination = args.output / ("parity-calibrated.json" if args.calibrator else "parity.json")
    write_json(destination, report)
    print(json.dumps({key: report[key] for key in ("status", "cases", "checks", "max_absolute_score_difference", "mean_absolute_score_difference")}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mlx", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--backend", choices=("torch", "mlx"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--inputs", type=Path, default=Path("evaluations/regression-inputs.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path)
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--gpu-exclusive-confirmation", default="")
    args = parser.parse_args()
    if args.backend:
        run_backend(args)
        return
    if not args.reference or not args.mlx:
        raise SystemExit("Both reference and MLX deployment artifacts are required")
    if not args.compare_only:
        if not args.gpu_exclusive_confirmation:
            raise SystemExit("GPU exclusivity must be coordinated before running parity")
        if args.output.exists():
            raise SystemExit("Refusing to overwrite a numerical regression run")
        args.output.mkdir(parents=True)
        for backend, model in (("torch", args.reference), ("mlx", args.mlx)):
            subprocess.run([sys.executable, "-m", "evaluations.parity", "--backend", backend,
                            "--model", str(model), "--device", args.device, "--inputs", str(args.inputs),
                            "--output", str(args.output)], check=True)
    compare(args)


if __name__ == "__main__":
    main()
