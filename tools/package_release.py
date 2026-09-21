"""Assemble an immutable, evaluated research bundle without altering frozen inputs.

This command does not train, calibrate, select a checkpoint, read Test examples,
or upload files. The evaluator's frozen reports are its only quality evidence.
Run from the repository root with: uv run python -m tools.package_release --help
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import yaml

from evaluations.freeze import directory_hashes, verify_freeze
from pastewhat_ranker.calibration import load_calibrator
from run_contract import ORIGINAL_SUGGESTED_TARGETS, load_run_plan


ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "052592a15d198d9ad47da779604259b10b47b7aa"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path.name}")
    return value


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected an ordinary artifact file: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(source) != digest(destination):
        raise ValueError(f"Artifact copy failed integrity check: {source.name}")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def percent(value) -> str:
    return "not defined" if value is None else f"{100 * value:.2f}%"


def model_card(metrics: dict, release_status: str, release_commit: str, manifest: dict) -> str:
    current, previous = metrics["ranker"]["overall"], metrics["baseline"]["overall"]
    passed = release_status == "accepted_on_frozen_synthetic_benchmark"
    statement = (
        "The preregistered acceptance criteria passed on the frozen synthetic benchmark."
        if passed else
        "This is a diagnostic research candidate. One or more release targets were not met; it is not an accepted production replacement."
    )
    counts = manifest["split_targets"]
    scale = (f"This release used {counts['train']:,} main Train, {counts['dev']:,} Dev, "
             f"{counts['calibration']:,} Calibration and {counts['test']:,} Test episodes. "
             + ("The pre-registered run plan and its measured cost rationale are bundled in `run_plan.json`. "
                "These are the actual registered sizes, not a claim that the original suggested sizes were completed."
                if manifest.get("run_plan_sha256") else "These are the original suggested split sizes."))
    return f"""---
license: apache-2.0
base_model: convaiinnovations/laya-multilingual
tags:
- clipboard-ranking
- cross-encoder
- multilingual
- mlx
- synthetic-data
- decision-label-distillation
---

# PasteWhat-Ranker-v1

{statement}

This model scores **existing** clipboard candidates and may return `null`. It does not generate paste contents or chain-of-thought. The encoder is fully fine-tuned from the original non-quantized Laya-multilingual checkpoint at revision `{SOURCE_REVISION}`. Its original task heads and calibration are replaced with a candidate scoring head and a candidate-group-aware abstention head.

The teacher is `kimi-for-coding`. Training targets are independently checked decision labels, not teacher logits or reasoning traces. All reported quality data are synthetic; agent and teacher review is **not independent human validation**, and these metrics do not establish real-user accuracy.

| Frozen Test metric | Existing PasteWhat workflow | This release |
|---|---:|---:|
| Answerable Top-1 with the actual recommendation policy | {percent(previous.get('answerable_top1'))} | {percent(current.get('answerable_top1'))} |
| Recommendation precision | {percent(previous.get('recommendation_precision'))} | {percent(current.get('recommendation_precision'))} |
| Recommendation coverage | {percent(previous.get('coverage'))} | {percent(current.get('coverage'))} |
| False promotion on abstention cases | {percent(previous.get('false_promotion_rate'))} | {percent(current.get('false_promotion_rate'))} |

Read `metrics.json` for counts, raw ranking, failure denominators, family-level comparisons, paired uncertainty and every acceptance criterion. `reports/performance.json` records the measured Mac, candidate counts, token lengths, cold start, warm latency and unified-memory usage. `reports/calibration.json` reports the observed precision/coverage tradeoff; it is not a guarantee on new contexts.

## Inputs and execution

The model uses application **category**, focused-field metadata, selected/surrounding text and 1–20 candidate contents with kind and payload capabilities. It does not use the real application name, bundle identifier, process ID or window title. Candidate IDs map outputs only. Each context–candidate pair has a fixed maximum of 1,024 tokens. All candidates are retained; secure or empty requests bypass inference. `context_projection/README.md` specifies the native `pastewhat-focus-v1` capture format: actual insertion boundaries and bounded adjacent static labels, rather than an imagined editing location. Its production Swift implementation and source hashes are included.

Use the [uv-locked runtime and inference protocol](https://github.com/mizorewww/pastewhat-ranker-v1/tree/{release_commit}) from this exact release commit. The `mlx/` directory is the FP16 deployment model. The root `model.safetensors` is the PyTorch reference model. The supplied calibrator is bound by hashes to the MLX precision and preprocessing and must not be reused with altered weights. The AppKit adapter only enables a policy that met the registered calibration target.

After downloading the complete model snapshot and installing that source revision with `uv sync --extra eval`, the shared inference API is:

```python
import json
from pathlib import Path
from pastewhat_ranker.worker import RankerScorer
from pastewhat_ranker.calibration import apply_calibration, load_calibrator

model = Path("/path/to/downloaded/PasteWhat-Ranker-v1/mlx")
episode = json.loads(Path("episode.json").read_text())
calibrator = load_calibrator(
    model / "calibrator.json",
    weights_path=model / "model.safetensors",
    preprocess_path=model / "preprocess.json",
    require_accepted=True,
)
scorer = RankerScorer(model, backend="mlx")
result = apply_calibration(episode, scorer.score(episode), calibrator)
print(json.dumps(result, ensure_ascii=False))
```

`episode.json` contains `context` and the complete `entries` list under the documented input contract, without teacher labels. The result preserves each candidate ID and score, adds the group abstention score, and returns `recommendedID` as an original ID or `null`. This checked example requires a policy that met the registered calibration target. For a diagnostic bundle that missed that target, `RankerScorer.score` still exposes raw scores for research; those raw scores are not calibrated recommendations.

`data_manifest.json` gives actual split counts, conceptual-family partitioning, provenance and hashes. Train, Dev, Calibration and Test have separate roles. The final Test was opened for scoring only after the deployment weights, preprocessing and policy were frozen. No real clipboard history or user contexts were used.

{scale}

## Limitations

The benchmark covers a bounded synthetic task distribution. New workflows, missing accessibility context, unusual languages, unfamiliar payloads and long truncated content can change reliability. A high softmax score is not itself recommendation correctness. Abstention reduces coverage, and the precision/coverage pair must be considered together. This model recommends content; it does not authorize actions described in that content.

## Provenance and license

Release status: `{release_status}`. Source commit: `{release_commit}`. Frozen evaluation manifest SHA-256: `{manifest['freeze_sha256']}`.

Apache-2.0. See `LICENSE`, `NOTICE`, `provenance/initialization.json`, `release_manifest.json` and the [source repository](https://github.com/mizorewww/pastewhat-ranker-v1) for upstream attribution and reproducibility. The release manifest hashes every bundled file except itself.
"""


def assemble(args) -> dict:
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing release bundle")
    frozen = verify_freeze(args.freeze)
    plan_record = frozen["inputs"].get("run_plan")
    plan = load_run_plan(plan_record["path"]) if plan_record else None
    if plan and plan.sha256 != plan_record["sha256"]:
        raise ValueError("Release plan differs from the final Test freeze")
    if args.deployment.resolve() != Path(frozen["deployment"]["root"]).resolve():
        raise ValueError("Deployment directory is not the independently frozen model")
    config = read_json(args.reference / "config.json")
    training = read_json(args.reference / "training_summary.json")
    if plan:
        train_config = yaml.safe_load((args.reference / "train_config.yaml").read_text())
        if any(train_config.get(key) != value for key, value in plan.binding().items()):
            raise ValueError("Reference training belongs to a different registered run")
    if config.get("architecture") != "PasteWhatRanker" or config.get("source_revision") != SOURCE_REVISION:
        raise ValueError("Unexpected architecture or initialization revision")
    if training.get("status") != "completed" or training.get("engineering_overfit") is not False or training.get("global_steps", 0) <= 0:
        raise ValueError("Only a completed non-engineering trained checkpoint can be packaged")
    reference_hash, mlx_hash = digest(args.reference / "model.safetensors"), digest(args.deployment / "model.safetensors")
    if training.get("best_weight_sha256") != reference_hash:
        raise ValueError("Reference weights differ from their recorded training result")
    conversion = read_json(args.deployment / "conversion.json")
    if conversion.get("reference_weight_sha256") != reference_hash or conversion.get("mlx_weight_sha256") != mlx_hash or conversion.get("strict_parameter_load") is not True:
        raise ValueError("MLX conversion is not bound to these reference/deployment weights")
    for name in ("preprocess.json", "tokenizer/tokenizer.json"):
        if digest(args.reference / name) != digest(args.deployment / name):
            raise ValueError(f"Reference and deployment differ in {name}")
    calibrator_path = args.deployment / "calibrator.json"
    if digest(calibrator_path) != frozen["inputs"]["calibrator"]["sha256"]:
        raise ValueError("Deployment calibrator differs from final Test freeze")
    calibrator = load_calibrator(calibrator_path, weights_path=args.deployment / "model.safetensors", preprocess_path=args.deployment / "preprocess.json")
    metrics, parity = read_json(args.metrics), read_json(args.parity)
    calibration_report, performance = read_json(args.calibration_report), read_json(args.performance)
    freeze_hash = digest(args.freeze)
    if metrics.get("provenance", {}).get("freeze_sha256") != freeze_hash:
        raise ValueError("Quality report does not refer to this final Test freeze")
    if parity.get("reference_weight_sha256") != reference_hash or parity.get("mlx_weight_sha256") != mlx_hash:
        raise ValueError("Parity report belongs to different weights")
    calibration_provenance = calibrator.get("provenance", {})
    if calibration_report.get("provenance") != calibration_provenance:
        raise ValueError("Calibration report does not describe the bundled policy")
    if calibration_provenance.get("dataset_sha256") != frozen["inputs"]["calibration"]["sha256"]:
        raise ValueError("Calibration dataset differs from the final freeze")
    deployment_manifest_hash = calibration_provenance.get("deployment_manifest_sha256")
    if not deployment_manifest_hash or performance.get("deployment_manifest_sha256") != deployment_manifest_hash:
        raise ValueError("Performance was not measured on the calibration deployment version")
    if {row.get("candidate_count") for row in performance.get("results", [])} != {1, 5, 10, 20}:
        raise ValueError("Performance report must cover 1, 5, 10 and 20 candidates")
    for row in performance["results"]:
        if row.get("runtime") != "mlx" or row.get("repeats", 0) < 1 or row.get("warm_inference_ms_p50", 0) <= 0:
            raise ValueError("Performance report lacks actual MLX timing")
    accepted = (metrics.get("acceptance", {}).get("passed") is True
                and parity.get("status") == "passed"
                and calibrator.get("status") == "observed_precision_target_met")
    if not accepted and not args.allow_diagnostic:
        raise ValueError("Release gates did not all pass; --allow-diagnostic explicitly labels a research candidate")
    if parity.get("status") != "passed":
        raise ValueError("A numerically unverified deployment cannot be published, including as a diagnostic candidate")
    data_manifest = read_json(args.data_manifest)
    if data_manifest.get("human_validated") is not False:
        raise ValueError("Synthetic manifest must explicitly avoid a human-validation claim")
    split_targets = plan.document["split_targets"] if plan else ORIGINAL_SUGGESTED_TARGETS
    if plan and any(data_manifest.get(key) != value for key, value in plan.binding().items()):
        raise ValueError("Dataset manifest belongs to a different registered run")
    for split, expected in split_targets.items():
        record = data_manifest.get("splits", {}).get(split, {})
        if record.get("episodes") != expected or not record.get("sha256"):
            raise ValueError(f"The complete planned {split} split is required before this release")
        if split in ("calibration", "test") and record["sha256"] != frozen["inputs"][split]["sha256"]:
            raise ValueError(f"Data manifest {split} differs from the final freeze")
    release_status = "accepted_on_frozen_synthetic_benchmark" if accepted else "research_candidate_quality_targets_not_met"
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".ranker-release-", dir=args.output.parent))
    try:
        for name in ("model.safetensors", "config.json", "preprocess.json", "train_config.yaml", "training_summary.json"):
            copy_file(args.reference / name, staging / name)
        # Copy only the inference artifact allowlist, never optimizer recovery,
        # teacher prompts, credential files or arbitrary contents of a folder.
        for source_root, destination_root in ((args.reference, staging), (args.deployment, staging / "mlx")):
            tokenizer = source_root / "tokenizer"
            for file in sorted(tokenizer.rglob("*")):
                if file.is_file():
                    copy_file(file, destination_root / "tokenizer" / file.relative_to(tokenizer))
        for name in ("model.safetensors", "config.json", "preprocess.json", "conversion.json", "calibrator.json"):
            copy_file(args.deployment / name, staging / "mlx" / name)
        copy_file(calibrator_path, staging / "calibrator.json")
        for name in ("Models.swift", "RecommendationContext.swift", "FocusText.swift", "CandidateProjection.swift",
                     "ProjectSyntheticContext.swift", "ProjectSyntheticCandidates.swift", "README.md", "provenance.json"):
            copy_file(ROOT / "tools/context_projection" / name, staging / "context_projection" / name)
        for source, destination in (
            (args.metrics, "metrics.json"), (args.data_manifest, "data_manifest.json"),
            (args.parity, "reports/parity.json"), (args.performance, "reports/performance.json"),
            (args.calibration_report, "reports/calibration.json"), (args.freeze, "provenance/final-test-freeze.json"),
            (ROOT / "provenance/initialization.json", "provenance/initialization.json"),
            (ROOT / "uv.lock", "uv.lock"), (ROOT / "pyproject.toml", "pyproject.toml"),
            (ROOT / "LICENSE", "LICENSE"), (ROOT / "NOTICE", "NOTICE"),
        ):
            copy_file(source, staging / destination)
        if plan:
            copy_file(plan.path, staging / "run_plan.json")
            copy_file(ROOT / "run_contract.py", staging / "provenance/run_contract.py")
            copy_file(ROOT / "RUN_PLAN_FORMAT.md", staging / "provenance/RUN_PLAN_FORMAT.md")
        manifest = {"version": "pastewhat-release-bundle-v1", "model_name": "PasteWhat-Ranker-v1",
                    "status": release_status, "created_at": datetime.now(timezone.utc).isoformat(),
                    "code_commit": code_commit, "freeze_sha256": freeze_hash,
                    "reference_weight_sha256": reference_hash, "mlx_weight_sha256": mlx_hash,
                    "synthetic_only": True, "human_validated": False, "split_targets": split_targets,
                    **(plan.binding() if plan else {})}
        card = model_card(metrics, release_status, code_commit, manifest)
        (staging / "README.md").write_text(card)
        (staging / "model_card.md").write_text(card)
        manifest["files"] = directory_hashes(staging)
        write_json(staging / "release_manifest.json", manifest)
        # Recheck read-only inputs after the potentially lengthy weight copies.
        verify_freeze(args.freeze)
        if plan:
            plan.verify_unchanged()
        if digest(args.reference / "model.safetensors") != reference_hash:
            raise ValueError("Reference weights changed during release assembly")
        staging.rename(args.output)
        return {"status": release_status, "directory": str(args.output), "files": len(manifest["files"]),
                "release_manifest_sha256": digest(args.output / "release_manifest.json")}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference", "deployment", "freeze", "metrics", "parity", "performance", "calibration-report", "data-manifest", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--allow-diagnostic", action="store_true")
    print(json.dumps(assemble(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
