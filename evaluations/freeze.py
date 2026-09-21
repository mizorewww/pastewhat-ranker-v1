"""Freeze deployment artifacts before opening the final held-out test."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from evaluations.common import load_jsonl, require_plan_data_path, sha256, validate_formal_heldout_allocation, write_json
from run_contract import load_run_plan
from pastewhat_ranker.calibration import VERSION as CALIBRATOR_VERSION
from evaluations.training_provenance import training_handoff_inputs
from evaluations.observations import SUPPLEMENT_PATH, ADAPTER_PATH, assignment_path, validate_allocation as validate_observation_allocation
from data_tools.teacher import canonical_bytes
import hashlib


V7_CONTRACT = "teacher-episodes-v7-batched-decisions"


def heldout_audit_inputs(plan, split, episodes, audit):
    """Validate the protocol-specific data evidence without exposing examples."""
    if plan.document["teacher_contract_version"] != V7_CONTRACT:
        expected = validate_observation_allocation(episodes, plan.binding(), split)
        if audit.get("observation_supplement") != expected:
            raise ValueError("Legacy observation supplement differs from its accepted data audit")
        return {"observation_supplement": SUPPLEMENT_PATH, "observation_adapter": ADAPTER_PATH,
                split + "_observation_assignment": assignment_path(plan.binding(), split),
                "candidate_label_protocol": Path("data_tools/labeling.py")}
    variants = dict(Counter(row["provenance"]["observation_variant"] for row in episodes))
    if (audit.get("teacher_contract_version") != V7_CONTRACT or audit.get("observation_counts") != variants or
            variants.get("no_accessibility", 0) != len(episodes) * 4 // 100 or
            variants.get("generic_field", 0) != len(episodes) * 2 // 100):
        raise ValueError("V7 initial observation sampling differs from the accepted data audit")
    evidence = audit.get("teacher_and_batch_files")
    if not isinstance(evidence, dict) or not evidence or hashlib.sha256(canonical_bytes(evidence)).hexdigest() != audit.get("teacher_audit_bundle_sha256"):
        raise ValueError("V7 final freeze requires the complete source and raw decision-audit bundle")
    inputs = {"v7_teacher_protocol": Path("data_tools/v7.py"),
              "teacher_client_source": Path("data_tools/teacher.py"),
              "teacher_rate_control_source": Path("data_tools/rate_limit.py"),
              "observation_adapter": Path("data_tools/observations.py"),
              "v7_evaluation_protocol": Path("docs/EVALUATION_V7.md")}
    for index, (path, expected_hash) in enumerate(sorted(evidence.items())):
        if sha256(path) != expected_hash:
            raise ValueError("An audited v7 source or teacher response changed before final freeze")
        inputs[f"v7_{split}_evidence_{index:05d}"] = Path(path)
    return inputs


def directory_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise ValueError(f"Artifact root does not exist: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store")
    if not paths:
        raise ValueError("Cannot freeze an empty artifact directory")
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def verify_freeze(path: Path, dataset: Path | None = None) -> dict:
    frozen = json.loads(path.read_text())
    if frozen.get("status") != "frozen_for_final_test":
        raise ValueError("Final Test requires an explicitly approved freeze")
    plan = load_run_plan(frozen["inputs"]["run_plan"]["path"])
    if any(frozen.get(key) != value for key, value in plan.binding().items()):
        raise ValueError("Final freeze belongs to a different registered production plan")
    for section in ("deployment", "baseline", "baseline_model", "runtime_code", "evaluation_code", "context_projection"):
        root = Path(frozen[section]["root"])
        actual = directory_hashes(root)
        if actual != frozen[section]["files"]:
            raise ValueError(f"Frozen {section} files changed")
    if "baseline_runtime" in frozen and directory_hashes(Path(frozen["baseline_runtime"]["root"])) != frozen["baseline_runtime"]["files"]:
        raise ValueError("Frozen Laya comparator runtime source changed")
    if "jev" in frozen:
        if directory_hashes(Path(frozen["jev"]["root"])) != frozen["jev"]["files"]:
            raise ValueError("Frozen Jev client files changed")
    for name, item in frozen["inputs"].items():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen {name} changed")
    if dataset and sha256(dataset) != frozen["inputs"]["test"]["sha256"]:
        raise ValueError("The requested dataset is not the frozen final Test")
    if dataset:
        require_plan_data_path(plan, "test", dataset)
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--baseline-runtime", type=Path, default=Path("../laya-mlx/laya_mlx"))
    parser.add_argument("--evaluation-manifest", type=Path, help="Immutable deployment/runtime/authorization record written before final Test")
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--jev", type=Path, help="Optional pinned Jev client snapshot; remote weights remain rolling")
    parser.add_argument("--jev-commit")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--calibration-audit", type=Path)
    parser.add_argument("--test-audit", type=Path)
    parser.add_argument("--preprocess-source", type=Path, default=Path("src/pastewhat_ranker/preprocess.py"))
    parser.add_argument("--family-partition", type=Path, default=Path("data_tools/family_partition.json"))
    parser.add_argument("--authorization", required=True, help="Exact parent-agent freeze authorization reference, not a fabricated user approval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    args.calibration_audit = args.calibration_audit or Path("reports/evaluation") / plan.run_id / "data-calibration-audit.json"
    args.test_audit = args.test_audit or Path("reports/evaluation") / plan.run_id / "data-test-audit.json"
    calibrator = json.loads(args.calibrator.read_text())
    if calibrator.get("version") != CALIBRATOR_VERSION:
        raise SystemExit("Expected the deployment-version correctness calibrator")
    if any(calibrator.get("provenance", {}).get(key) != value for key, value in plan.binding().items()):
        raise SystemExit("Calibrator belongs to a different registered production plan")
    training_inputs = training_handoff_inputs(plan, args.deployment)
    allocation = {}
    protocol_inputs = {}
    partition = json.loads(args.family_partition.read_text())
    partition_hash = sha256(args.family_partition)
    for split, dataset, audit_path in (("calibration", args.calibration, args.calibration_audit),
                                       ("test", args.test, args.test_audit)):
        require_plan_data_path(plan, split, dataset)
        episodes = load_jsonl(dataset)
        allocation[split] = validate_formal_heldout_allocation(episodes, split, partition, plan)
        audit = json.loads(audit_path.read_text())
        if (audit.get("passed") is not True or audit.get("formal_run") is not True or audit.get("split") != split or
                audit.get("episodes") != allocation[split]["episodes"] or
                any(audit.get(key) != value for key, value in plan.binding().items()) or
                audit.get("data_sha256") != sha256(dataset) or audit.get("partition_sha256") != partition_hash):
            raise SystemExit("Final freeze requires a passing audit bound to the complete " + split + " data")
        protocol_inputs.update(heldout_audit_inputs(plan, split, episodes, audit))
    inputs = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in {
        "test": args.test, "calibration": args.calibration, "calibrator": args.calibrator,
        "run_plan": args.run_plan, "run_contract_source": Path("run_contract.py"),
        "dependency_lock": Path("uv.lock"), "package_manifest": Path("pyproject.toml"),
        "baseline_dependency_lock": Path("../laya-mlx/uv.lock"),
        "preprocess_source": args.preprocess_source, "family_partition": args.family_partition,
        "context_projection_adapter": Path("tools/project_context.py"),
        "candidate_projection_adapter": Path("tools/project_candidates.py"),
        "compact_authoring_builder": Path("data_tools/authoring.py"),
        "evaluation_protocol": Path("docs/EVALUATION_PROTOCOL.md"),
        "calibration_audit": args.calibration_audit, "test_audit": args.test_audit,
        **({"evaluation_manifest": args.evaluation_manifest} if args.evaluation_manifest else {}),
        **training_inputs, **protocol_inputs,
    }.items()}
    record = {
        "version": "pastewhat-release-freeze-v1", "status": "frozen_for_final_test",
        "authorization": args.authorization, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "baseline_commit": args.baseline_commit,
        "deployment": {"root": str(args.deployment.resolve()), "files": directory_hashes(args.deployment)},
        "baseline": {"root": str(args.baseline.resolve()), "files": directory_hashes(args.baseline)},
        "baseline_model": {"root": str(args.baseline_model.resolve()), "files": directory_hashes(args.baseline_model)},
        "baseline_runtime": {"root": str(args.baseline_runtime.resolve()), "files": directory_hashes(args.baseline_runtime)},
        "runtime_code": {"root": str(Path("src/pastewhat_ranker").resolve()), "files": directory_hashes(Path("src/pastewhat_ranker"))},
        "evaluation_code": {"root": str(Path("evaluations").resolve()), "files": directory_hashes(Path("evaluations"))},
        "context_projection": {"root": str(Path("tools/context_projection").resolve()), "files": directory_hashes(Path("tools/context_projection"))},
        "inputs": inputs, "calibration_status": calibrator["status"],
        "heldout_allocation": allocation,
        **plan.binding(),
        "quality_target": {"answerable_top1_delta": 0.05, "key_group_maximum_decline": 0.05,
                           "key_group_minimum_answerable": 30, "recommendation_precision": 0.95},
    }
    if args.jev:
        record["jev"] = {"root": str(args.jev.resolve()), "files": directory_hashes(args.jev),
                         "commit": args.jev_commit, "model_weights": "rolling remote; actual responding versions recorded per request"}
    write_json(args.output, record)
    verify_freeze(args.output, args.test)
    print(json.dumps({"frozen": True, "manifest_sha256": sha256(args.output), "calibration_status": calibrator["status"]}))


if __name__ == "__main__":
    main()
