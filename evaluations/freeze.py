"""Freeze deployment artifacts before opening the final held-out test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from evaluations.common import load_jsonl, sha256, validate_formal_heldout_allocation, write_json


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
    for section in ("deployment", "baseline", "baseline_model", "runtime_code", "evaluation_code", "context_projection"):
        root = Path(frozen[section]["root"])
        actual = directory_hashes(root)
        if actual != frozen[section]["files"]:
            raise ValueError(f"Frozen {section} files changed")
    if "jev" in frozen:
        if directory_hashes(Path(frozen["jev"]["root"])) != frozen["jev"]["files"]:
            raise ValueError("Frozen Jev client files changed")
    for name, item in frozen["inputs"].items():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen {name} changed")
    if dataset and sha256(dataset) != frozen["inputs"]["test"]["sha256"]:
        raise ValueError("The requested dataset is not the frozen final Test")
    return frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--jev", type=Path, help="Optional pinned Jev client snapshot; remote weights remain rolling")
    parser.add_argument("--jev-commit")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--calibration-audit", type=Path, default=Path("reports/data-calibration-audit.json"))
    parser.add_argument("--test-audit", type=Path, default=Path("reports/data-test-audit.json"))
    parser.add_argument("--preprocess-source", type=Path, default=Path("src/pastewhat_ranker/preprocess.py"))
    parser.add_argument("--family-partition", type=Path, default=Path("data_tools/family_partition.json"))
    parser.add_argument("--authorization", required=True, help="Exact parent-agent freeze authorization reference, not a fabricated user approval")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibrator = json.loads(args.calibrator.read_text())
    if calibrator.get("version") != "pastewhat-calibrator-v1":
        raise SystemExit("Expected the deployment-version correctness calibrator")
    allocation = {}
    partition = json.loads(args.family_partition.read_text())
    partition_hash = sha256(args.family_partition)
    for split, dataset, audit_path in (("calibration", args.calibration, args.calibration_audit),
                                       ("test", args.test, args.test_audit)):
        allocation[split] = validate_formal_heldout_allocation(load_jsonl(dataset), split, partition)
        audit = json.loads(audit_path.read_text())
        if (audit.get("passed") is not True or audit.get("split") != split or
                audit.get("episodes") != allocation[split]["episodes"] or
                audit.get("data_sha256") != sha256(dataset) or audit.get("partition_sha256") != partition_hash):
            raise SystemExit("Final freeze requires a passing audit bound to the complete " + split + " data")
    inputs = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in {
        "test": args.test, "calibration": args.calibration, "calibrator": args.calibrator,
        "preprocess_source": args.preprocess_source, "family_partition": args.family_partition,
        "context_projection_adapter": Path("tools/project_context.py"),
        "evaluation_protocol": Path("docs/EVALUATION_PROTOCOL.md"),
        "calibration_audit": args.calibration_audit, "test_audit": args.test_audit,
    }.items()}
    record = {
        "version": "pastewhat-release-freeze-v1", "status": "frozen_for_final_test",
        "authorization": args.authorization, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "baseline_commit": args.baseline_commit,
        "deployment": {"root": str(args.deployment.resolve()), "files": directory_hashes(args.deployment)},
        "baseline": {"root": str(args.baseline.resolve()), "files": directory_hashes(args.baseline)},
        "baseline_model": {"root": str(args.baseline_model.resolve()), "files": directory_hashes(args.baseline_model)},
        "runtime_code": {"root": str(Path("src/pastewhat_ranker").resolve()), "files": directory_hashes(Path("src/pastewhat_ranker"))},
        "evaluation_code": {"root": str(Path("evaluations").resolve()), "files": directory_hashes(Path("evaluations"))},
        "context_projection": {"root": str(Path("tools/context_projection").resolve()), "files": directory_hashes(Path("tools/context_projection"))},
        "inputs": inputs, "calibration_status": calibrator["status"],
        "heldout_allocation": allocation,
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
