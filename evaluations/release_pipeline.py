"""Conditionally authorized, durable evaluation handoff; never tunes on Test.

Waits for real training completion and complete independently audited heldout
data, then runs existing verification/calibration/scoring commands in sequence.
It never publishes a model or reads Train/Dev examples.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from data_tools.teacher import atomic_json
from evaluations.common import sha256, write_json
from evaluations.freeze import directory_hashes, heldout_audit_inputs, teacher_transition_inputs, verify_freeze
from evaluations.training_provenance import training_handoff_inputs
from run_contract import load_run_plan

ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_PLAN_SHA = "e5491476a01f3cd3d1b3f3778ac90f4e1dbeadca0e001a078003633266b7963f"
AUTHORIZATION = (
    "Parent /root conditional authorization received 2026-09-21: wait for this run's real complete training handoff and heldout datasets; "
    "verify training proofs, final MLX parity/performance, fit Calibration and its threshold, then freeze actual weights/preprocessing/policy/code "
    "and perform one final paired Test. No further parent reply is required once these conditions actually hold. "
    "Do not use Test for checkpoint/parameter selection or change labels/thresholds after Test. "
    "Registered plan SHA256=" + AUTHORIZED_PLAN_SHA
)
BASELINE_COMMIT = "87f9c096847f525a4c92958adce17a077c575844"
JEV_COMMIT = "e1663b79d42c2c4f4ecd1afbb2cf508c3192523e"


def now():
    return datetime.now(timezone.utc).isoformat()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("An immutable evaluation artifact changed: " + str(path))
    else:
        write_json(path, value)


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def readiness(plan):
    requirements = {"training_handoff": plan.pipeline_directory / "ready-for-calibration.json"}
    for split in ("calibration", "test"):
        requirements[split] = plan.data_path(split)
        requirements[split + "_audit"] = Path("reports/evaluation") / plan.run_id / ("data-" + split + "-audit.json")
        requirements[split + "_manifest"] = Path("data/evaluator-manifests") / plan.run_id / (split + ".manifest.json")
    for split in ("train", "dev"):
        requirements[split + "_manifest"] = plan.data_path(split).with_suffix(".manifest.json")
    return [name for name, path in requirements.items() if not path.is_file()]


class Pipeline:
    def __init__(self, plan):
        self.plan = plan
        self.directory = ROOT / "local/evaluator-release" / plan.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.reports = ROOT / "reports/evaluation" / plan.run_id / "final"
        self.reports.mkdir(parents=True, exist_ok=True)
        self.active_stage = "waiting_for_real_training_and_heldout"
        self.python = str(ROOT / ".venv/bin/python")

    def status(self, phase, **values):
        self.active_stage = phase
        record = {**self.plan.binding(), "phase": phase, "updated_at": now(), "pid": os.getpid(), **values}
        atomic_json(self.directory / "status.json", record)
        print(json.dumps(record), flush=True)

    def step(self, name, command, result):
        """Resume completed stages; never automatically rerun a partial Test."""
        self.plan.verify_unchanged()
        result = Path(result)
        record_path = self.directory / "steps" / (name + ".json")
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if record["command"] != command or record["result_path"] != str(result):
                raise ValueError("An evaluation stage command changed on resume: " + name)
            if record["status"] == "completed":
                if sha256(result) != record["result_sha256"]:
                    raise ValueError("Completed stage evidence changed: " + name)
                return
            if record.get("child_pid"):
                while process_alive(record["child_pid"]):
                    self.status(name, state="waiting_for_existing_stage_process", child_pid=record["child_pid"])
                    time.sleep(30)
            if not result.is_file():
                raise ValueError("Interrupted stage requires diagnosis; it will not be rerun automatically: " + name)
            record.update(status="completed", result_sha256=sha256(result), recovered_completed_artifact=True)
            atomic_json(record_path, record)
            return
        if result.exists():
            raise ValueError("Unexpected unregistered stage output: " + str(result))
        record = {**self.plan.binding(), "status": "starting", "command": command,
                  "result_path": str(result), "started_at": now()}
        atomic_json(record_path, record)
        self.status(name, state="running")
        log_path = self.directory / "logs" / (name + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"}, start_new_session=True)
            record.update(status="running", child_pid=process.pid)
            atomic_json(record_path, record)
            code = process.wait()
        if code != 0 or not result.is_file():
            record.update(status="failed", exit_code=code, finished_at=now())
            atomic_json(record_path, record)
            raise RuntimeError("Evaluation stage failed: " + name + "; see its private stage log")
        record.update(status="completed", result_sha256=sha256(result), exit_code=code, finished_at=now())
        atomic_json(record_path, record)

    def module(self, name, *args):
        return [self.python, "-m", "evaluations." + name, *(str(value) for value in args)]

    def data_manifest(self):
        result = {**self.plan.binding(), "teacher_contract_version": self.plan.document["teacher_contract_version"],
                  "human_validated": False, "splits": {}}
        transition_inputs = teacher_transition_inputs(self.plan)
        if transition_inputs:
            result["teacher_transition"] = file_record(transition_inputs["teacher_transition"])
            result["pi_runtime_pins"] = file_record(transition_inputs["pi_runtime_pins"])
        for split in ("train", "dev", "calibration", "test"):
            source = (self.plan.data_path(split).with_suffix(".manifest.json") if split in {"train", "dev"}
                      else Path("data/evaluator-manifests") / self.plan.run_id / (split + ".manifest.json"))
            value = json.loads(source.read_text())
            if (value.get("episodes") != self.plan.target(split) or value.get("human_validated") is not False or
                    any(value.get(key) != expected for key, expected in self.plan.binding().items())):
                raise ValueError("A complete split manifest differs from the registered run: " + split)
            result["splits"][split] = {key: value[key] for key in ("episodes", "sha256", "human_validated", "teacher_contract_version")}
            result["splits"][split]["source_manifest"] = file_record(source)
            if transition_inputs:
                if not value.get("teacher_sources"):
                    raise ValueError("A complete split lacks its actual per-role teacher distribution: " + split)
                result["splits"][split]["teacher_sources"] = value["teacher_sources"]
        path = self.reports / "data_manifest.json"
        immutable_json(path, result)
        return path

    def run(self):
        missing = readiness(self.plan)
        while missing:
            self.status("waiting_for_real_training_and_heldout", state="waiting", missing=missing,
                        student_test_started=False)
            time.sleep(30)
            self.plan.verify_unchanged()
            missing = readiness(self.plan)
        handoff_path = self.plan.pipeline_directory / "ready-for-calibration.json"
        handoff = json.loads(handoff_path.read_text())
        reference = Path(handoff["reference_checkpoint"]).resolve()
        deployment = Path(handoff["deployment_checkpoint"]).resolve()
        evidence = training_handoff_inputs(self.plan, deployment)
        for split in ("calibration", "test"):
            manifest = json.loads((Path("data/evaluator-manifests") / self.plan.run_id / (split + ".manifest.json")).read_text())
            audit = json.loads(Path(manifest["audit_path"]).read_text())
            if manifest["audit_sha256"] != sha256(manifest["audit_path"]) or audit.get("passed") is not True or audit.get("data_sha256") != sha256(self.plan.data_path(split)):
                raise ValueError("Heldout data is not complete with its original independent audit")
            # This is heldout-owner code. No Train/Dev examples are opened.
            episodes = [json.loads(line) for line in self.plan.data_path(split).read_text().splitlines() if line]
            heldout_audit_inputs(self.plan, split, episodes, audit)
        # The training pipeline holds this lock throughout every GPU stage and
        # releases it only after writing the completed handoff and exiting.
        gpu_lock = (self.plan.pipeline_directory / "pipeline.lock").open("a")
        while True:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                self.status("waiting_for_training_gpu_release", state="waiting")
                time.sleep(30)
        training_handoff_inputs(self.plan, deployment)
        gpu_confirmation = "Completed registered training proof verified; training pipeline.lock exclusively held by evaluator " + self.plan.run_id
        reservation = Path("local/pipeline/gpu-reservation.json")
        atomic_json(reservation, {**self.plan.binding(), "owner": "independent_evaluator", "pid": os.getpid(),
            "status": "reserved", "training_handoff_sha256": sha256(handoff_path), "acquired_at": now()})
        try:
            self.evaluate(reference, deployment, evidence, gpu_confirmation)
        finally:
            atomic_json(reservation, {**self.plan.binding(), "owner": "independent_evaluator", "pid": os.getpid(), "status": "released", "released_at": now()})
            fcntl.flock(gpu_lock, fcntl.LOCK_UN)
            gpu_lock.close()

    def evaluate(self, reference, deployment, evidence, gpu_confirmation):
        baseline = (ROOT / "local/frozen-comparators/laya-87f9c09").resolve()
        jev = (ROOT / "local/frozen-comparators/jev-e1663b7").resolve()
        baseline_model = (ROOT.parent / "laya-mlx/models/laya-multilingual").resolve()
        baseline_python = (ROOT.parent / "laya-mlx/.venv/bin/python").absolute()
        for root, expected in ((baseline, BASELINE_COMMIT), (jev, JEV_COMMIT)):
            snapshot = json.loads((root / "snapshot.json").read_text())
            if snapshot["revision"] != expected or any(sha256(root / name) != value for name, value in snapshot["files"].items()):
                raise ValueError("The preregistered comparator snapshot changed")
        ranker_command = [self.python, "-m", "pastewhat_ranker.worker", "--backend", "mlx", "--model", str(deployment)]
        baseline_command = [str(baseline_python), str(baseline / "worker.py"), "--backend", "mlx", "--model", str(baseline_model)]
        jev_command = [self.python, str(jev / "worker.py"), "--backend", "jev"]
        deployment_manifest = self.reports / "deployment-manifest.json"
        if not deployment_manifest.exists():
            write_json(deployment_manifest, {**self.plan.binding(), "reference_weight_sha256": sha256(reference / "model.safetensors"),
                "mlx_weight_sha256": sha256(deployment / "model.safetensors"), "preprocess_sha256": sha256(deployment / "preprocess.json"),
                "deployment_files_before_calibration": directory_hashes(deployment),
                "runtime_files": directory_hashes(ROOT / "src/pastewhat_ranker"),
                "baseline_runtime_files": directory_hashes(ROOT.parent / "laya-mlx/laya_mlx"),
                "baseline_uv_lock": file_record(ROOT.parent / "laya-mlx/uv.lock"),
                "python_executables": {"ranker": file_record(self.python), "baseline": file_record(baseline_python)},
                "platform": platform.platform(), "authorization": AUTHORIZATION,
                "training_proofs": {key: file_record(value) for key, value in evidence.items()}})
        registered = json.loads(deployment_manifest.read_text())
        if (registered["reference_weight_sha256"] != sha256(reference / "model.safetensors") or
                registered["mlx_weight_sha256"] != sha256(deployment / "model.safetensors") or
                registered["preprocess_sha256"] != sha256(deployment / "preprocess.json")):
            raise ValueError("The selected deployment changed during evaluation")
        parity_dir = self.reports / "numerical-parity"
        parity_args = ("--reference", reference, "--mlx", deployment, "--output", parity_dir,
                       "--gpu-exclusive-confirmation", gpu_confirmation)
        self.step("parity", self.module("parity", *parity_args), parity_dir / "parity.json")
        if json.loads((parity_dir / "parity.json").read_text())["status"] != "passed":
            raise ValueError("Independent MLX parity failed; final Test remains unopened")
        performance_dir = self.reports / "performance"
        self.step("performance", self.module("performance", "--command-json", json.dumps(ranker_command),
            "--tokenizer", deployment / "tokenizer", "--deployment-manifest", deployment_manifest,
            "--output", performance_dir, "--gpu-exclusive-confirmation", gpu_confirmation), performance_dir / "performance.json")
        cal_scores = self.directory / "calibration-scores"
        self.step("calibration_scores", self.module("score", "--run-plan", self.plan.path,
            "--data", self.plan.data_path("calibration"), "--split", "calibration", "--protocol", "ranker",
            "--command-json", json.dumps(ranker_command), "--output", cal_scores), cal_scores / "completion.json")
        if json.loads((cal_scores / "completion.json").read_text()).get("failures") != 0:
            raise ValueError("Calibration inference failed; do not fit on a silently reduced subset")
        calibration = self.reports / "calibration"
        self.step("calibration_fit", self.module("calibrate", "--run-plan", self.plan.path,
            "--data", self.plan.data_path("calibration"), "--scores", cal_scores / "scores.jsonl",
            "--deployment-manifest", deployment_manifest, "--weights", deployment / "model.safetensors",
            "--preprocess", deployment / "preprocess.json", "--output", calibration), calibration / "calibration-report.json")
        calibrator = calibration / "calibrator.json"
        self.step("calibrated_parity", self.module("parity", *parity_args, "--compare-only", "--calibrator", calibrator), parity_dir / "parity-calibrated.json")
        if json.loads((parity_dir / "parity-calibrated.json").read_text())["status"] != "passed":
            raise ValueError("Calibrated numerical parity failed; final Test remains unopened")
        installed = deployment / "calibrator.json"
        if installed.exists():
            if installed.read_bytes() != calibrator.read_bytes():
                raise ValueError("Deployment already contains a different calibration policy")
        else:
            shutil.copy2(calibrator, installed)
        data_manifest = self.data_manifest()
        freeze = self.reports / "final-test-freeze.json"
        self.step("freeze", self.module("freeze", "--run-plan", self.plan.path,
            "--deployment", deployment, "--baseline", baseline, "--baseline-model", baseline_model,
            "--baseline-commit", BASELINE_COMMIT, "--jev", jev, "--jev-commit", JEV_COMMIT,
            "--test", self.plan.data_path("test"), "--calibration", self.plan.data_path("calibration"),
            "--calibrator", installed, "--evaluation-manifest", deployment_manifest,
            "--authorization", AUTHORIZATION, "--output", freeze), freeze)
        verify_freeze(freeze, self.plan.data_path("test"))
        score_paths = {}
        for protocol, command in (("ranker", ranker_command), ("baseline", baseline_command), ("jev", jev_command)):
            output = self.directory / ("test-" + protocol)
            self.step("test_" + protocol, self.module("score", "--run-plan", self.plan.path,
                "--data", self.plan.data_path("test"), "--split", "test", "--protocol", protocol,
                "--command-json", json.dumps(command), "--freeze", freeze, "--output", output), output / "completion.json")
            score_paths[protocol] = output / "scores.jsonl"
        result_dir = self.reports / "acceptance"
        self.step("paired_report", self.module("report", "--run-plan", self.plan.path,
            "--data", self.plan.data_path("test"), "--baseline", score_paths["baseline"],
            "--ranker-scores", score_paths["ranker"], "--jev", score_paths["jev"],
            "--calibrator", installed, "--weights", deployment / "model.safetensors",
            "--preprocess", deployment / "preprocess.json", "--freeze", freeze, "--output", result_dir), result_dir / "metrics.json")
        verify_freeze(freeze, self.plan.data_path("test"))
        artifacts = {"freeze": file_record(freeze), "metrics": file_record(result_dir / "metrics.json"),
            "parity": file_record(parity_dir / "parity-calibrated.json"),
            "performance": file_record(performance_dir / "performance.json"),
            "calibration_report": file_record(calibration / "calibration-report.json"), "data_manifest": file_record(data_manifest)}
        for name, path in (("reference", reference), ("deployment", deployment)):
            manifest = self.reports / (name + "-directory-files.json")
            immutable_json(manifest, directory_hashes(path))
            artifacts[name] = {"path": str(path), "sha256": sha256(manifest),
                "manifest_path": str(manifest), "manifest_sha256": sha256(manifest),
                "weight_sha256": sha256(path / "model.safetensors")}
        metrics = json.loads((result_dir / "metrics.json").read_text())
        status = "accepted" if metrics["acceptance"]["passed"] else "diagnostic_quality_targets_not_met"
        ready = {**self.plan.binding(), "version": "pastewhat-evaluation-handoff-v1", "status": status,
            "completed_at": now(), "artifacts": artifacts, "acceptance": metrics["acceptance"],
            "human_validated": False, "test_parameters_frozen_before_scoring": True,
            "authorization": AUTHORIZATION, "publication_owner": "root"}
        immutable_json(self.directory / "ready-for-publication.json", ready)
        self.status("evaluation_complete", state=status, ready_for_publication=file_record(self.directory / "ready-for-publication.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Read prerequisites only; no model/API calls")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = load_run_plan(args.run_plan)
    if plan.sha256 != AUTHORIZED_PLAN_SHA:
        raise SystemExit("Conditional final-Test authorization applies only to the exact registered efficient plan")
    if args.check:
        print(json.dumps({**plan.binding(), "missing": readiness(plan), "student_inference_used": False}))
        return
    pipeline = Pipeline(plan)
    lock = (pipeline.directory / ".pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (pipeline.directory / "ready-for-publication.json").exists():
        print(json.dumps({"status": "already_complete", "handoff": str(pipeline.directory / "ready-for-publication.json")}))
        return
    atomic_json(pipeline.directory / "process.json", {**plan.binding(), "pid": os.getpid(), "started_at": now(), "authorization": AUTHORIZATION})
    try:
        pipeline.run()
    except Exception as error:
        pipeline.status(pipeline.active_stage, state="diagnostic_incomplete_requires_review",
                        error_type=type(error).__name__, reason=str(error), ready_for_publication=False)
        raise


if __name__ == "__main__":
    main()
