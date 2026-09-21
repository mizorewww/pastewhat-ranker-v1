"""Publish a completed independent evaluation, then enable accepted app artifacts.

This waiter never opens Calibration/Test examples, fits a model, or selects a
policy. The existing assembler rechecks the actual frozen training/evaluation
evidence before any remote write. Credentials come from the Hub's cached login.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

from evaluations.freeze import directory_hashes, verify_freeze
from run_contract import load_run_plan
from tools.package_release import digest, read_json

ROOT = Path(__file__).resolve().parents[1]
REPO = "aac6fef/PasteWhat-Ranker-v1"
ACCEPTED = "accepted_on_frozen_synthetic_benchmark"
ARTIFACTS = ("reference", "deployment", "freeze", "metrics", "parity",
             "performance", "calibration_report", "data_manifest")
SECRET_PATTERNS = (rb"sk-kimi-[A-Za-z0-9]{16,}", rb"apikey_[A-Za-z0-9_]{24,}",
                   rb"hf_[A-Za-z0-9]{25,}")


def write_private(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".publication-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_record(path: Path, value: dict) -> None:
    write_private(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def check_record(record: dict, *, directory: bool = False) -> Path:
    path = Path(record["path"])
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
        raise ValueError("Publication inputs must be ordinary artifacts under this repository")
    if directory:
        manifest = Path(record["manifest_path"])
        if manifest.is_symlink() or not manifest.resolve().is_relative_to(ROOT):
            raise ValueError("Directory manifests must be ordinary artifacts under this repository")
        if manifest.resolve().is_relative_to(path.resolve()):
            raise ValueError("A directory manifest cannot include itself")
        if digest(manifest) != record["manifest_sha256"] or record["sha256"] != record["manifest_sha256"]:
            raise ValueError("Directory handoff manifest changed")
        if directory_hashes(path) != read_json(manifest):
            raise ValueError("Directory bytes differ from their completed handoff")
        if digest(path / "model.safetensors") != record["weight_sha256"]:
            raise ValueError("Handoff weights changed")
    elif digest(path) != record["sha256"]:
        raise ValueError("Publication handoff artifact changed")
    return path


def verify_bundle(path: Path, plan, freeze: Path) -> dict:
    manifest = read_json(path / "release_manifest.json")
    if any(manifest.get(key) != value for key, value in plan.binding().items()):
        raise ValueError("An existing bundle belongs to another run")
    if manifest["freeze_sha256"] != digest(freeze):
        raise ValueError("An existing bundle belongs to another final evaluation")
    if any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError("Release bundles cannot contain symlinks")
    actual = directory_hashes(path)
    actual.pop("release_manifest.json")
    if actual != manifest["files"]:
        raise ValueError("Release bundle changed after assembly")
    for name in actual:
        file = path / name
        if file.suffix != ".safetensors" and file.stat().st_size <= 20 * 1024 * 1024:
            content = file.read_bytes()
            if any(re.search(pattern, content) for pattern in SECRET_PATTERNS):
                raise ValueError("Credential-shaped content found in the release allowlist")
    verify_freeze(freeze)
    return manifest


def source_ready() -> str:
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT).returncode:
        raise ValueError("Commit source changes before publishing their release")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    # Completed training/evaluation can create new local reports. They are
    # bundled by the assembler's explicit evidence list, never auto-staged.
    if any(name and not name.startswith("reports/") for name in untracked):
        raise ValueError("Untracked implementation files must be committed before publication")
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=ROOT, text=True).strip()
    if remote not in {"https://github.com/mizorewww/pastewhat-ranker-v1.git",
                      "https://github.com/mizorewww/pastewhat-ranker-v1",
                      "git@github.com:mizorewww/pastewhat-ranker-v1.git"}:
        raise ValueError("Unexpected source publication repository")
    for name in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"):
        if not name:
            continue
        path = ROOT / name
        if path.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Large artifacts must not be published to source Git")
        if any(re.search(pattern, path.read_bytes()) for pattern in SECRET_PATTERNS):
            raise ValueError("Credential-shaped content found in tracked source")
    subprocess.run(["git", "push", "origin", "HEAD:refs/heads/main"], cwd=ROOT, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def verify_remote(api, bundle: Path, revision: str) -> None:
    info = api.model_info(REPO, revision=revision, files_metadata=True)
    if info.private:
        raise ValueError("The requested public model repository became private")
    remote = {item.rfilename: item for item in info.siblings}
    local = directory_hashes(bundle)
    if set(remote) - {".gitattributes"} != set(local):
        raise ValueError("Remote file list differs from the release allowlist")
    for name in local:
        path, item = bundle / name, remote[name]
        if item.size != path.stat().st_size:
            raise ValueError("Published file size mismatch")
        if item.lfs is not None:
            valid = item.lfs.sha256 == local[name]
        else:
            content = path.read_bytes()
            valid = item.blob_id == hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if not valid:
            raise ValueError("Published file hash mismatch")


def upload(bundle: Path, manifest: dict) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    if api.whoami().get("name") != "aac6fef":
        raise ValueError("Unexpected Hub account")
    info = api.model_info(REPO, files_metadata=True)
    if info.private:
        raise ValueError("The model repository must be public")
    names = {item.rfilename for item in info.siblings}
    if "release_manifest.json" in names:
        previous = Path(hf_hub_download(REPO, "release_manifest.json", revision=info.sha))
        if digest(previous) != digest(bundle / "release_manifest.json"):
            raise ValueError("A different release already occupies this model repository")
        revision = info.sha
    else:
        if names - {"README.md", ".gitattributes"}:
            raise ValueError("Unexpected files in the pending model repository")
        commit = api.upload_folder(
            repo_id=REPO, repo_type="model", folder_path=bundle,
            allow_patterns=list(directory_hashes(bundle)), parent_commit=info.sha,
            commit_message="Publish trained PasteWhat ranker and frozen evaluation artifacts",
            commit_description="Release status: " + manifest["status"] + ". Synthetic benchmark only.",
        )
        revision = commit.oid
    verify_remote(api, bundle, revision)
    return {"repo_id": REPO, "revision": revision, "url": f"https://huggingface.co/{REPO}/tree/{revision}",
            "remote_hashes_verified": True}


def app_smoke(bundle: Path, destination: Path) -> dict:
    # These are public, unlabeled backend regression inputs, never heldout data.
    app = ROOT.parent / "pastewhat"
    source_hashes = directory_hashes(app / "engine")
    weight_hash = digest(bundle / "mlx/model.safetensors")
    calibration_hash = digest(bundle / "mlx/calibrator.json")
    if destination.exists():
        previous = read_json(destination)
        if (previous.get("status") == "passed" and previous.get("weights_sha256") == weight_hash
                and previous.get("calibrator_sha256") == calibration_hash
                and previous.get("app_engine_hashes") == source_hashes):
            return previous
    rows = [json.loads(line) for line in (ROOT / "evaluations/regression-inputs.jsonl").read_text().splitlines()]
    cases = []
    for count in (1, 5, 10, 20):
        row = next(item for item in rows if len(item["entries"]) == count)
        cases.append({"id": f"publication-{count}", "context": row["context"], "entries": row["entries"]})
    cases.extend(({"id": "publication-empty", "context": cases[0]["context"], "entries": []},
                  {"id": "publication-secure", "context": {**cases[0]["context"], "isSecure": True}, "entries": cases[0]["entries"]}))
    result = subprocess.run([str(ROOT / ".venv/bin/python"), "-u", str(app / "engine/worker.py"),
                             "--backend", "ranker", "--model", str(bundle / "mlx")],
                            input="".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
                            capture_output=True, text=True, timeout=480, check=True)
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    if [row.get("id") for row in outputs] != [row["id"] for row in cases]:
        raise ValueError("App worker lost request identity")
    for case, output in zip(cases, outputs, strict=True):
        bypass = not case["entries"] or case["context"].get("isSecure")
        if output.get("decision") in {"model_unavailable", "invalid_request", "error"}:
            raise ValueError("Released model is not usable through the app adapter")
        if bypass:
            if output.get("recommendedID") is not None or output.get("inferenceCount") != 0:
                raise ValueError("The app failed a secure/empty inference bypass")
        elif ({item["id"] for item in output.get("rankings", [])} != {item["id"] for item in case["entries"]}
              or output.get("recommendedID") not in {None, *(item["id"] for item in case["entries"])}):
            raise ValueError("The app changed the full candidate set or output IDs")
    record = {"status": "passed", "cases": len(cases), "candidate_counts": [1, 5, 10, 20],
              "empty_secure_bypass": True, "synthetic_unlabeled_inputs": True,
              "weights_sha256": weight_hash, "calibrator_sha256": calibration_hash,
              "app_engine_hashes": source_hashes,
              "latency_ms": [row.get("elapsedMS") for row in outputs[:4]],
              "claim": "App protocol integration, not recommendation accuracy or live accessibility capture."}
    write_record(destination, record)
    return record


def enable_app(bundle: Path, plan) -> dict:
    app_root = ROOT.parent / "pastewhat"
    app = app_root / "dist/PasteWhat.app"
    subprocess.run(["codesign", "--verify", "--strict", str(app)], check=True, capture_output=True)
    for source in (app_root / "engine").glob("*.py"):
        if digest(source) != digest(app / "Contents/Resources/engine" / source.name):
            raise ValueError("Build the current app adapter before enabling the model")
    support = Path.home() / "Library/Application Support/PasteWhat"
    support.mkdir(parents=True, exist_ok=True)
    destination = support / "engine.json"
    backup = support / f"engine.before-{plan.run_id}.json"
    if destination.exists() and not backup.exists():
        write_private(backup, destination.read_bytes())
    configuration = {"backend": "ranker", "pythonPath": str(ROOT / ".venv/bin/python"), "modelPath": str(bundle / "mlx")}
    write_record(destination, configuration)
    binary = app / "Contents/MacOS/PasteWhat"
    processes = subprocess.run(["pgrep", "-x", "PasteWhat"], capture_output=True, text=True)
    for value in processes.stdout.split():
        observed = subprocess.run(["ps", "-ww", "-p", value, "-o", "comm="], capture_output=True, text=True)
        if observed.returncode:
            continue
        command = observed.stdout.strip()
        if command == str(binary):
            children = subprocess.run(["pgrep", "-P", value], capture_output=True, text=True)
            for child in children.stdout.split():
                try:
                    os.kill(int(child), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                os.kill(int(value), signal.SIGTERM)
            except ProcessLookupError:
                continue
            for _ in range(50):
                try:
                    os.kill(int(value), 0)
                except ProcessLookupError:
                    break
                time.sleep(.1)
            else:
                raise ValueError("The previous app instance has not exited")
    subprocess.run(["open", str(app)], check=True)
    return {"configured": True, "backend": "ranker", "configuration_path": str(destination),
            "previous_configuration_backup": str(backup), "app_path": str(app),
            "live_accessibility_capture_verified": False}


def record_publication(completed: dict, metrics_path: Path, plan) -> None:
    """Commit only the public receipt and owned status documents, not raw runs."""
    receipt = ROOT / "reports/publication" / (plan.run_id + ".json")
    public_record = {key: completed[key] for key in (*plan.binding(), "status", "release_manifest_sha256",
                                                    "evaluation_handoff_sha256", "publication", "app_protocol")}
    public_record["app"] = {key: value for key, value in completed["app"].items()
                            if key in {"configured", "backend", "reason", "live_accessibility_capture_verified"}}
    write_record(receipt, public_record)
    url = completed["publication"]["url"]
    state = completed["status"]
    announcement = ("A trained model passed the frozen synthetic release criteria and is published."
                    if state == ACCEPTED else
                    "A trained diagnostic model is published; the frozen synthetic release criteria were not all met.")
    pending = ("**Training and data production are in progress. No trained, calibrated or accepted release is claimed yet.** "
               "The implemented model and passing engineering checks below establish that the pipeline runs, not that recommendation accuracy has improved. "
               "See [execution status](RUN_STATUS.md) for completed and remaining work.")
    readme = ROOT / "README.md"
    text = readme.read_text()
    if pending in text:
        readme.write_text(text.replace(pending, f"**{announcement}** [Published model]({url}). "
                          "See [execution status](RUN_STATUS.md) for measured acceptance results. "
                          "Engineering checks and predeclared targets below are separate from the final benchmark.", 1))
    metrics = read_json(metrics_path)
    status = (f"# Execution status\n\n{announcement}\n\n"
              f"Registered run: `{plan.run_id}`. Release status: `{state}`.\n\n"
              f"[Published model and frozen artifacts]({url}) · [Publication receipt](reports/publication/{plan.run_id}.json)\n\n"
              "The registered data quotas, pilot, diagnostic learning curve, all three main seeds, "
              "hard-example training round, independent MLX verification, calibration, and final frozen Test "
              "have completed. Checkpoint selection used Dev; the final Test did not select training parameters.\n\n"
              f"Acceptance results: `{json.dumps(metrics['acceptance'], sort_keys=True)}`.\n\n"
              "Quality figures describe synthetic data only. Agent/teacher review is not human validation. "
              "The app protocol was checked with public synthetic inputs when the calibration policy was usable. "
              "Live accessibility capture and real-user accuracy remain unverified.\n")
    (ROOT / "RUN_STATUS.md").write_text(status)
    names = [str(receipt.relative_to(ROOT)), "README.md", "RUN_STATUS.md"]
    subprocess.run(["git", "add", "--", *names], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--", *names], cwd=ROOT).returncode:
        subprocess.run(["git", "commit", "--only", "-m", "Record trained ranker publication and measured release status", "--", *names], cwd=ROOT, check=True)
    subprocess.run(["git", "push", "origin", "HEAD:refs/heads/main"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-plan", type=Path, required=True)
    parser.add_argument("--once", action="store_true", help="Exit without publishing when evaluation is not ready")
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    local = ROOT / "local/publication" / plan.run_id
    local.mkdir(parents=True, exist_ok=True)
    status_path = local / "status.json"
    ready_path = ROOT / "local/evaluator-release" / plan.run_id / "ready-for-publication.json"
    phase = "waiting_for_independent_evaluation"

    def status(**extra):
        write_record(status_path, {**plan.binding(), "phase": phase, "updated_at": datetime.now(timezone.utc).isoformat(), **extra})

    with (local / "publisher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            while not ready_path.exists():
                status(required_handoff=str(ready_path))
                if args.once:
                    return
                time.sleep(30)
            ready = read_json(ready_path)
            if (ready.get("version") != "pastewhat-evaluation-handoff-v1"
                    or ready.get("status") not in {"accepted", "diagnostic_quality_targets_not_met"}):
                raise ValueError("Independent evaluation is not in a completed publication state")
            if any(ready.get(key) != value for key, value in plan.binding().items()):
                raise ValueError("Publication handoff belongs to another run")
            paths = {name: check_record(ready["artifacts"][name], directory=name in {"reference", "deployment"}) for name in ARTIFACTS}
            phase = "assembling_verified_release"
            status()
            source_ready()
            bundle = ROOT / "artifacts/PasteWhat-Ranker-v1"
            if not bundle.exists():
                # The waiter can outlive implementation work. Load the assembler
                # only after the completed handoff and committed source checks.
                command = [sys.executable, "-m", "tools.package_release"]
                for name, path in paths.items():
                    command.extend(["--" + name.replace("_", "-"), str(path)])
                command.extend(["--output", str(bundle), "--allow-diagnostic"])
                subprocess.run(command, cwd=ROOT, check=True)
            manifest = verify_bundle(bundle, plan, paths["freeze"])
            smoke = None
            if read_json(bundle / "mlx/calibrator.json")["status"] == "observed_precision_target_met":
                phase = "checking_app_protocol"
                status()
                smoke = app_smoke(bundle, local / "app-integration.json")
            phase = "publishing_and_verifying_hub_files"
            status()
            publication = upload(bundle, manifest)
            verify_bundle(bundle, plan, paths["freeze"])
            app = {"configured": False, "reason": "Quality targets were not all met; current app configuration retained."}
            if manifest["status"] == ACCEPTED:
                phase = "enabling_accepted_app_model"
                status()
                app = enable_app(bundle, plan)
            completed = {**plan.binding(), "status": manifest["status"], "bundle": str(bundle),
                         "release_manifest_sha256": digest(bundle / "release_manifest.json"),
                         "evaluation_handoff_sha256": digest(ready_path), "publication": publication,
                         "app": app, "app_protocol": smoke}
            phase = "recording_publication"
            record_publication(completed, paths["metrics"], plan)
            write_record(local / "completed.json", completed)
            phase = "published"
            status(**completed)
            print(json.dumps(completed, ensure_ascii=False), flush=True)
        except Exception as error:
            status(error_type=type(error).__name__, publication_complete=False)
            raise SystemExit("Publication stopped in " + phase + ": " + type(error).__name__) from None


if __name__ == "__main__":
    main()
