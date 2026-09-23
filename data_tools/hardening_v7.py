"""New v7 Train pool → Dev-selected v0 mining → one blind confirmation → mixture."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
from pathlib import Path
import random
import subprocess
import sys
import time

from data_tools.content import content_fingerprint
from data_tools.freeze import choose_registered, publish_bytes
from data_tools.resources import bounded_futures, worker_budget
from data_tools.teacher import atomic_json, audit_source, canonical_bytes, make_teacher_client, sha256, utc_now
from data_tools.v7 import PROTOCOL, ROOT, label_with_one_repair, remap_labels, same_action, visible_batch
from pastewhat_ranker.model import sha256_file
from pastewhat_ranker.preprocess import Preprocessor
from run_contract import load_run_plan


def read_train(path, plan):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT) or any(word in str(path.relative_to(ROOT)).lower() for word in ("heldout", "calibration", "test")):
        raise ValueError("Hardening only reads owned Train artifacts")
    allowed = {row["id"] for row in json.loads((ROOT / "data_tools/family_partition.json").read_text())["families"]["train"]}
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate Train identity")
    for row in rows:
        if row["family_id"] not in allowed or row["provenance"].get("teacher_contract_version") != PROTOCOL or any(row["provenance"].get(key) != value for key, value in plan.binding().items()):
            raise ValueError("Hardening source crosses run/protocol/family ownership")
    return rows


def idle_v0(plan):
    directory = ROOT / plan.pipeline_directory
    if not all((directory / name).exists() for name in ("ranker-v0-ready.json", "status.json")):
        return None
    ready = json.loads((directory / "ranker-v0-ready.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    if any(ready.get(key) != value or status.get(key) != value for key, value in plan.binding().items()):
        raise ValueError("v0 handoff/status belongs to a different plan")
    hard = plan.document["hardening"]
    expected = ROOT / plan.data_path("hardening")
    if ready.get("selected_by") != "Dev only":
        raise ValueError("v0 was not selected only on Dev")
    if status.get("phase") != "hardening_preparation" or status.get("status") != "waiting_for_frozen_train_dev_data" or Path(status.get("required_path", "")).resolve() != expected.resolve() or status.get("required_count") != hard["accepted_new"] + hard["retained_original"] or expected.exists():
        return None
    return ready


def run(command, log, plan):
    plan.verify_unchanged()
    with log.open("a") as stream:
        stream.write(json.dumps({"at": utc_now(), "command": command}) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Hardening subprocess failed: {log.relative_to(ROOT)}")


def review(group, client, directory, plan):
    digest = sha256(canonical_bytes({"rows": group, "binding": plan.binding(), "protocol": PROTOCOL, "teacher_runtime": getattr(client, "runtime", None)}))
    path = directory / (digest + ".json")
    if path.exists():
        return json.loads(path.read_text())
    visible, mapping = visible_batch(group, int(digest[:12], 16), "independent-post-mining")
    audits = []
    labels, label_audits, errors = label_with_one_repair(client, visible, phase="v7-post-mining-blind-confirmation", request_id=digest, remember=lambda result: audits.append(result.audit_id))
    labels = remap_labels(labels, mapping)
    audit_for = {mapping[key][0]: result.audit_id for key, result in label_audits.items()}
    accepted, rejected = [], []
    for row in group:
        observed = labels.get(row["id"])
        evidence = {"id": row["id"], "content_sha256": content_fingerprint(row), "original_label": row["label"], "observed_label": observed, "audit_id": audit_for.get(row["id"])}
        if evidence["audit_id"]:
            evidence["teacher_source"] = audit_source(json.loads((client.audit_dir / (evidence["audit_id"] + ".json")).read_text()))
        if observed is not None and same_action(row["label"], observed):
            accepted.append(evidence)
        else:
            rejected.append({**evidence, "reason": "Blind confirmation disagrees or no valid label; original label remains unchanged"})
    result = {**plan.binding(), "input_sha256": sha256(canonical_bytes(group)), "teacher_contract_version": PROTOCOL, "accepted": accepted, "rejected": rejected, "format_errors": errors, "audit_ids": sorted(set(audits)), "student_predictions_in_teacher_input": False, "teacher_label_edits": 0, "reviewed_at": utc_now()}
    atomic_json(path, result)
    return result


def publish(original, proposals, records, selection, plan, preprocessor, directory):
    confirmed = {row["id"]: row for record in records for row in record["accepted"]}
    selected = [row for row in proposals if row["id"] in confirmed][:plan.document["hardening"]["accepted_new"]]
    if len(selected) != plan.document["hardening"]["accepted_new"]:
        atomic_json(directory / "insufficient-confirmed-new.json", {**plan.binding(), "confirmed": len(confirmed), "required": plan.document["hardening"]["accepted_new"], "reason": "Do not rerun reviewers until preferred labels appear; a new registered candidate pool is required"})
        raise ValueError("Too few confirmed new cases in the finite nomination pool")
    old_hashes = {content_fingerprint(row) for row in original}
    new = []
    for row in selected:
        if content_fingerprint(row) in old_hashes:
            raise ValueError("New hardening case repeats original Train content")
        prepared = preprocessor.prepare_episode(row)
        if prepared["context"] != row["context"] or prepared["entries"] != row["entries"] or prepared["preprocessing"]["visible_sha256"] != row["provenance"]["visible_sha256"]:
            raise ValueError("Post-mining input changed from the original teacher view")
        item = copy.deepcopy(row)
        item["provenance"]["post_mining_confirmation"] = confirmed[row["id"]]
        new.append(item)
    partition_path = ROOT / "data_tools/family_partition.json"
    old = choose_registered(original, json.loads(partition_path.read_text()), "train", plan.document["hardening"]["retained_original"])
    chosen = old + new
    random.Random(42).shuffle(chosen)
    fingerprints = [content_fingerprint(row) for row in chosen]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("Hardening mixture repeats visible content")
    payload = b"".join(canonical_bytes(row) + b"\n" for row in chosen)
    output = ROOT / plan.data_path("hardening")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {**plan.binding(), "stage": "hardening", "split": "train", "episodes": len(chosen), "sha256": sha256(payload), "family_partition_sha256": sha256(partition_path.read_bytes()), "teacher_contract_version": PROTOCOL, "retained_original": len(old), "accepted_new": len(new), "original_rows_preserved_byte_for_byte": True, "mining_proposals_sha256": selection["proposals_sha256"], "confirmed_new": len(confirmed), "confirmation_rejected": sum(len(record["rejected"]) for record in records), "student_disagreement_is_not_teacher_truth": True, "human_validated": False, "labels": dict(Counter(row["label"]["decision"] for row in chosen)), "created_at": utc_now()}
    correction = ROOT / "configs/teacher_correction_swe2_uid.json"
    if correction.is_file():
        manifest["teacher_correction"] = {"path": str(correction.relative_to(ROOT)), "sha256": sha256(correction.read_bytes())}
    atomic_json(output.with_suffix(".fingerprints.json"), {**plan.binding(), "content_sha256": sorted(fingerprints)})
    atomic_json(output.with_suffix(".manifest.json"), manifest)
    publish_bytes(output, payload)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-plan", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    if plan.document["teacher_contract_version"] != PROTOCOL:
        raise ValueError("This coordinator only runs v7")
    directory = ROOT / "local/v7" / plan.run_id / "hardening"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = ROOT / plan.data_path("hardening")
    if output.exists():
        manifest = json.loads(output.with_suffix(".manifest.json").read_text())
        if manifest["sha256"] != sha256(output.read_bytes()) or any(manifest.get(key) != value for key, value in plan.binding().items()):
            raise ValueError("Existing hardening snapshot differs from the plan")
        return
    original_path = ROOT / plan.data_path("train")
    while not original_path.exists():
        atomic_json(directory / "status.json", {**plan.binding(), "phase": "waiting_for_main_train_snapshot", "updated_at": utc_now()})
        time.sleep(30)
    original = read_train(original_path, plan)
    pool_path = ROOT / "local/v7" / plan.run_id / "hard-pool/train.jsonl"
    pool_completion = pool_path.parent / "train.run-completion.json"
    if not pool_completion.exists() or json.loads(pool_completion.read_text()).get("status") != "complete":
        run([sys.executable, "-m", "data_tools.generate_v7", "--run-plan", str(plan.path), "--split", "train", "--hard-pool", "--workers", str(args.workers), "--backfill-rounds", "12"], directory / "pool.log", plan)
    pool = read_train(pool_path, plan)
    if len(pool) != plan.document["hardening"]["pool_episodes"]:
        raise ValueError("New pool did not reach its registered size")
    while not (ready := idle_v0(plan)):
        atomic_json(directory / "status.json", {**plan.binding(), "phase": "waiting_for_idle_dev_selected_v0", "updated_at": utc_now()})
        time.sleep(30)
    ready_path = ROOT / plan.pipeline_directory / "ranker-v0-ready.json"
    model = directory / "v0-mlx"
    if not (model / "conversion.json").exists():
        run([sys.executable, "-m", "pastewhat_ranker.export", "--model", ready["checkpoint"], "--output", str(model)], directory / "export.log", plan)
    if not idle_v0(plan):
        raise RuntimeError("Training is no longer waiting; GPU mining cannot start")
    mining = directory / "mining"
    selection_path = mining / "selection.json"
    if not selection_path.exists():
        command = [sys.executable, "-m", "tools.mine_training_pool", "--pool", str(pool_path), "--original-train", str(original_path), "--v0-ready", str(ready_path), "--model", str(model), "--output", str(mining), "--run-plan", str(plan.path), "--gpu-exclusive-confirmation", "Pipeline idle hardening wait verified; status SHA " + sha256((ROOT / plan.pipeline_directory / "status.json").read_bytes())]
        if mining.exists():
            command.append("--resume")
        run(command, directory / "mining.log", plan)
    selection = json.loads(selection_path.read_text())
    provenance = json.loads((mining / "provenance.json").read_text())
    checks = {**plan.binding(), "version": "pastewhat-train-mining-v1", "requested_proposals": plan.document["hardening"]["review_nominations"], "v0_handoff_sha256": sha256_file(ready_path), "pool_sha256": sha256_file(pool_path), "original_train_sha256": sha256_file(original_path), "mlx_weight_sha256": sha256_file(model / "model.safetensors"), "preprocess_sha256": sha256_file(model / "preprocess.json"), "mining_source_sha256": sha256_file(ROOT / "tools/mine_training_pool.py")}
    if any(provenance.get(key) != value for key, value in checks.items()):
        raise ValueError("Cached mining provenance changed")
    proposals = read_train(mining / "proposals.jsonl", plan)
    if (sha256((mining / "proposals.jsonl").read_bytes()) != selection["proposals_sha256"]
            or selection.get("status") != "requires_blind_teacher_review_not_training_ready"
            or selection.get("proposals") != len(proposals)
            or len(proposals) != plan.document["hardening"]["review_nominations"]
            or any(selection.get(key) != value for key, value in plan.binding().items())):
        raise ValueError("Cached proposals changed")
    client = make_teacher_client(directory / "teacher")
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        batches = (proposals[index:index + 10] for index in range(0, len(proposals), 10))
        submit = lambda pool, batch: pool.submit(review, batch, client, directory / "reviews", plan)
        capacity = lambda: worker_budget(plan, "train", args.workers, hard_pool=True)
        for future in bounded_futures(executor, batches, submit, capacity):
            records.append(future.result())
            atomic_json(directory / "status.json", {**plan.binding(), "phase": "blind_post_mining_confirmation", "accepted": sum(len(record["accepted"]) for record in records), "rejected": sum(len(record["rejected"]) for record in records), "updated_at": utc_now()})
    manifest = publish(original, proposals, records, selection, plan, Preprocessor(str(ROOT.parent / "laya-mlx/models/laya-multilingual/tokenizer")), directory)
    atomic_json(directory / "complete.json", manifest)


if __name__ == "__main__":
    main()
