"""Reproducible full-encoder training, with recovery and Dev-only selection.

Training code refuses Calibration/Test paths. Persisted manifests identify the
exact data, source weights, software, code, random seed, and completed progress.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .model import PasteWhatRanker, collate_encoded, group_loss, sha256_file
from .preprocess import Preprocessor


def read_allowed_data(path):
    path = Path(path)
    forbidden = ("test", "calibration", "heldout", "held-out")
    if any(part.lower().startswith(forbidden) for part in path.parts):
        raise ValueError("Training code must never inspect Calibration/Test/heldout data")
    episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({episode["id"] for episode in episodes}) != len(episodes):
        raise ValueError("Duplicate episode IDs in training input")
    if not episodes:
        raise ValueError("Empty dataset")
    return episodes


def config_hash(config):
    import hashlib
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def git_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def precision_context(config):
    if config.get("precision", "bf16") == "float32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if config.get("precision", "bf16") == "bf16" else torch.float16
    return torch.autocast(config.get("device", "mps"), dtype=dtype)


def evaluate_dev(model, episodes, encoded, preprocessor, config):
    model.eval()
    correct = selected_correct = selected_count = abstain_count = abstain_correct = promotions = 0
    total_loss, records, groups = 0.0, [], {}
    device = config.get("device", "mps")
    batch_size = config.get("eval_batch_episodes", 4)
    with torch.inference_mode():
        for start in range(0, len(episodes), batch_size):
            batch_episodes = episodes[start:start + batch_size]
            batch = collate_encoded(batch_episodes, encoded[start:start + batch_size], preprocessor.pad_id, device)
            # Dev uses full FP32 reference precision, independent of training AMP.
            logits = model(batch["input_ids"], batch["attention_mask"], batch["candidate_counts"])
            total_loss += group_loss(logits, batch["positive_mask"]).item() * len(batch_episodes)
            for row, episode, ids, count in zip(logits.cpu().tolist(), batch_episodes, batch["candidate_ids"], batch["candidate_counts"]):
                position = max(range(count), key=row.__getitem__)
                recommended = ids[position] if row[position] > row[-1] else None
                expected = episode["label"]
                good = (recommended in expected["acceptable_ids"] if expected["decision"] == "select" else recommended is None)
                correct += good
                promotions += recommended is not None
                if expected["decision"] == "select":
                    selected_count += 1
                    selected_correct += good
                else:
                    abstain_count += 1
                    abstain_correct += good
                records.append({"id": episode["id"], "recommended_id": recommended, "correct": good})
                group_names = ["application:" + episode["context"].get("applicationCategory", "unknown")]
                if expected["decision"] == "select":
                    kinds = sorted({entry["kind"] for entry in episode["entries"] if entry["id"] in expected["acceptable_ids"]})
                    group_names += ["kind:" + kind for kind in kinds]
                else:
                    group_names += ["abstain:" + str(expected.get("abstain_reason", "unknown"))]
                for name in group_names:
                    group = groups.setdefault(name, {"episodes": 0, "correct": 0})
                    group["episodes"] += 1
                    group["correct"] += int(good)
    select_rate = selected_correct / max(1, selected_count)
    abstain_rate = abstain_correct / max(1, abstain_count)
    for group in groups.values():
        group["decision_accuracy"] = group["correct"] / group["episodes"]
    return {"episodes": len(episodes), "loss": total_loss / len(episodes), "groups": groups,
            "decision_accuracy": correct / len(episodes), "correct_decisions": correct,
            "answerable_top1": select_rate, "correct_answerable": selected_correct, "answerable": selected_count,
            "abstain_accuracy": abstain_rate, "correct_abstain": abstain_correct, "abstain": abstain_count,
            "no_answer_false_promotion_rate": 1 - abstain_rate,
            "coverage": promotions / len(episodes),
            "recommendation_precision": selected_correct / max(1, promotions),
            "selection_metric": (select_rate + abstain_rate) / 2 if abstain_count and selected_count else correct / len(episodes),
            "selection_metric_name": "balanced_select_abstain_accuracy",
            "records": records}


def make_optimizer(model, config, phase):
    head_only = phase == "head_warmup"
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(not head_only)
    model.encoder.gradient_checkpointing = bool(config.get("gradient_checkpointing", True) and not head_only)
    heads = [parameter for name, parameter in model.named_parameters() if not name.startswith("encoder.")]
    groups = [{"params": heads, "lr": config["head_lr"], "initial_lr": config["head_lr"]}]
    if not head_only:
        groups.append({"params": list(model.encoder.parameters()), "lr": config["encoder_lr"], "initial_lr": config["encoder_lr"]})
    return torch.optim.AdamW(groups, weight_decay=config.get("weight_decay", 0.01), foreach=False)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_recovery(directory, model, optimizer, state, tokenizer):
    import shutil
    import uuid
    latest = directory / "latest"
    previous = latest.resolve() if latest.exists() else None
    # A new complete generation is written first. An atomic symlink swap publishes
    # it, keeping the previous complete generation usable across interruption.
    temporary = directory / (".saving-" + uuid.uuid4().hex)
    model.save_pretrained(temporary, tokenizer)
    payload = {"optimizer": optimizer.state_dict(), "state": state,
               "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
               "numpy_rng": np.random.get_state()}
    if config_device(model) == "mps":
        payload["mps_rng"] = torch.mps.get_rng_state()
    torch.save(payload, temporary / "optimizer.pt")
    state = dict(state, model_sha256=sha256_file(temporary / "model.safetensors"), optimizer_sha256=sha256_file(temporary / "optimizer.pt"))
    atomic_json(temporary / "progress.json", state)
    generation = directory / ("recovery-" + str(state["global_step"]) + "-" + uuid.uuid4().hex[:8])
    temporary.rename(generation)
    next_pointer = directory / (".latest-" + uuid.uuid4().hex)
    next_pointer.symlink_to(generation.name, target_is_directory=True)
    os.replace(next_pointer, latest)
    atomic_json(directory / "progress.json", state)
    if previous is not None and previous.parent == directory.resolve() and previous.name.startswith("recovery-"):
        shutil.rmtree(previous)


def config_device(model):
    return next(model.parameters()).device.type


def run(config, output, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists() and not resume:
        raise ValueError("Run directory already exists; use --resume or a new output path")
    torch.set_num_threads(config.get("cpu_threads", 8))
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.get("device", "mps") == "mps":
        torch.mps.manual_seed(seed)
    train = read_allowed_data(config["train_data"])
    dev = read_allowed_data(config["dev_data"])
    if config.get("train_limit"):
        # Deterministic first N from a separately frozen shuffled training snapshot.
        train = train[:config["train_limit"]]
    families = {episode["family_id"] for episode in train}
    overlap = families & {episode["family_id"] for episode in dev}
    if overlap and not config.get("engineering_overfit", False):
        raise ValueError("Train and Dev conceptual families overlap")
    p = Preprocessor(Path(config["initial_model"]) / "tokenizer")
    encoded_train = [p.encode_episode(episode) for episode in train]
    encoded_dev = [p.encode_episode(episode) for episode in dev]
    effective = config.get("effective_batch_episodes", 16)
    micro = config.get("micro_batch_episodes", 1)
    if not 1 <= micro <= effective:
        raise ValueError("Invalid micro/effective episode batch sizes")
    initial_manifest = {
        "config": config, "config_sha256": config_hash(config), "code_revision": git_revision(),
        "train_sha256": sha256_file(config["train_data"]), "dev_sha256": sha256_file(config["dev_data"]),
        "train_episodes": len(train), "dev_episodes": len(dev), "train_families": len(families),
        "initial_weight_sha256": sha256_file(Path(config["initial_model"]) / "model.safetensors"),
        "preprocess": p.manifest(), "python": platform.python_version(), "torch": torch.__version__,
        "platform": platform.platform(), "started_unix": time.time(),
        "selection": "Dev balanced_select_abstain_accuracy; tie: decision_accuracy, lower Dev loss",
        "training_scope": "full encoder plus both newly initialized heads after head warmup",
        "training_source_sha256": {name: sha256_file(Path(__file__).parent / name)
                                   for name in ("train.py", "model.py", "encoder.py", "preprocess.py")},
    }
    prior = json.loads((output / "run_manifest.json").read_text()) if resume else initial_manifest
    for key in ("config_sha256", "train_sha256", "dev_sha256", "initial_weight_sha256", "training_source_sha256"):
        if initial_manifest[key] != prior[key]:
            raise ValueError(f"Resume provenance mismatch: {key}")
    if not resume:
        atomic_json(output / "run_manifest.json", initial_manifest)
        (output / "train_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    phases = (["head_warmup"] if config.get("head_warmup_steps", 200) else []) + ["full"]
    state = {"phase": phases[0], "epoch": 0, "offset": 0, "phase_step": 0, "global_step": 0,
             "seen_episodes": 0, "best_key": [-1, -1, -1], "completed": False,
             "config_sha256": config_hash(config)}
    device = config.get("device", "mps")
    if resume:
        latest = output / "latest"
        recovery_state = json.loads((latest / "progress.json").read_text())
        if recovery_state["model_sha256"] != sha256_file(latest / "model.safetensors") or recovery_state["optimizer_sha256"] != sha256_file(latest / "optimizer.pt"):
            raise ValueError("Incomplete recovery write: checkpoint checksums disagree")
        model = PasteWhatRanker.from_pretrained(latest, device=device)
        recovery = torch.load(latest / "optimizer.pt", map_location="cpu", weights_only=False)
        state = recovery["state"]
        if state["completed"]:
            return json.loads((output / "training_summary.json").read_text())
        optimizer = make_optimizer(model, config, state["phase"])
        optimizer.load_state_dict(recovery["optimizer"])
        torch.set_rng_state(recovery["torch_rng"])
        random.setstate(recovery["python_rng"])
        np.random.set_state(recovery["numpy_rng"])
        if device == "mps":
            torch.mps.set_rng_state(recovery["mps_rng"])
    else:
        model = PasteWhatRanker.from_pretrained(config["initial_model"], device=device)
        optimizer = make_optimizer(model, config, state["phase"])
    scaler = torch.amp.GradScaler(device, enabled=config.get("precision", "bf16") == "fp16")
    log = (output / "events.jsonl").open("a", buffering=1)
    start_time = time.monotonic()
    last_saved = state["global_step"]
    stop = False
    try:
        while not stop:
            phase = state["phase"]
            head_only = phase == "head_warmup"
            epoch = state["epoch"]
            total_steps = (config.get("head_warmup_steps", 200) if head_only
                           else math.ceil(len(train) / effective) * config.get("epochs", 2))
            order = list(range(len(train)))
            random.Random(seed + epoch + (1_000_000 if head_only else 0)).shuffle(order)
            # Optional bucket sorting within small shuffled windows lowers padding
            # while retaining every episode/candidate and deterministic boundaries.
            if config.get("length_bucket_window", 0):
                window = config["length_bucket_window"]
                for i in range(0, len(order), window):
                    order[i:i + window] = sorted(order[i:i + window], key=lambda j: max(map(len, encoded_train[j]["input_ids"])))
            for offset in range(state["offset"], len(order), effective):
                indices = order[offset:offset + effective]
                update_start = time.monotonic()
                model.train()
                optimizer.zero_grad(set_to_none=True)
                warm_steps = max(1, math.ceil(total_steps * config.get("warmup_ratio", 0.05)))
                step_number = state["phase_step"] + 1
                scale = min(1.0, step_number / warm_steps)
                if step_number > warm_steps:
                    scale = max(0.0, (total_steps - step_number) / max(1, total_steps - warm_steps))
                if config.get("constant_lr", False):
                    scale = 1.0
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * scale
                loss_sum = torch.zeros((), device=device)
                pairs = tokens = 0
                for micro_start in range(0, len(indices), micro):
                    chosen = indices[micro_start:micro_start + micro]
                    batch_episodes = [train[i] for i in chosen]
                    batch = collate_encoded(batch_episodes, [encoded_train[i] for i in chosen], p.pad_id, device)
                    with precision_context(config):
                        logits = model(batch["input_ids"], batch["attention_mask"], batch["candidate_counts"])
                        loss = group_loss(logits, batch["positive_mask"]) * (len(chosen) / len(indices))
                    scaler.scale(loss).backward()
                    loss_sum += loss.detach()
                    pairs += sum(batch["candidate_counts"])
                    tokens += int(batch["attention_mask"].sum().item())
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                if not bool(torch.isfinite(grad_norm)) or not bool(torch.isfinite(loss_sum)):
                    raise FloatingPointError("Nonfinite loss or gradients; run halted without accepting the step")
                scaler.step(optimizer)
                scaler.update()
                state.update(offset=offset + len(indices), phase_step=step_number,
                             global_step=state["global_step"] + 1, seen_episodes=state["seen_episodes"] + len(indices))
                event = {"event": "update", "phase": phase, "epoch": epoch + 1,
                         "step": state["global_step"], "phase_step": step_number,
                         "episodes": len(indices), "pairs": pairs, "tokens": tokens,
                         "loss": loss_sum.item(), "gradient_norm": grad_norm.item(),
                         "seconds": time.monotonic() - update_start,
                         "elapsed_seconds": time.monotonic() - start_time,
                         "learning_rates": [group["lr"] for group in optimizer.param_groups]}
                log.write(json.dumps(event) + "\n")
                if state["global_step"] % config.get("log_every_steps", 10) == 0:
                    print(json.dumps(event), flush=True)
                if state["global_step"] - last_saved >= config.get("save_every_steps", 100):
                    save_recovery(output, model, optimizer, state, p_path(config))
                    last_saved = state["global_step"]
                if head_only and step_number >= total_steps:
                    break
                if config.get("max_steps") and state["global_step"] >= config["max_steps"]:
                    stop = True
                    break
            phase_finished = head_only and state["phase_step"] >= total_steps
            if not head_only:
                dev_report = evaluate_dev(model, dev, encoded_dev, p, config)
                record_path = output / f"dev-epoch-{epoch + 1:02d}.json"
                atomic_json(record_path, dev_report)
                brief = {key: value for key, value in dev_report.items() if key != "records"}
                log.write(json.dumps({"event": "dev", "epoch": epoch + 1, **brief}) + "\n")
                print(json.dumps({"event": "dev", "epoch": epoch + 1, **brief}), flush=True)
                key = [dev_report["selection_metric"], dev_report["decision_accuracy"], -dev_report["loss"]]
                if key > state["best_key"]:
                    state["best_key"] = key
                    model.save_pretrained(output / "best", p_path(config))
                    atomic_json(output / "best" / "dev_metrics.json", brief)
                    atomic_json(output / "best" / "selection.json", {"epoch": epoch + 1, "step": state["global_step"], "key": key})
                if config.get("engineering_overfit") and dev_report["decision_accuracy"] >= config.get("overfit_target", 0.99):
                    stop = True
                if epoch + 1 >= config.get("epochs", 2):
                    stop = True
            if phase_finished:
                state.update(phase="full", epoch=0, offset=0, phase_step=0)
                optimizer = make_optimizer(model, config, "full")
                log.write(json.dumps({"event": "unfreeze_all_encoder_parameters", "step": state["global_step"]}) + "\n")
            elif not stop:
                state.update(epoch=epoch + 1, offset=0)
            if stop:
                state["completed"] = True
            save_recovery(output, model, optimizer, state, p_path(config))
            last_saved = state["global_step"]
    finally:
        log.close()
    summary = {"status": "completed", "global_steps": state["global_step"],
               "seen_episodes_including_head_warmup": state["seen_episodes"],
               "best_dev_key": state["best_key"], "elapsed_this_process_seconds": time.monotonic() - start_time,
               "best_checkpoint": str(output / "best"), "manifest": str(output / "run_manifest.json"),
               "engineering_overfit": bool(config.get("engineering_overfit", False))}
    atomic_json(output / "training_summary.json", summary)
    if (output / "best").exists():
        atomic_json(output / "best" / "training_summary.json", summary)
        (output / "best" / "train_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return summary


def p_path(config):
    return Path(config["initial_model"]) / "tokenizer"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(run(config, args.output, args.resume), indent=2))


if __name__ == "__main__":
    main()
