"""Freeze three task-head seeds while proving every encoder tensor is identical."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from pastewhat_ranker.model import PasteWhatRanker, SOURCE_REPO, SOURCE_REVISION, sha256_file


def head_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith("encoder."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--base", default="checkpoints/initial")
    parser.add_argument("--output", default="reports/training/seed-initializations.json")
    args = parser.parse_args()
    target = Path(args.output)
    frozen = ({row["seed"]: row for row in json.loads(target.read_text())["seeds"]}
              if target.exists() else {})
    if args.source:
        source = Path(args.source)
    else:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION,
                                       allow_patterns=["model.safetensors", "encoder/config.json", "tokenizer/*"]))
    torch.set_num_threads(8)
    base = PasteWhatRanker.from_pretrained(args.base).eval()
    if base.config["initialization_seed"] != 42:
        raise ValueError("The pilot/base initialization must be seed 42")
    records = []
    for seed in (42, 43, 44):
        path = Path(args.base) if seed == 42 else Path(args.base).with_name("initial-seed-" + str(seed))
        if not (path / "model.safetensors").exists():
            fresh = PasteWhatRanker.initialize(source, seed=seed)
            fresh.save_pretrained(path, source / "tokenizer")
            del fresh
        model = PasteWhatRanker.from_pretrained(path).eval()
        if model.config["initialization_seed"] != seed:
            raise ValueError("Existing seed initialization has the wrong declared seed")
        reference = base.encoder.state_dict()
        candidate = model.encoder.state_dict()
        if reference.keys() != candidate.keys() or any(not torch.equal(reference[name], candidate[name]) for name in reference):
            raise ValueError("Encoder tensors differ between seed initializations")
        weight_sha = sha256_file(path / "model.safetensors")
        if seed in frozen and frozen[seed]["model_sha256"] != weight_sha:
            raise ValueError("Previously frozen seed initialization weights have changed")
        records.append({"seed": seed, "initial_model": str(path),
                        "model_sha256": weight_sha,
                        "head_sha256": head_hash(model), "encoder_tensors_compared": len(reference),
                        "encoder_tensor_equality_to_seed42": True,
                        "initialization_seed": seed, "training_seed": seed,
                        "randomness_scope": ["new_task_head_initialization", "training_episode_shuffle", "training_head_dropout"]})
        del model
    if len({row["head_sha256"] for row in records}) != 3:
        raise ValueError("Task heads must differ across all three seeds")
    result = {"status": "passed", "source_repo": SOURCE_REPO, "source_revision": SOURCE_REVISION,
              "selection": "The published seed is chosen only by Dev after the same main-training schedule",
              "pilot_and_main_seed42_share_identical_untrained_initialization": True,
              "seeds": records}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
