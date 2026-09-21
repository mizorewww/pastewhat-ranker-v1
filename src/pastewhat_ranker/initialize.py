"""Freeze the original non-quantized Laya encoder and new task heads locally."""

import argparse
import json
from pathlib import Path

from .model import PasteWhatRanker, SOURCE_REPO, SOURCE_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Existing original upstream snapshot; otherwise download the pinned revision")
    parser.add_argument("--output", default="checkpoints/initial")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    target = Path(args.output)
    if (target / "model.safetensors").exists():
        raise SystemExit("Initialization already exists; use a new output directory to avoid changing a frozen source")
    if args.source:
        source = Path(args.source)
    else:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION,
                                       allow_patterns=["model.safetensors", "encoder/config.json", "tokenizer/*"]))
    model = PasteWhatRanker.initialize(source, seed=args.seed)
    model.save_pretrained(target, source / "tokenizer")
    print(json.dumps({"output": str(target), "source_repo": SOURCE_REPO,
                      "source_revision": SOURCE_REVISION,
                      "initialization_audit": model.config["initialization_audit"]}, indent=2))


if __name__ == "__main__":
    main()
