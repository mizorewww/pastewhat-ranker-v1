"""Convert the complete learned encoder and both heads to MLX FP16, strictly."""

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.torch import load_file

from .mlx_model import MLXRanker
from .model import sha256_file


def export_model(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    config.update(deployment_backend="mlx", deployment_precision="float16",
                  reference_weight_sha256=sha256_file(source / "model.safetensors"))
    tensors = load_file(str(source / "model.safetensors"))
    converted = {}
    for name, tensor in tensors.items():
        for head in ("candidate_score_head", "abstain_head"):
            if name.startswith(head + "."):
                name = head + ".layers." + name[len(head) + 1:]
        if name in converted:
            raise ValueError("Weight conversion produced a name collision")
        converted[name] = mx.array(tensor.float().numpy().astype(np.float16))
    model = MLXRanker(config)
    model.load_weights(list(converted.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    mx.save_safetensors(str(destination / "model.safetensors"), converted)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copytree(source / "tokenizer", destination / "tokenizer", dirs_exist_ok=True)
    shutil.copyfile(source / "preprocess.json", destination / "preprocess.json")
    for optional in ("calibrator.json", "train_config.yaml", "training_summary.json"):
        if (source / optional).exists():
            shutil.copyfile(source / optional, destination / optional)
    record = {"format": "MLX FP16", "tensor_count": len(converted),
              "reference_weight_sha256": config["reference_weight_sha256"],
              "mlx_weight_sha256": sha256_file(destination / "model.safetensors"),
              "strict_parameter_load": True}
    (destination / "conversion.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(export_model(args.model, args.output), indent=2))


if __name__ == "__main__":
    main()
