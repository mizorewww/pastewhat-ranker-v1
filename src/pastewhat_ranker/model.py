"""Cross-encoder candidate scores and a permutation-invariant group abstain head."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .encoder import Encoder
from .preprocess import Preprocessor

SOURCE_REPO = "convaiinnovations/laya-multilingual"
SOURCE_REVISION = "052592a15d198d9ad47da779604259b10b47b7aa"
SOURCE_WEIGHT_SHA256 = "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PasteWhatRanker(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config["encoder"]
        d, h = cfg["hidden_size"], config.get("head_hidden", 256)
        dropout = config.get("head_dropout", 0.1)
        self.encoder = Encoder(cfg, config.get("max_pair_tokens", 1024))
        self.candidate_score_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 1))
        self.abstain_head = nn.Sequential(nn.Linear(2 * d + 1, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, input_ids, attention_mask, candidate_counts):
        if not candidate_counts or min(candidate_counts) < 1 or max(candidate_counts) > 20:
            raise ValueError("Model forward requires 1–20 candidates per episode")
        if sum(candidate_counts) != input_ids.shape[0]:
            raise ValueError("Candidate counts must match flattened text pairs")
        representations = self.encoder(input_ids, attention_mask.bool())[:, 0]
        candidate = self.candidate_score_head(representations).squeeze(-1).float()
        padded_width = max(candidate_counts)
        logits, offset = [], 0
        for count in candidate_counts:
            hidden = representations[offset:offset + count]
            # Accumulate pooling in FP32 to reduce precision-dependent abstention.
            hidden32 = hidden.float()
            pooled = torch.cat((hidden32.mean(0), hidden32.amax(0),
                                hidden32.new_tensor([float(count)]).log1p()), dim=0)
            abstain = self.abstain_head(pooled).reshape(1).float()
            pad = candidate.new_full((padded_width - count,), -torch.inf)
            logits.append(torch.cat((candidate[offset:offset + count], pad, abstain)))
            offset += count
        return torch.stack(logits)

    @classmethod
    def initialize(cls, source, seed=42):
        source = Path(source)
        fingerprint = sha256_file(source / "model.safetensors")
        if fingerprint != SOURCE_WEIGHT_SHA256:
            raise ValueError("Initialization weights do not match the frozen upstream revision")
        encoder = json.loads((source / "encoder/config.json").read_text())
        config = {"architecture": "PasteWhatRanker", "format_version": 1, "encoder": encoder,
                  "head_hidden": 256, "head_dropout": 0.1, "max_pair_tokens": 1024,
                  "source_repo": SOURCE_REPO, "source_revision": SOURCE_REVISION,
                  "source_weight_sha256": fingerprint, "initialization_seed": seed}
        torch.manual_seed(seed)
        model = cls(config)
        weights = load_file(str(source / "model.safetensors"))
        encoder_weights = {name.removeprefix("encoder."): value for name, value in weights.items() if name.startswith("encoder.")}
        removed = sorted(name for name in weights if not name.startswith("encoder."))
        allowed = ("head.", "type_emb.", "scorer.", "act_head.", "temperature")
        if any(not name.startswith(allowed) for name in removed):
            raise ValueError("Unknown upstream parameters were encountered")
        # strict=True is essential: no missing/unexpected encoder keys are allowed.
        model.encoder.load_state_dict(encoder_weights, strict=True)
        config["initialization_audit"] = {
            "encoder_tensors_loaded": len(encoder_weights), "missing_encoder_keys": [],
            "unexpected_encoder_keys": [], "discarded_parameters": removed,
            "source_dtypes": sorted({str(value.dtype) for value in weights.values()}),
            "training_parameter_dtype": "torch.float32",
            "encoder_parameter_count": sum(p.numel() for p in model.encoder.parameters()),
            "head_parameter_count": sum(p.numel() for name, p in model.named_parameters() if not name.startswith("encoder.")),
        }
        return model

    def save_pretrained(self, destination, tokenizer_source):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        weights = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}
        save_file(weights, str(destination / "model.safetensors"))
        (destination / "config.json").write_text(json.dumps(self.config, indent=2) + "\n")
        source = Path(tokenizer_source)
        if source.is_file():
            source = source.parent
        target = destination / "tokenizer"
        if source.resolve() != target.resolve():
            shutil.copytree(source, target, dirs_exist_ok=True)
        preprocess = Preprocessor(target)
        (destination / "preprocess.json").write_text(json.dumps(preprocess.manifest(), indent=2) + "\n")

    @classmethod
    def from_pretrained(cls, path, device="cpu"):
        path = Path(path)
        model = cls(json.loads((path / "config.json").read_text()))
        model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
        return model.to(device)


def collate(episodes, preprocessor, device="cpu", extra_padding=0):
    encoded = [preprocessor.encode_episode(episode) for episode in episodes]
    return collate_encoded(episodes, encoded, preprocessor.pad_id, device, extra_padding)


def collate_encoded(episodes, encoded, pad_id, device="cpu", extra_padding=0):
    sequences = [sequence for item in encoded for sequence in item["input_ids"]]
    if not sequences or any(not item["input_ids"] for item in encoded):
        raise ValueError("Empty episodes bypass model inference")
    length = max(map(len, sequences)) + extra_padding
    if length > 1024:
        raise ValueError("Padding exceeds maximum sequence length")
    ids = torch.full((len(sequences), length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = torch.tensor(sequence)
        mask[index, :len(sequence)] = True
    counts = [len(item["candidate_ids"]) for item in encoded]
    positives = torch.zeros((len(episodes), max(counts) + 1), dtype=torch.bool)
    for index, (episode, item) in enumerate(zip(episodes, encoded)):
        if "label" not in episode:
            continue
        label = episode["label"]
        if label["decision"] == "abstain":
            if label.get("acceptable_ids"):
                raise ValueError("Abstain labels must not contain acceptable candidates")
            positives[index, -1] = True
        elif label["decision"] == "select":
            good = set(label["acceptable_ids"])
            if not good or good - set(item["candidate_ids"]):
                raise ValueError("Select label contains missing candidate IDs")
            for position, identifier in enumerate(item["candidate_ids"]):
                positives[index, position] = identifier in good
        else:
            raise ValueError("Unknown decision label")
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "candidate_counts": counts, "positive_mask": positives.to(device),
            "candidate_ids": [item["candidate_ids"] for item in encoded]}


def group_loss(logits, positive_mask):
    """Mean episode loss: logsumexp(all valid actions) - logsumexp(acceptable)."""
    if logits.shape != positive_mask.shape or not bool(positive_mask.any(dim=-1).all()):
        raise ValueError("Each episode needs at least one acceptable action")
    if not bool(torch.isfinite(logits.masked_select(positive_mask)).all()):
        raise ValueError("A positive action points to a padding candidate")
    good = logits.float().masked_fill(~positive_mask, -torch.inf)
    return (torch.logsumexp(logits.float(), dim=-1) - torch.logsumexp(good, dim=-1)).mean()
