"""Post-implementation model invariants and GPU throughput; no held-out data.

These hand-authored engineering fixtures are not accuracy benchmarks and are
never included in Train, Dev, Calibration, or Test. The report records real checks.
"""

import argparse
import copy
import json
import math
import platform
import statistics
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import torch

from pastewhat_ranker.mlx_model import MLXRanker, collate_mlx
from pastewhat_ranker.model import PasteWhatRanker, collate, group_loss
from pastewhat_ranker.preprocess import Preprocessor


def engineering_episode(count=3):
    return {
        "id": "engineering-only", "family_id": "engineering-excluded-from-all-data",
        "context": {"applicationCategory": "terminal", "inputSurface": "shell_prompt", "fieldRole": "AXTextArea",
                    "fieldLabel": "Command", "selectedText": "", "surroundingText": "List only local branches.",
                    "hasAccessibility": True, "isSecure": False},
        "entries": [{"id": f"candidate-{i}", "text": ("git branch" if i == 0 else f"git branch --list topic-{i}"),
                     "kind": "command", "capabilities": ["text"], "sourceCategory": "development"}
                    for i in range(count)],
        "label": {"decision": "select", "acceptable_ids": ["candidate-0"], "abstain_reason": None},
    }


def output(model, batch):
    return model(batch["input_ids"], batch["attention_mask"], batch["candidate_counts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="checkpoints/initial")
    parser.add_argument("--output", default="reports/training/preflight.json")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(8)
    device = "mps"
    model = PasteWhatRanker.from_pretrained(args.model, device=device).eval()
    p = Preprocessor(Path(args.model) / "tokenizer")
    episode = engineering_episode()
    checks, tolerances = {}, {"torch_padding": 2e-5, "mlx_fp16_absolute": 0.05}
    with torch.inference_mode():
        baseline = output(model, collate([episode], p, device)).cpu()
        permuted = copy.deepcopy(episode)
        permuted["entries"] = list(reversed(episode["entries"]))
        actual = output(model, collate([permuted], p, device)).cpu()
        checks["candidate_permutation_max_error"] = (actual[:, :-1].flip(-1) - baseline[:, :-1]).abs().max().item()
        checks["abstain_permutation_error"] = (actual[:, -1] - baseline[:, -1]).abs().max().item()
        padded = output(model, collate([episode], p, device, extra_padding=13)).cpu()
        checks["padding_max_error"] = (padded - baseline).abs().max().item()
        batched = output(model, collate([engineering_episode(1), episode], p, device)).cpu()
        checks["candidate_padding_excluded"] = bool(torch.isneginf(batched[0, 1:-1]).all())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, Path(args.model) / "tokenizer")
            loaded = PasteWhatRanker.from_pretrained(directory, device=device).eval()
            saved_output = output(loaded, collate([episode], p, device)).cpu()
            checks["save_reload_max_error"] = (saved_output - baseline).abs().max().item()
            del loaded
    simple = torch.tensor([[math.log(.2), math.log(.3), math.log(.5)]])
    positive = torch.tensor([[True, True, False]])
    checks["multi_positive_loss_error"] = abs(group_loss(simple, positive).item() + math.log(.5))
    renamed = copy.deepcopy(episode)
    renamed["id"], renamed["family_id"] = "a-different-id", "a-different-family"
    for entry in renamed["entries"]:
        entry["id"] += "-renamed"
    renamed["label"]["acceptable_ids"] = [renamed["entries"][0]["id"]]
    checks["ids_and_labels_excluded"] = p.encode_episode(episode)["input_ids"] == p.encode_episode(renamed)["input_ids"]
    long_episode = engineering_episode(20)
    long_episode["context"]["surroundingText"] = "范围与否定 evidence. " * 900
    for entry in long_episode["entries"]:
        entry["text"] = "candidate body with multilingual 文本 " * 700
    prepared = p.prepare_episode(long_episode)
    checks["preparation_idempotent"] = prepared == p.prepare_episode(prepared)
    checks["max_pair_tokens"] = max(map(len, p.encode_episode(prepared)["input_ids"]))
    # All candidate counts run through the actual deployment conversion before training.
    mlx_model = MLXRanker.from_pretrained(Path(args.model) / "mlx")
    parity = []
    for count in (1, 5, 10, 20):
        fixture = engineering_episode(count)
        with torch.inference_mode():
            ref = output(model, collate([fixture], p, device)).cpu()[0].tolist()
        b = collate_mlx([fixture], p)
        result = output(mlx_model, b)
        mx.eval(result)
        values = result[0].tolist()
        parity.append({"candidate_count": count, "max_absolute_error": max(abs(a - b) for a, b in zip(ref, values)),
                       "torch_top_action": max(range(len(ref)), key=ref.__getitem__),
                       "mlx_top_action": max(range(len(values)), key=values.__getitem__)})
    del mlx_model
    mx.clear_cache()
    model.train()
    model.encoder.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, foreach=False)
    abstain_episode = engineering_episode(3)
    abstain_episode["label"] = {"decision": "abstain", "acceptable_ids": [], "abstain_reason": "no_match"}
    batch = collate([abstain_episode], p, device)
    timings = []
    for step in range(6):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=torch.bfloat16):
            logits = output(model, batch)
            loss = group_loss(logits, batch["positive_mask"])
        loss.backward()
        if step == 0:
            checks["encoder_grad_norm"] = model.encoder.layers[0].attn.Wqkv.weight.grad.norm().item()
            checks["abstain_grad_norm"] = model.abstain_head[0].weight.grad.norm().item()
            checks["rank_head_grad_norm"] = model.candidate_score_head[0].weight.grad.norm().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.mps.synchronize()
        timings.append(time.perf_counter() - start)
        print(json.dumps({"step": step, "seconds": timings[-1], "loss": loss.item()}), flush=True)
    checks["finite_loss"] = math.isfinite(loss.item())
    passed = (checks["candidate_permutation_max_error"] < tolerances["torch_padding"]
              and checks["abstain_permutation_error"] < tolerances["torch_padding"]
              and checks["padding_max_error"] < tolerances["torch_padding"]
              and checks["save_reload_max_error"] == 0 and checks["multi_positive_loss_error"] < 1e-6
              and checks["candidate_padding_excluded"] and checks["ids_and_labels_excluded"]
              and checks["preparation_idempotent"] and checks["max_pair_tokens"] <= 1024
              and all(checks[name] > 0 for name in ("encoder_grad_norm", "abstain_grad_norm", "rank_head_grad_norm"))
              and all(row["max_absolute_error"] < tolerances["mlx_fp16_absolute"] for row in parity)
              and checks["finite_loss"])
    report = {"status": "passed" if passed else "failed", "checks": checks,
              "tolerances": tolerances, "mlx_fp16_parity": parity,
              "timing": {"candidate_count": 3, "pair_tokens": batch["input_ids"].shape[1],
                         "all_step_seconds": timings, "warm_median_step_seconds": statistics.median(timings[2:]),
                         "includes_optimizer_update_every_episode": True,
                         "training_dtype": "MPS BF16 autocast with FP32 master weights"},
              "software": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform()},
              "not_accuracy_benchmark": True}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
