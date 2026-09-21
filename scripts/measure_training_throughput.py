"""Measure equivalent episode batches on Train only before fixing execution shape."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from pastewhat_ranker.model import PasteWhatRanker, collate_encoded, group_loss
from pastewhat_ranker.preprocess import Preprocessor
from pastewhat_ranker.train import read_allowed_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--model", default="checkpoints/initial")
    parser.add_argument("--output", default="reports/training/throughput.json")
    args = parser.parse_args()
    torch.set_num_threads(8)
    p = Preprocessor(Path(args.model) / "tokenizer")
    episodes = read_allowed_data(args.train, expected_split="train")[:32]
    if len(episodes) < 16:
        raise ValueError("Need at least sixteen training episodes")
    encoded = [p.encode_episode(episode) for episode in episodes]
    report = []
    for micro in (1, 2, 4):
        torch.manual_seed(42)
        model = PasteWhatRanker.from_pretrained(args.model, device="mps").train()
        model.encoder.gradient_checkpointing = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, foreach=False)
        timings = []
        for iteration in range(3):
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, 16, micro):
                batch = collate_encoded(episodes[offset:offset + micro], encoded[offset:offset + micro], p.pad_id, "mps")
                with torch.autocast("mps", dtype=torch.bfloat16):
                    logits = model(batch["input_ids"], batch["attention_mask"], batch["candidate_counts"])
                    loss = group_loss(logits, batch["positive_mask"]) * micro / 16
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite gradients during throughput check")
            optimizer.step()
            torch.mps.synchronize()
            timings.append(time.monotonic() - start)
            print(json.dumps({"micro_batch": micro, "iteration": iteration, "seconds": timings[-1]}), flush=True)
        report.append({"micro_batch_episodes": micro, "effective_batch_episodes": 16,
                       "seconds_per_update": timings, "warm_seconds_per_episode": statistics.mean(timings[1:]) / 16,
                       "mps_allocated_bytes": torch.mps.current_allocated_memory(),
                       "mps_driver_bytes": torch.mps.driver_allocated_memory()})
        del optimizer, model
        torch.mps.empty_cache()
    result = {"training_data": args.train, "candidate_counts": [len(e["entries"]) for e in episodes[:16]],
              "pair_token_lengths": [list(map(len, e["input_ids"])) for e in encoded[:16]],
              "precision": "BF16 autocast, FP32 parameters/Adam moments", "gradient_checkpointing": True,
              "trials": report, "selected_micro_batch": min(report, key=lambda row: row["warm_seconds_per_episode"])["micro_batch_episodes"]}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
