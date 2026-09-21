"""JSONL scoring protocol, without teacher labels or generated content.

Input: one episode per line. Output: original candidate IDs with scores, group
abstain score, latency and runtime. Secure/empty episodes bypass inference.
Calibration is deliberately a separate, frozen deployment policy.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

from .preprocess import Preprocessor


class RankerScorer:
    def __init__(self, model_path, backend="mlx", device="mps"):
        self.backend, self.device = backend, device
        self.preprocessor = Preprocessor(Path(model_path) / "tokenizer")
        if backend == "mlx":
            from .mlx_model import MLXRanker
            self.model = MLXRanker.from_pretrained(model_path)
        elif backend == "torch":
            from .model import PasteWhatRanker
            self.model = PasteWhatRanker.from_pretrained(model_path, device=device).eval()
        else:
            raise ValueError("Unknown ranker backend")

    def score(self, episode):
        start = time.perf_counter()
        prepared = self.preprocessor.prepare_episode(episode)
        if prepared["context"]["isSecure"] or not prepared["entries"]:
            return {"id": episode.get("id"), "candidateScores": [], "abstainScore": None,
                    "recommendedID": None, "latencyMS": 0, "runtime": self.backend,
                    "bypass": "secure_field" if prepared["context"]["isSecure"] else "empty_candidates", "error": None}
        if self.backend == "mlx":
            import mlx.core as mx
            from .mlx_model import collate_mlx
            batch = collate_mlx([prepared], self.preprocessor)
            logits = self.model(batch["input_ids"], batch["attention_mask"], batch["candidate_counts"])
            mx.eval(logits)
            scores = logits[0].tolist()
        else:
            import torch
            from .model import collate
            batch = collate([prepared], self.preprocessor, self.device)
            with torch.inference_mode():
                logits = self.model(batch["input_ids"], batch["attention_mask"], batch["candidate_counts"])
            scores = logits[0].cpu().tolist()
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("Nonfinite ranker scores")
        return {"id": episode.get("id"),
                "candidateScores": [{"id": entry["id"], "score": float(score)}
                                    for entry, score in zip(prepared["entries"], scores[:-1])],
                "abstainScore": float(scores[-1]), "latencyMS": (time.perf_counter() - start) * 1000,
                "runtime": self.backend, "error": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=("mlx", "torch"), default="mlx")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    scorer = RankerScorer(args.model, args.backend, args.device)
    for line in sys.stdin:
        episode = {}
        try:
            if len(line) > 2_000_000:
                raise ValueError("Episode request exceeds byte limit")
            episode = json.loads(line)
            response = scorer.score(episode)
        except Exception as error:
            response = {"id": episode.get("id") if isinstance(episode, dict) else None,
                        "candidateScores": [], "abstainScore": None, "latencyMS": None,
                        "runtime": args.backend, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
