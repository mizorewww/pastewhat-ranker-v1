"""Standalone local deployment timing on synthetic inputs, never test labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time

from evaluations.common import percentile, sha256, write_json
from evaluations.score import Worker, normalize_response
from pastewhat_ranker.preprocess import Preprocessor


def make_request(count: int, preprocessor: Preprocessor, text_words: int) -> tuple[dict, list[int]]:
    context = {"applicationCategory": "writing", "inputSurface": "text", "fieldRole": "AXTextArea",
               "fieldLabel": "Text", "selectedText": "", "surroundingText": "Paste the project note that says the release is ready.",
               "hasAccessibility": True, "isSecure": False}
    entries = [{"id": f"bench-{index}", "kind": "text", "capabilities": ["text"], "sourceCategory": "writing",
                "text": f"Synthetic project {index}: " + "release note example context token " * max(1, text_words // 5)} for index in range(count)]
    prepared = preprocessor.prepare_episode({"id": "performance-request", "context": context, "entries": entries})
    lengths = [len(tokens) for tokens in preprocessor.encode_episode(prepared)["input_ids"]]
    return prepared, lengths


def process_rss_bytes(pid: int) -> int | None:
    try:
        # macOS ps reports resident memory in KiB. Unified GPU allocations may
        # also be resident; RSS is not a claim of isolated Metal allocation size.
        return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()) * 1024
    except (subprocess.CalledProcessError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-json", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--text-words", type=int, default=180)
    parser.add_argument("--gpu-exclusive-confirmation", required=True,
                        help="Record the orchestration message confirming training and other GPU inference are idle")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a performance run")
    args.output.mkdir(parents=True)
    command = json.loads(args.command_json)
    preprocessor = Preprocessor(args.tokenizer)
    results = []
    for count in (1, 5, 10, 20):
        request, lengths = make_request(count, preprocessor, args.text_words)
        started = time.perf_counter()
        worker = Worker(command, args.output / f"worker-{count}.stderr.log", 240)
        try:
            cold = normalize_response(request, worker.request(request), "ranker")
            cold_ms = (time.perf_counter() - started) * 1000
            if cold.get("error"):
                raise RuntimeError("Performance cold inference failed: " + cold["error"])
            latencies, process_latencies, memory = [], [], []
            for repeat in range(args.repeats):
                value = {**request, "id": f"performance-{count}-{repeat}"}
                response = normalize_response(value, worker.request(value), "ranker")
                if response.get("error"):
                    raise RuntimeError("Performance warm inference failed: " + response["error"])
                latencies.append(float(response["latencyMS"]))
                process_latencies.append(float(response["roundTripMS"]))
                rss = process_rss_bytes(worker.process.pid)
                if rss is not None:
                    memory.append(rss)
            results.append({"candidate_count": count, "encoded_pairs": count, "pair_token_lengths": lengths,
                            "cold_process_to_first_result_ms": cold_ms, "first_score_ms": cold.get("latencyMS"),
                            "warm_inference_ms_p50": statistics.median(latencies), "warm_inference_ms_p95": percentile(latencies, 95),
                            "warm_roundtrip_ms_p50": statistics.median(process_latencies), "warm_roundtrip_ms_p95": percentile(process_latencies, 95),
                            "observed_max_rss_bytes": max(memory) if memory else None,
                            "warm_inference_ms": latencies, "warm_roundtrip_ms": process_latencies,
                            "runtime": cold.get("runtime"), "repeats": args.repeats})
        finally:
            worker.close()
        print(json.dumps({key: value for key, value in results[-1].items() if key not in {"warm_inference_ms", "warm_roundtrip_ms", "pair_token_lengths"}}), flush=True)
    report = {"platform": platform.platform(), "machine": platform.machine(), "command": command,
              "preprocessing": preprocessor.manifest(), "deployment_manifest_sha256": sha256(args.deployment_manifest),
              "gpu_exclusive_confirmation": args.gpu_exclusive_confirmation, "results": results,
              "memory_measurement": "maximum post-inference macOS process RSS; not isolated GPU memory and not a continuous peak sampler",
              "child_peak_rss_platform_units": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              "scope": "worker inference only; excludes AppKit window, AX capture, and clipboard collection"}
    write_json(args.output / "performance.json", report)


if __name__ == "__main__":
    main()
