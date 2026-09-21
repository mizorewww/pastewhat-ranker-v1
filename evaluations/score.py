"""Run a persistent worker on prepared inputs, preserving every failed outcome."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import selectors
import subprocess
import time

from evaluations.common import inference_request, load_jsonl, score_features, sha256, validate_formal_heldout_allocation, write_json
from evaluations.freeze import directory_hashes, verify_freeze


class Worker:
    def __init__(self, command: list[str], log_path: Path, timeout: float):
        self.log = log_path.open("x")
        self.timeout = timeout
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false"}
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=environment, text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def request(self, value: dict) -> dict:
        start = time.perf_counter()
        try:
            if self.process.poll() is not None:
                raise RuntimeError("worker_exited")
            self.process.stdin.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
            self.process.stdin.flush()
            if not self.selector.select(self.timeout):
                self.process.kill()
                raise TimeoutError("worker_response_timeout")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("worker_closed_output")
            response = json.loads(line)
            if response.get("id") != value["id"]:
                raise ValueError("worker_response_id_mismatch")
            response["roundTripMS"] = (time.perf_counter() - start) * 1000
            return response
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            return {"id": value["id"], "error": type(error).__name__ + ":" + str(error),
                    "decision": "inference_failure", "recommendedID": None,
                    "roundTripMS": (time.perf_counter() - start) * 1000}

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.selector.close()
        self.log.close()


def warmup_request() -> dict:
    return {"id": "unscored-warmup", "context": {
        "applicationCategory": "writing", "inputSurface": "text", "fieldRole": "AXTextField",
        "fieldLabel": "Name", "selectedText": "", "surroundingText": "Enter the synthetic project name Example Project.",
        "hasAccessibility": True, "isSecure": False}, "entries": [
            {"id": "warm-a", "text": "Example Project", "kind": "text", "capabilities": ["text"], "sourceCategory": "writing"},
            {"id": "warm-b", "text": "Another Project", "kind": "text", "capabilities": ["text"], "sourceCategory": "writing"},
        ]}


def normalize_response(episode: dict, response: dict, protocol: str) -> dict:
    if response.get("error"):
        return response
    if protocol == "ranker":
        if not episode["context"].get("isSecure") and episode["entries"]:
            try:
                _, raw_top, _ = score_features(episode, response)
                response["rawTopID"] = raw_top
            except (KeyError, TypeError, ValueError) as error:
                return {**response, "error": "invalid_score_response:" + str(error), "decision": "inference_failure", "recommendedID": None}
    else:
        response["latencyMS"] = response.get("elapsedMS")
        ranked = response.get("rankings", [])
        if ranked:
            response["rawTopID"] = ranked[0].get("id")
        if response.get("decision") == "invalid_request":
            response["error"] = "baseline_invalid_request"
        if protocol == "jev" and response.get("decision") == "remote_unavailable":
            response["error"] = "jev_remote_unavailable"
        # In the pinned production worker, decide() may replace the fallback
        # error message with a normal ranking explanation. An attempted Laya
        # call followed by fallback is still an inference failure.
        if protocol == "baseline" and response.get("mode") == "fallback" and response.get("inferenceCount", 0) > 0:
            response["error"] = "baseline_model_failure"
        # A production deterministic/no-context response may be 'fallback'. A
        # failed attempted model call is identified by its actual failure message.
        if response.get("mode") == "fallback" and response.get("message") and any(
                word in response["message"] for word in ("Laya", "模型", "推理未完成", "本次推荐未完成")):
            response["error"] = "baseline_model_failure"
    return response


def command_artifacts(command: list[str], protocol: str, frozen: dict | None) -> dict:
    def option(name: str, default=None):
        if name not in command:
            return default
        if command.count(name) != 1 or command.index(name) + 1 >= len(command):
            raise ValueError("Ambiguous worker command option: " + name)
        return command[command.index(name) + 1]

    if "--no-model" in command:
        raise ValueError("Acceptance inference requires the actual model")
    backend = option("--backend", "mlx")
    result = {"backend": backend}
    if protocol in {"ranker", "baseline"}:
        model = option("--model")
        if model is None or backend != "mlx":
            raise ValueError("Calibration and acceptance require an explicit final MLX model directory")
        root = Path(model).expanduser().resolve()
        result["model_root"] = str(root)
        result["model_files"] = directory_hashes(root)
        if frozen:
            expected = "deployment" if protocol == "ranker" else "baseline_model"
            if root != Path(frozen[expected]["root"]).resolve():
                raise ValueError("Worker model path differs from the frozen model artifact")
        if protocol == "ranker":
            if "pastewhat_ranker.worker" not in command:
                raise ValueError("Use the versioned ranker worker for ranker acceptance")
            result.update(weights_sha256=sha256(root / "model.safetensors"),
                          preprocess_sha256=sha256(root / "preprocess.json"),
                          runtime_files=directory_hashes(Path(__file__).resolve().parents[1] / "src/pastewhat_ranker"))
    elif backend != "jev":
        raise ValueError("Jev comparison must explicitly select the Jev backend")
    if frozen and protocol in {"baseline", "jev"}:
        source = Path(frozen["baseline" if protocol == "baseline" else "jev"]["root"])
        if not any(Path(part).expanduser().resolve() == source / "worker.py" for part in command):
            raise ValueError("Worker command differs from the frozen comparator source")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "test", "regression"), required=True)
    parser.add_argument("--command-json", required=True, help="JSON array of command arguments; no shell expansion")
    parser.add_argument("--protocol", choices=("ranker", "baseline", "jev"), default="ranker")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an inference run")
    frozen = None
    if args.split == "test":
        if args.freeze is None:
            raise SystemExit("Final Test requires the parent-approved freeze manifest")
        frozen = verify_freeze(args.freeze, args.data)
    episodes = load_jsonl(args.data)
    if args.split != "regression" and any(row.get("split") != args.split for row in episodes):
        raise SystemExit("Dataset split disagrees with requested inference phase")
    if args.split != "regression":
        partition = Path(frozen["inputs"]["family_partition"]["path"]) if frozen else Path("data_tools/family_partition.json")
        validate_formal_heldout_allocation(episodes, args.split, json.loads(partition.read_text()))
    command = json.loads(args.command_json)
    if not isinstance(command, list) or not command or any(not isinstance(item, str) for item in command):
        raise SystemExit("Expected a JSON array of command arguments")
    artifacts = command_artifacts(command, args.protocol, frozen)
    args.output.mkdir(parents=True)
    manifest = {"split": args.split, "data_sha256": sha256(args.data), "episodes": len(episodes),
                "command": command, "protocol": args.protocol, "artifacts": artifacts, "platform": platform.platform(),
                "started_at": datetime.now(timezone.utc).isoformat(), "score_code_sha256": sha256(__file__),
                "freeze_sha256": sha256(args.freeze) if args.freeze else None,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    write_json(args.output / "run-manifest.json", manifest)
    start = time.perf_counter()
    worker = Worker(command, args.output / "worker.stderr.log", args.timeout)
    failures = 0
    try:
        warmup = worker.request(warmup_request())
        warmup["process_start_to_warmup_ms"] = (time.perf_counter() - start) * 1000
        write_json(args.output / "warmup.json", warmup)
        if warmup.get("error"):
            raise RuntimeError("Unscored warmup failed; no dataset results produced")
        if args.protocol == "ranker" and warmup.get("runtime") != "mlx":
            raise RuntimeError("Ranker acceptance must use actual MLX inference")
        if args.protocol == "baseline" and warmup.get("mode") != "laya":
            raise RuntimeError("Baseline must demonstrate actual Laya model inference during warmup")
        if args.protocol == "jev" and (warmup.get("mode") != "jev" or not warmup.get("modelVersion")):
            raise RuntimeError("Jev warmup must demonstrate an actual remote response with model version")
        with (args.output / "scores.jsonl").open("x") as handle:
            for index, episode in enumerate(episodes):
                request = inference_request(episode)
                response = normalize_response(episode, worker.request(request), args.protocol)
                failures += bool(response.get("error"))
                handle.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                if (index + 1) % 100 == 0 or index + 1 == len(episodes):
                    print(json.dumps({"phase": args.split, "scored": index + 1, "total": len(episodes), "failures": failures}), flush=True)
    finally:
        worker.close()
    if args.split == "test":
        verify_freeze(args.freeze, args.data)
    if sha256(args.data) != manifest["data_sha256"] or command_artifacts(command, args.protocol, frozen) != artifacts:
        raise RuntimeError("Dataset, model files or runtime changed during scoring; no completed run can be used")
    write_json(args.output / "completion.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
               "episodes": len(episodes), "failures": failures, "scores_sha256": sha256(args.output / "scores.jsonl")})


if __name__ == "__main__":
    main()
