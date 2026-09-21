"""Compile/use the pinned native projection before teacher input budgeting.

Run as JSONL CLI or import project_context(context). This adapter reads no apps,
clipboard, model predictions, or labels. Compilation is cached under local/.
"""
from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/context_projection"


@functools.lru_cache(maxsize=1)
def executable() -> Path:
    sources = [SOURCE / name for name in ("Models.swift", "RecommendationContext.swift", "ProjectSyntheticContext.swift")]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()
    directory = ROOT / "local/native-projection" / digest
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "project-context"
    with (directory / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.is_file():
            temporary = directory / f".project-context-{os.getpid()}"
            subprocess.run(["swiftc", "-swift-version", "6", "-warnings-as-errors", "-parse-as-library",
                            *map(str, sources), "-o", str(temporary)], check=True, capture_output=True)
            temporary.replace(binary)
    return binary


def project_context(context: dict) -> dict:
    allowed = {"applicationCategory", "inputSurface", "fieldRole", "fieldLabel", "selectedText",
               "surroundingText", "hasAccessibility", "isSecure"}
    if set(context) - allowed:
        raise ValueError("Synthetic context contains fields outside the deployment contract")
    complete = {"applicationCategory": "unknown", "fieldRole": "", "fieldLabel": "",
                "selectedText": "", "surroundingText": "", "hasAccessibility": True, "isSecure": False,
                **context}
    process = subprocess.run([str(executable())], input=json.dumps(complete, ensure_ascii=False) + "\n",
                             capture_output=True, text=True)
    if process.returncode:
        # Do not forward a runtime error that might contain data from a caller.
        raise ValueError("Native synthetic context projection rejected the input")
    return json.loads(process.stdout)


def main():
    for line in sys.stdin:
        value = json.loads(line)
        print(json.dumps(project_context(value), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
