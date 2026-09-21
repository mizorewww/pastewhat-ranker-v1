"""Compile/use the pinned native projection before teacher input budgeting.

Run as JSONL CLI or import project_context(context, capture=...). This adapter reads no apps,
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
    sources = [SOURCE / name for name in ("Models.swift", "RecommendationContext.swift", "FocusText.swift", "ProjectSyntheticContext.swift")]
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


def project_context(context: dict, *, capture: dict | None = None) -> dict:
    allowed = {"applicationCategory", "inputSurface", "fieldRole", "fieldLabel", "selectedText",
               "surroundingText", "hasAccessibility", "isSecure"}
    if not isinstance(context, dict) or set(context) - allowed:
        raise ValueError("Synthetic context contains fields outside the deployment contract")
    complete = {"applicationCategory": "unknown", "fieldRole": "", "fieldLabel": "",
                "selectedText": "", "surroundingText": "", "hasAccessibility": True, "isSecure": False,
                **context}
    if capture is not None:
        capture_fields = {"textWindow", "selectionLocation", "selectionLength", "nearbyText"}
        if not isinstance(capture, dict) or set(capture) != capture_fields:
            raise ValueError("Capture must contain exactly the observable text and selection fields")
        if not isinstance(capture["textWindow"], str):
            raise ValueError("Capture textWindow must be a string")
        if (not isinstance(capture["nearbyText"], list)
                or any(not isinstance(value, str) for value in capture["nearbyText"])):
            raise ValueError("Capture nearbyText must be an array of strings")
        for key in ("selectionLocation", "selectionLength"):
            if capture[key] is not None and (type(capture[key]) is not int or capture[key] < 0):
                raise ValueError("Capture selection uses nonnegative UTF-16 indices or null")
        complete["capture"] = capture
    process = subprocess.run([str(executable())], input=json.dumps(complete, ensure_ascii=False) + "\n",
                             capture_output=True, text=True)
    if process.returncode:
        # Do not forward a runtime error that might contain data from a caller.
        raise ValueError("Native synthetic context projection rejected the input")
    return json.loads(process.stdout)


def main():
    for line in sys.stdin:
        value = json.loads(line)
        if "context" in value:
            if set(value) - {"context", "capture"}:
                raise ValueError("Projection wrapper accepts context and capture only")
            projected = project_context(value["context"], capture=value.get("capture"))
        else:
            projected = project_context(value)
        print(json.dumps(projected, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
