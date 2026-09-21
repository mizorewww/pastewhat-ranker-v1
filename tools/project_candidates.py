"""Project synthetic payload fixtures through the production Swift codec.

Authoring metadata is not model evidence. Only the returned five-field native
candidates may be budgeted and shown to teachers or students. No real files,
clipboard contents, applications, or image assets are read by this adapter.
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
CATEGORIES = {"browser", "development", "terminal", "mail", "messaging", "writing",
              "spreadsheet", "creative", "file_management", "unknown"}


@functools.lru_cache(maxsize=1)
def executable() -> Path:
    sources = [SOURCE / name for name in ("Models.swift", "RecommendationContext.swift",
                                          "CandidateProjection.swift", "ProjectSyntheticCandidates.swift")]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()
    directory = ROOT / "local/native-candidate-projection" / digest
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "project-candidates"
    with (directory / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.is_file():
            temporary = directory / f".project-candidates-{os.getpid()}"
            subprocess.run(["swiftc", "-swift-version", "6", "-warnings-as-errors", "-parse-as-library",
                            *map(str, sources), "-o", str(temporary)], check=True, capture_output=True)
            temporary.replace(binary)
    return binary


def validate(entries: list[dict]) -> None:
    if not isinstance(entries, list) or not 1 <= len(entries) <= 20:
        raise ValueError("Synthetic payload projection requires 1–20 candidates")
    identifiers: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"id", "sourceCategory", "payload"}:
            raise ValueError("Authored candidates contain only id, sourceCategory and payload")
        identifier = entry["id"]
        if not isinstance(identifier, str) or not identifier or len(identifier) > 128 or identifier in identifiers:
            raise ValueError("Candidate IDs must be nonempty, bounded and unique")
        identifiers.add(identifier)
        if not isinstance(entry["sourceCategory"], str) or entry["sourceCategory"] not in CATEGORIES:
            raise ValueError("Candidate source must be an application category")
        payload = entry["payload"]
        if not isinstance(payload, dict):
            raise ValueError("Synthetic payload must be an object")
        if payload.get("type") == "text":
            if (set(payload) != {"type", "text"} or not isinstance(payload["text"], str)
                    or not payload["text"].strip() or len(payload["text"].encode("utf-8")) > 128_000):
                raise ValueError("Text fixtures require nonempty UTF-8 text of at most 128,000 bytes")
        elif payload.get("type") == "file":
            names = payload.get("names")
            if set(payload) != {"type", "names"} or not isinstance(names, list) or not 1 <= len(names) <= 20:
                raise ValueError("File fixtures require 1–20 synthetic basenames")
            if any(not isinstance(name, str) or not name or name in {".", ".."}
                   or len(name.encode("utf-8")) > 255 or any(char in name for char in "/\\\x00\n\r")
                   for name in names):
                raise ValueError("File fixture names must be bounded basenames, without paths or newlines")
            if len(set(names)) != len(names):
                raise ValueError("File fixtures require distinct synthetic files")
        elif payload.get("type") == "image":
            if (set(payload) != {"type", "width", "height"}
                    or any(type(payload.get(key)) is not int or not 1 <= payload[key] <= 8192
                           for key in ("width", "height"))
                    or payload["width"] * payload["height"] > 16_777_216):
                raise ValueError("Image fixtures require integer PNG dimensions within the bounded pixel budget")
        else:
            raise ValueError("Synthetic payload type must be text, file or image")


def project_candidates(entries: list[dict]) -> list[dict]:
    validate(entries)
    process = subprocess.run([str(executable())], input=json.dumps(entries, ensure_ascii=False) + "\n",
                             capture_output=True, text=True)
    if process.returncode:
        raise ValueError("Native synthetic candidate projection rejected the input")
    projected = json.loads(process.stdout)
    if [row["id"] for row in projected] != [row["id"] for row in entries]:
        raise ValueError("Native candidate projection changed the result mapping")
    return projected


def main():
    for line in sys.stdin:
        print(json.dumps(project_candidates(json.loads(line)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
