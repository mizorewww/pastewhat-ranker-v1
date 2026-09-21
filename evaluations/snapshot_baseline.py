"""Copy an exact Git-revision comparator without changing the working app."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import subprocess
from pathlib import Path

from evaluations.common import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path("../pastewhat"))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--protocol", choices=("baseline", "jev"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite a pinned comparator snapshot")
    revision = subprocess.check_output(["git", "-C", str(args.repository), "rev-parse", args.revision + "^{commit}"], text=True).strip()
    names = ["worker.py", "ranking.py"] + (["jev.py"] if args.protocol == "jev" else [])
    payloads = {}
    for name in names:
        payloads[name] = subprocess.check_output(["git", "-C", str(args.repository), "show", revision + ":engine/" + name])
    args.output.mkdir(parents=True)
    for name, payload in payloads.items():
        (args.output / name).write_bytes(payload)
    record = {"revision": revision, "protocol": args.protocol,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "source_repository": "https://github.com/mizorewww/pastewhat",
              "files": {name: sha256(args.output / name) for name in names},
              "runtime_and_model": "Recorded separately by freeze.py and the actual scoring run"}
    write_json(args.output / "snapshot.json", record)
    print({"revision": revision, "protocol": args.protocol, "snapshot_manifest_sha256": sha256(args.output / "snapshot.json")})


if __name__ == "__main__":
    main()
