"""Label usable cached author drafts rejected only by count/opaque-ID plumbing.

No author API call is made. Earlier records and labels are never changed, and
drafts with an explicit semantic rejection are not eligible for this recovery.
"""
from __future__ import annotations

import argparse
import copy
import json
import time

from data_tools.content import ContentRegistry
from data_tools.generate_v7 import ROOT, publish_pool
from data_tools.teacher import TeacherClient, canonical_bytes, sha256
from data_tools.v7 import produce_batch
from pastewhat_ranker.preprocess import Preprocessor
from run_contract import load_run_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-plan", required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    args = parser.parse_args()
    plan = load_run_plan(args.run_plan)
    base = ROOT / "local/v7" / plan.run_id
    directory = base / "batches" / args.split
    records = [json.loads(path.read_text()) for path in directory.glob("*.json")]
    accepted = {row["id"] for record in records for row in record["accepted"]}
    client = TeacherClient(base / "teacher" / args.split)
    authors = {}
    for path in client.audit_dir.glob("*.json"):
        audit = json.loads(path.read_text())
        if audit.get("phase") == "v7-author" and audit.get("status") == "success":
            batch, attempt = audit["request_id"].rsplit("-a", 1)
            authors.setdefault(batch, []).append((int(attempt), client._result(audit, cache_hit=True)))
    preprocessor = Preprocessor(str(ROOT.parent / "laya-mlx/models/laya-multilingual/tokenizer"))
    registry = ContentRegistry(base / "content.sqlite3")
    started = time.monotonic()
    for record in records:
        if record["status"] != "complete" or "cached_author_source" in record["spec"]:
            continue
        semantic = {row["id"] for row in record["rejected"] if "original_label" in row}
        eligible = {row["id"] for row in record["rejected"] if row.get("id") and any(reason in row.get("reason", "") for reason in ("Candidate count differs", "at most180 characters"))}
        if any("Compact labels must cover" in row.get("reason", "") for row in record["rejected"]):
            eligible.update(item["id"] for item in record["spec"]["plans"])
        eligible -= accepted | semantic
        for _, author in sorted(authors.get(record["spec"]["batch_id"], []), reverse=True):
            drafts = author.parsed.get("episodes", [])
            available = {row.get("slot") for row in drafts if isinstance(row, dict) and isinstance(row.get("candidates"), list) and 1 <= len(row["candidates"]) <= 20}
            identifiers = eligible & available
            if not identifiers:
                continue
            spec = copy.deepcopy(record["spec"])
            spec["plans"] = [item for item in spec["plans"] if item["id"] in identifiers]
            spec["cached_author_source"] = {"original_batch_id": record["spec"]["batch_id"], "audit_id": author.audit_id, "reason": "Only count, obsolete180-character admission, or opaque-ID plumbing failed; no prior accepted/semantic-rejected label is reused"}
            spec["batch_id"] += "-cached-" + sha256(canonical_bytes(sorted(identifiers)))[:8]
            result = produce_batch(spec, client=client, preprocessor=preprocessor, destination=directory / (spec["batch_id"] + ".json"), claim=registry.claim, cached_author=author)
            accepted.update(row["id"] for row in result["accepted"])
            # Exactly one cached draft per source slot is tried; no resampling
            # until a favorable teacher label is obtained.
            eligible -= identifiers
            print(json.dumps(publish_pool(base, args.split, plan, started), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
