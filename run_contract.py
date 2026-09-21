"""Pre-registered production sizes and bindings, separate from model code.

This module never opens examples or labels. A changed plan is a different run;
the original conceptual-family partition and engineering artifacts stay intact.
No smaller production plan is selected merely by importing this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "dev", "calibration", "test")
ORIGINAL_SUGGESTED_TARGETS = {"train": 20_000, "dev": 1_000, "calibration": 1_000, "test": 2_000}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def family_quotas(partition: dict, split: str, total: int) -> dict[str, int]:
    """Balanced counts, with deterministic remainder in partition order."""
    if split not in SPLITS:
        raise ValueError("Unknown data split")
    families = [row["id"] for row in partition["families"][split]]
    if not families or len(families) != len(set(families)):
        raise ValueError("The family partition must contain unique nonempty families")
    _positive(total, "split total")
    if total < len(families):
        raise ValueError("A registered split must cover every pre-existing family")
    quotient, remainder = divmod(total, len(families))
    return {family: quotient + (index < remainder) for index, family in enumerate(families)}


def action_quotas(counts: dict[str, int]) -> dict[str, dict[str, int]]:
    """Allocate exact global 70/20/10 buckets without modulo-prefix bias.

    Whole-episode rounding is half-up; largest fractional remainders receive the
    extra rows, with ties in frozen family order. Missing intent combines both
    ambiguous and insufficient-context cases.
    Candidate counts/language must still be sampled independently by the owner.
    """
    for name, count in counts.items():
        _positive(count, name)
    total = sum(counts.values())
    if not total:
        raise ValueError("Action allocation requires families")
    result = {name: {"select": 0, "no_match": 0, "missing_intent": count}
              for name, count in counts.items()}
    for action, numerator in (("select", 7), ("no_match", 2)):
        target = (total * numerator + 5) // 10
        for name, count in counts.items():
            assigned = min(count * numerator // 10, result[name]["missing_intent"])
            result[name][action] = assigned
            result[name]["missing_intent"] -= assigned
        remaining = target - sum(row[action] for row in result.values())
        order = sorted(counts, key=lambda name: -(counts[name] * numerator % 10))
        for name in order:
            if remaining and result[name]["missing_intent"]:
                result[name][action] += 1
                result[name]["missing_intent"] -= 1
                remaining -= 1
        if remaining:
            raise ValueError("The declared split is too small for its global action ratios")
    return result


@dataclass(frozen=True)
class RunPlan:
    path: Path
    sha256: str
    _document: dict

    @property
    def document(self) -> dict:
        return deepcopy(self._document)

    @property
    def run_id(self) -> str:
        return self._document["run_id"]

    def target(self, split: str) -> int:
        return self._document["split_targets"][split]

    def binding(self) -> dict:
        return {"run_id": self.run_id, "run_plan_sha256": self.sha256}

    def verify_unchanged(self) -> None:
        if digest(self.path) != self.sha256:
            raise ValueError("The registered plan changed while this run was active")

    def data_path(self, stage: str) -> Path:
        if stage in {"calibration", "test"}:
            return Path("local/evaluator-heldout") / self.run_id / f"{stage}.jsonl"
        names = {"pilot": "pilot-train.jsonl", "train": "train.jsonl", "dev": "dev.jsonl",
                 "hardening": "hardening-train.jsonl"}
        return Path("data/frozen") / self.run_id / names[stage]

    @property
    def pipeline_directory(self) -> Path:
        return Path("local/pipeline") / self.run_id

    @property
    def checkpoint_directory(self) -> Path:
        return Path("checkpoints") / self.run_id

    @property
    def report_directory(self) -> Path:
        return Path("reports/training") / self.run_id


def load_run_plan(path: str | Path, *, verify_sources: bool = True) -> RunPlan:
    path = Path(path).resolve()
    document = json.loads(path.read_text())
    required = {"version", "run_id", "split_targets", "pilot_episodes", "hardening", "training_seeds",
                "epochs", "head_warmup_steps", "effective_batch_episodes", "family_partition_sha256",
                "projection_provenance_sha256", "teacher_contract_version", "registration_reason",
                "registered_at", "cost_evidence", "quality_gates"}
    if not isinstance(document, dict) or set(document) != required:
        raise ValueError("Run plan fields differ from the registered schema")
    if document["version"] != "pastewhat-run-plan-v1" or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", document["run_id"]):
        raise ValueError("Invalid run plan version or identity")
    targets = document["split_targets"]
    if not isinstance(targets, dict) or set(targets) != set(SPLITS):
        raise ValueError("All four split targets must be explicit")
    for split, count in targets.items():
        _positive(count, split)
    if _positive(document["pilot_episodes"], "pilot episodes") > targets["train"]:
        raise ValueError("Pilot must fit inside the registered main Train set")
    hard = document["hardening"]
    if not isinstance(hard, dict) or set(hard) != {"pool_episodes", "review_nominations", "accepted_new", "retained_original"}:
        raise ValueError("Hard pool, nominations and accepted mixture are distinct quantities")
    for name, count in hard.items():
        _positive(count, name)
    if not hard["accepted_new"] <= hard["review_nominations"] <= hard["pool_episodes"] or hard["retained_original"] > targets["train"]:
        raise ValueError("Hardening quantities are inconsistent")
    if document["training_seeds"] != [42, 43, 44] or document["epochs"] != {"pilot": 2, "main": 2, "hardening": 1}:
        raise ValueError("The registered three-seed, staged training route must be retained")
    _positive(document["head_warmup_steps"], "head warmup steps")
    if document["effective_batch_episodes"] != 16:
        raise ValueError("Effective batch remains 16 episodes")
    gates = {"recommendation_precision": 0.95, "minimum_calibration_recommendations": 25,
             "top1_improvement": 0.05, "maximum_group_regression": 0.05,
             "minimum_answerable_per_test_family": 30, "minimum_dev_group_episodes": 20}
    if document["quality_gates"] != gates:
        raise ValueError("A size change cannot silently relax the registered quality gates")
    for key in ("teacher_contract_version", "registration_reason", "registered_at"):
        if not isinstance(document[key], str) or not document[key].strip():
            raise ValueError(f"{key} must be explicitly recorded")
    partition_path = ROOT / "data_tools/family_partition.json"
    projection_path = ROOT / "tools/context_projection/provenance.json"
    for key, source in (("family_partition_sha256", partition_path), ("projection_provenance_sha256", projection_path)):
        if not isinstance(document[key], str) or not re.fullmatch(r"[0-9a-f]{64}", document[key]):
            raise ValueError(f"{key} must be a SHA256")
        if verify_sources and digest(source) != document[key]:
            raise ValueError(f"Registered source changed: {key}")
    evidence = document["cost_evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Register the measured evidence behind the selected scale")
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("Cost evidence needs a relative report path and SHA256")
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("Invalid cost report binding")
        if verify_sources and digest(ROOT / relative) != item["sha256"]:
            raise ValueError("Registered cost evidence changed")
    if verify_sources:
        partition = json.loads(partition_path.read_text())
        for split, count in targets.items():
            action_quotas(family_quotas(partition, split, count))
        action_quotas(family_quotas(partition, "train", document["pilot_episodes"]))
    return RunPlan(path, digest(path), document)
