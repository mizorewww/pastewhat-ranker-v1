"""Registered local scheduling, independent of teacher inputs and cache identity."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
import json
from pathlib import Path
import time

from data_tools.teacher import sha256

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_PATH = ROOT / "configs/resource_supplement_swe2.json"
TRANSITION_PATH = ROOT / "configs/teacher_transition_swe2.json"


def load_resources(plan=None):
    if not RESOURCE_PATH.is_file():
        return None
    content = RESOURCE_PATH.read_bytes()
    document = json.loads(content)
    expected = {
        "version": "pastewhat-resource-supplement-v1", "provider": "devin", "model": "swe-2",
        "global_max_in_flight": 12, "fallback_max_in_flight": 6,
        "initial_workers": {"train": 6, "dev": 2, "calibration": 2, "test": 2},
        "executor_max_workers": 12, "borrow_slots_per_finished_split": 2,
        "borrow_when_frozen": ["dev", "calibration", "test"],
        "hard_pool_requires_frozen": "train", "on_http_429": "sticky_cap_6_and_bounded_backoff",
        "cache_identity_effect": "none", "teacher_transition_sha256": sha256(TRANSITION_PATH.read_bytes()),
    }
    if plan is not None:
        expected.update(plan.binding())
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError("Resource supplement differs from the registered operational contract")
    return document, {"path": str(RESOURCE_PATH.relative_to(ROOT)), "sha256": sha256(content)}


def activate_resources(coordinator, plan, *, drain_token):
    """One explicit zero-lease activation; retries cannot undo a sticky fallback."""
    loaded = load_resources(plan)
    if loaded is None:
        raise ValueError("No registered resource supplement to activate")
    document, binding = loaded
    with coordinator._state() as state:
        previous = state.get("resource_activation")
        if previous:
            if previous["resource_supplement"] != binding:
                raise ValueError("A different resource supplement was already activated")
            return {**previous, "already_activated": True, "effective_max_in_flight": state["max_in_flight"]}
        drain = state.get("operator_drain", {})
        if drain.get("token") != drain_token or state.get("pause_reason") != "operator_scheduler_drain" or state.get("cooldown_until") != drain.get("deadline"):
            raise ValueError("Capacity activation requires the owned live maintenance drain")
        if state["leases"] or state.get("blocked_reason"):
            raise ValueError("Capacity activation requires zero leases and no authentication block")
        previous_cap = state["max_in_flight"]
        chosen = document["fallback_max_in_flight"] if state.get("resource_concurrency_reduction") else document["global_max_in_flight"]
        state["max_in_flight"] = chosen
        record = {"resource_supplement": binding, "at": time.time(), "previous_max_in_flight": previous_cap, "activated_max_in_flight": chosen, "operator_drain_token": drain_token, "provider_cooldown_preserved": True}
        state["resource_activation"] = record
        state.setdefault("resource_activation_history", []).append(record)
        return record


def worker_budget(plan, split, ceiling, *, hard_pool=False):
    """Only inspect frozen readiness existence, never another owner's examples."""
    loaded = load_resources(plan)
    if loaded is None:
        return max(1, ceiling)
    document, _ = loaded
    if hard_pool and not (ROOT / plan.data_path(document["hard_pool_requires_frozen"])).is_file():
        return 0
    desired = document["initial_workers"][split]
    if split == "train":
        desired += document["borrow_slots_per_finished_split"] * sum((ROOT / plan.data_path(stage)).is_file() for stage in document["borrow_when_frozen"])
    return min(ceiling, document["executor_max_workers"], desired)


def bounded_futures(executor, items, submit, capacity):
    """Submit only the current allowance and discover newly free split slots.

    Capacity reductions drain already submitted work naturally. Task order and
    payloads stay intact; no mass queue can preclaim later source slots.
    """
    pending = iter(items)
    active, exhausted = set(), False
    try:
        while active or not exhausted:
            limit = max(0, capacity())
            while not exhausted and len(active) < limit:
                try:
                    item = next(pending)
                except StopIteration:
                    exhausted = True
                    break
                active.add(submit(executor, item))
            if not active:
                if not exhausted:
                    time.sleep(5)
                continue
            finished, _ = wait(active, timeout=5, return_when=FIRST_COMPLETED)
            for future in finished:
                active.remove(future)
                yield future
    finally:
        # Running requests are never killed; the executor context waits for them.
        for future in active:
            future.cancel()
