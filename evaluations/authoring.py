"""Heldout-owned profiles for the shared deterministic compact fixture builder."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from data_tools.authoring import AUTHORING_PROTOCOL, compile_compact_episode

PROFILE_PATH = Path(__file__).with_name("authoring-profiles.json")
BUILDER_PATH = Path(__file__).resolve().parents[1] / "data_tools/authoring.py"


def candidate_space(family_id: str) -> tuple[int, ...]:
    """Finite status vocabulary is declared before labels, never inferred from them."""
    return tuple(range(1, 5 if family_id == "calendar_rsvp_status" else 21))

# This variation was already assigned independently of labels in the sampling
# plan. It changes observable field hosting, never its requested operation.
ALTERNATE_CATEGORY = {
    "development": "browser", "spreadsheet": "development", "unknown": "browser",
    "browser": "writing", "writing": "browser", "mail": "writing", "creative": "writing",
}


def profile_for_spec(family_id: str, spec: dict) -> dict:
    profiles = json.loads(PROFILE_PATH.read_text())["profiles"]
    profile = copy.deepcopy(profiles[family_id])
    if spec.get("field_overrides_app_category"):
        profile["applicationCategory"] = ALTERNATE_CATEGORY[profile["applicationCategory"]]
    return profile


def compile_episode(item: dict, *, episode_id: str, family_id: str, spec: dict) -> dict:
    if type(item.get("slot")) is not int or item["slot"] != spec["slot"]:
        raise ValueError("Compact author slot differs from its predeclared plan")
    return compile_compact_episode(item, episode_id=episode_id,
                                   profile=profile_for_spec(family_id, spec),
                                   candidate_count=spec["candidate_count"])
