"""Pre-label variants of information actually available from AX capture.

This never changes candidate payloads or an existing decision label. Owners
register untouched author slots first and apply a variant before native capture,
student budgeting and both blind labels. No hidden view reaches any reviewer.
"""
from __future__ import annotations

import copy


OBSERVATION_PROTOCOL = "registered-observation-variants-v1"
VARIANTS = frozenset({"standard", "no_accessibility", "generic_field"})


def apply_observation_variant(raw_episode, variant):
    """Project a raw author fixture to one pre-registered observation state."""
    if variant not in VARIANTS:
        raise ValueError("Unregistered observation variant")
    if "label" in raw_episode or "provenance" in raw_episode or "preprocessing" in raw_episode:
        raise ValueError("Observation variants apply only before any labeling or budgeting")
    raw = copy.deepcopy(raw_episode)
    if variant == "standard":
        return raw
    context = raw["context"]
    if context.get("isSecure"):
        raise ValueError("Secure input is a deployment bypass, not an observation variant")
    context["selectedText"] = ""
    context["surroundingText"] = ""
    context["inputSurface"] = "unknown"
    if variant == "no_accessibility":
        context.update(hasAccessibility=False, fieldLabel="", fieldRole="")
        raw["capture"] = {"textWindow": "", "nearbyText": []}
    else:
        context["hasAccessibility"] = True
        raw["capture"] = {"beforeSelection": "", "afterSelection": "", "nearbyText": []}
    return raw
