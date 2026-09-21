"""Conservative synthetic-data gates for observable paste placement.

These gates reject an authored episode; they never rewrite a context or label.
They are dataset-production restrictions, not a new feature passed to the model.
"""

from __future__ import annotations

import re
import functools
import json
from pathlib import Path

from data_tools.content import content_fingerprint

REVIEW_PATH = Path(__file__).resolve().parents[1] / "data/train_dev.placement_review.json"


@functools.lru_cache(maxsize=4)
def rejected_content(stamp):
    if not stamp:
        return {}
    return {item["content_sha256"]: item["reason"] for item in json.loads(REVIEW_PATH.read_text()).get("rejections", [])}


HTTP_METHOD_AUTHORING = (
    "This operation must use an actual standalone HTTP method text field: "
    "fieldLabel='HTTP method', fieldRole='AXTextField', applicationCategory='development'. "
    "Raw context.surroundingText stays empty. Put actual static request-editor help in "
    "capture.nearbyText, not source code or a fill-in-the-blank quiz. The method field is empty "
    "(capture.textWindow='', selectionLocation=0, selectionLength=0), or its entire "
    "current method token is selected. Bare method candidates are directly usable there. "
    "Do not use a code editor, ___, fabricated cursor markers, or unquoted JavaScript identifiers."
)

CODE_AUTHORING = (
    "For code-editor SELECT scenarios, provide a real capture selection/caret with exact "
    "UTF-16 offsets. An empty code input with actual nearby static task guidance is simplest. "
    "Pasting literally yields beforeSelection+candidate+afterSelection. "
    "Do not show existing executable target code without selecting it and then propose "
    "its replacement. Do not invent ___, [cursor], <cursor>, or placeholder insertion positions. "
    "Constraints must say which extra behavior is forbidden when it distinguishes candidates; "
    "otherwise harmless broader behavior may also be acceptable."
)


def authoring_requirement(family_id):
    return HTTP_METHOD_AUTHORING if family_id == "http_method" else CODE_AUTHORING


def placement_issue(episode, label=None):
    rejected = rejected_content(REVIEW_PATH.stat().st_mtime_ns if REVIEW_PATH.is_file() else 0)
    if content_fingerprint(episode) in rejected:
        return rejected[content_fingerprint(episode)]
    context = episode["context"]
    selected = context.get("selectedText", "")
    surrounding = context.get("surroundingText", "")
    code_editor = context.get("inputSurface") == "code_editor"
    if re.search(r"(?i)<cursor>|\[cursor\]|\[caret\]|<caret>|\|CURSOR\|", surrounding):
        return "Fabricated cursor marker is not a deployment-observable caret position"
    label = label or episode.get("label")
    focus = None
    if surrounding:
        try:
            parsed = json.loads(surrounding)
            if isinstance(parsed, dict) and parsed.get("format") == "pastewhat-focus-v1":
                focus = parsed
        except json.JSONDecodeError:
            pass
    if focus:
        before = focus.get("beforeSelection", "")
        after = focus.get("afterSelection", "")
        unselected = before + after if focus.get("selectionKnown") else focus.get("textWindow", "")
        if code_editor and re.search(r"_{3,}", unselected):
            return "Unselected code blank cannot be replaced by pasting an answer token"
        if label and label["decision"] == "select" and not focus.get("selectionKnown") and focus.get("textWindow"):
            return "Nonempty field has no observable caret or replacement range"
    elif surrounding and label and label["decision"] == "select":
        return "Select lacks a complete, observable production caret representation after budgeting"
    if episode.get("family_id") == "http_method":
        if context.get("fieldLabel") != "HTTP method" or context.get("fieldRole") != "AXTextField":
            return "HTTP method synthetic tasks require a real standalone method text field"
        if selected and not re.fullmatch(r"[A-Za-z-]+", selected.strip()):
            return "HTTP method selection is not the complete field's current method token"
    return None
