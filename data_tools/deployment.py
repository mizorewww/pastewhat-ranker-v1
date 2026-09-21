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
    "(capture.beforeSelection='', capture.afterSelection=''), or its entire "
    "current method token is selected. Bare method candidates are directly usable there. "
    "Do not use a code editor, ___, fabricated cursor markers, or unquoted JavaScript identifiers."
)

CODE_AUTHORING = (
    "For code-editor SELECT scenarios, author literal capture.beforeSelection/afterSelection "
    "and exact context.selectedText; production code calculates UTF-16 offsets. "
    "An empty code input with actual nearby static task guidance is simplest. "
    "Pasting literally yields beforeSelection+candidate+afterSelection. "
    "Do not show existing executable target code without selecting it and then propose "
    "its replacement. Do not invent ___, [cursor], <cursor>, or placeholder insertion positions. "
    "Constraints must say which extra behavior is forbidden when it distinguishes candidates; "
    "otherwise harmless broader behavior may also be acceptable."
)

PYTHON_AUTHORING = (
    "For Python, the visible literal paste result must be a complete compilable "
    "module/function, or the actual enclosing function must be visible in "
    "beforeSelection/afterSelection with exact indentation. A bare return, yield, "
    "break, continue or await cannot rely on an invisible enclosing function/loop. "
    "Use complete short function candidates in an empty editor when that is the "
    "simplest faithful scenario. Do not assume the editor auto-indents pasted text. "
)

GIT_DIFF_AUTHORING = (
    "For comparing commits, explicitly state the old/source revision and "
    "new/target revision and the required output (for example, the forward patch "
    "from old to new). Merely 'compare A with B' does not constrain direction. "
    "Do not infer a commit's parent from adjacent abbreviated git-log lines. "
    "Scope/output constraints must visibly distinguish git diff, reversed diff, "
    "git show, --stat or --name-only if they are different candidates. "
    "Nearby guidance must be actual static UI help, not terminal scrollback. "
)


def authoring_requirement(family_id):
    if family_id == "http_method":
        return HTTP_METHOD_AUTHORING
    extra = PYTHON_AUTHORING if family_id.startswith("python_") else GIT_DIFF_AUTHORING if family_id == "git_diff_selection" else ""
    return CODE_AUTHORING + extra


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
        if selected and ((before == selected and not after) or (after == selected and not before)):
            return "Synthetic whole-field replacement duplicates selectedText outside the replaced range"
        unselected = before + after if focus.get("selectionKnown") else focus.get("textWindow", "")
        if code_editor and re.search(r"_{3,}", unselected):
            return "Unselected code blank cannot be replaced by pasting an answer token"
        if label and label["decision"] == "select" and not focus.get("selectionKnown") and focus.get("textWindow"):
            return "Nonempty field has no observable caret or replacement range"
    elif surrounding and label and label["decision"] == "select":
        return "Select lacks a complete, observable production caret representation after budgeting"
    if label and label["decision"] == "select" and code_editor and episode.get("family_id", "").startswith("python_"):
        if focus and not focus.get("selectionKnown"):
            return "Python select requires an observed literal insertion range"
        before = focus.get("beforeSelection", "") if focus else ""
        after = focus.get("afterSelection", "") if focus else ""
        positives = set(label["acceptable_ids"])
        for entry in episode["entries"]:
            if entry["id"] not in positives:
                continue
            try:
                # Compile only: never execute generated clipboard code. These
                # synthetic cases intentionally include the complete short scope.
                compile(before + entry["text"] + after, "<synthetic-paste>", "exec")
            except (SyntaxError, ValueError) as exc:
                return f"Python positive is not syntactically usable at the visible paste location: {getattr(exc, 'msg', str(exc))}"
    if episode.get("family_id") == "http_method":
        if context.get("fieldLabel") != "HTTP method" or context.get("fieldRole") != "AXTextField":
            return "HTTP method synthetic tasks require a real standalone method text field"
        if selected and not re.fullmatch(r"[A-Za-z-]+", selected.strip()):
            return "HTTP method selection is not the complete field's current method token"
    return None
