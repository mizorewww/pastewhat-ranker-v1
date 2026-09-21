"""Independent heldout mother-task recipes for the efficient v7 author.

These are authoring instructions, not extra information for the student or its
blind labeler. All usable evidence must be rendered through the native fixture.
The immutable conceptual partition is unchanged; no old heldout row is reused.
"""
from __future__ import annotations

from evaluations.authoring import candidate_space, profile_for_spec


COMMON = """Create distinct tasks inside this one declared conceptual operation.
Use the compact slot/guidance/selected/candidates fixture. The literal candidate
must paste unchanged into the actual selected whole field. Put necessary facts,
execution environment, entity/range identifiers and output constraints in the
visible short guidance or selected text, not in an explanation or hidden story.
Vary task structure, meaningful contrasts and entities; preserve the plan's exact
candidate count, language, observation variant and action bucket.

For a select slot, the observed evidence must determine what the user wants.
Several acceptable candidates must be interchangeable ways to satisfy that SAME
established intention. Mutually exclusive personal preferences, possible future
actions, or merely valid field values are not interchangeable positive answers.
An empty input or one remaining candidate does not reveal a user's preference.
For no_match, establish the same kind of clear need but provide no usable match.
For missing intent, omit a necessary distinguishing fact and retain plausible
alternatives. Do not add an explicit 'unknown/undecided' cue merely to signal the
label. The registered no_accessibility/generic_field variants remove observation
before labeling; keep their candidate groups within this operation.

Do not create invisible-cursor fill-in-the-blank tasks. Candidate text is literal;
image/file text is a native summary, not a replacement for actual payload bytes.
Use exact source ranges and bounded inputs where correctness depends on them.
Names, URLs, paths and credentials are synthetic. Do not include teacher labels,
rationales or semantic candidate IDs in the authored fixture.
"""


RECIPES = {
    "archive_extract": "Archive create/extract tasks with explicit source archive, target directory, compression and selected members. Use only the visible stock macOS/bsdtar environment; distinguish option placement and literal paths. Do not switch to ordinary file-copy tasks.",
    "spreadsheet_distinct_count": "Count distinct values over an exactly stated cell range under the displayed formula engine. State blank, sentinel and empty-range behavior when those distinguish candidates. Contrast occurrence count, distinct count and frequency exactly once; do not turn this into lookup or SUM/AVERAGE.",
    "calendar_rsvp_status": "Record an already communicated attendee response in a calendar RSVP field. A select/no-match task needs a visible source response stating attendance, refusal or tentative attendance, tied to the event and attendee. Merely displaying an invitation, event time or list of valid RSVP options does not establish attendance. Missing-intent tasks lack that source response or distinguishing attendee/event. Synonymous status strings may be interchangeable only for the same established response; accepting and declining never form one acceptable set.",
    "url_fragment_anchor": "Navigate to a stated section within an explicitly identified document URL. Supply the actual section-to-fragment identifier in visible help when needed. Contrast fragment, query and whole-page destinations without pagination/cursor tasks.",
    "contact_international_dial": "Choose a telephone representation for a visible synthetic contact and explicit country/international dialing format. Supply country and dialing origin if required. Do not infer a contact identity from the app or a preferred code from candidate popularity.",
    "translation_formal_tone": "Translate the visible source meaning into a stated language and register for a stated addressee. Preserve requests, negation and politeness. Synonymous translations may be positive; incompatible formality or changed facts are not interchangeable. Keep tone/register as the central distinction.",
    "sql_group_having": "Write a whole SQLite query for a visible schema and explicit group aggregate condition. Name group keys, aggregate, boundary and desired output columns. Keep GROUP BY/HAVING versus row filtering central; no anti-joins or unrelated ordering tasks.",
    "chmod_octal_permission": "Set explicit owner/group/other permissions for a named synthetic path. State special bits, recursion and exact intended policy if relevant; distinguish each octal permission digit. Do not hide desired access policy in candidate order.",
    "git_restore_stash_scope": "Apply Git restore/stash operations to explicitly stated working-tree/index contents and exact affected paths. Supply the actual stash ref when the requested operation needs one; a human message is not automatically a ref. Specify whether index/worktree and stash retention may change.",
    "regex_anchored_search": "Choose a pattern body for the visible ECMAScript engine and an explicit whole-string, prefix, suffix or substring language. Specify whether empty input is allowed when quantifiers distinguish it. State allowed input characters and delimiters; equivalent accepted languages may be multi-positive.",
    "sql_antijoin_null": "Write a whole SQLite exclusion query from the visible schema, nullable columns and required treatment of NULL. State output columns and the relationship being excluded. Contrast NOT EXISTS, anti-join and NULL-sensitive NOT IN using that same task.",
    "url_pagination_cursor": "Continue a stated synthetic API/list response using a visible next cursor or page parameter, base endpoint and required filter state. Preserve opaque cursor bytes and explicit encoding rules; do not infer a missing continuation token.",
    "email_reply_forward_thread": "Compose an explicitly requested reply or forward using visible source message, recipients, speaker roles and thread purpose. Candidate bodies must preserve whose statements are being quoted. Do not assume the user wants a reply merely because an email field is open.",
    "json_patch_operation": "Provide a complete JSON Patch document for a visible initial JSON object and desired change. State array index and replace-versus-add semantics when relevant. Keep paths/escaping and existing-versus-absent members explicit; no hidden source document.",
    "shell_symlink_dereference": "Perform a bounded filesystem operation whose central distinction is preserving a symbolic link versus following its target. State the actual synthetic link/target relationship and stock macOS command environment. Avoid generic file-copy-only tasks.",
    "timezone_scheduling": "Convert a stated local date/time between named time zones into an explicit destination format. Provide date and ambiguity resolution when DST folds/gaps matter. No correct answer may depend on an unseen locale or unspecified date.",
    "spreadsheet_conditional_lookup": "Choose a complete formula with explicit lookup value, lookup/output ranges, matching condition and missing-match behavior under the visible engine. Keep conditional lookup central; no hidden worksheet cells or unstated source range.",
    "accessibility_alt_caption": "Write alternative text versus a visible caption for a stated purpose. Supply an actually visible textual description of relevant image content if semantic content is required; image dimensions or a filename do not reveal pixels. Preserve purpose and avoid invented details.",
    "units_decimal_locale": "Format or convert a visible measurement with an explicit source unit, target unit, decimal/group separator policy and required precision. State rounding when decisive. Locale identity alone must not stand in for absent numeric input.",
    "api_idempotency_header": "Choose whole HTTP header lines for an explicit retry/idempotency scenario and a visible request identity/key policy. State which operation is the same retry versus a new operation and which key must be reused. Never derive a hidden key from a candidate's position.",
}


def mother_task(family_id: str) -> str:
    return COMMON + "\nOperation recipe: " + RECIPES[family_id]


def recipe_id(family_id: str) -> str:
    if family_id not in RECIPES:
        raise ValueError("No heldout recipe is registered for this family")
    return "heldout-v7:" + family_id


__all__ = ["candidate_space", "profile_for_spec", "mother_task", "recipe_id"]
