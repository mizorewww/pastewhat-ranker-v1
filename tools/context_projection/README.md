# Production Swift projection

`Models.swift`, `RecommendationContext.swift`, `FocusText.swift` and
`CandidateProjection.swift` are unmodified copies from
[PasteWhat](https://github.com/mizorewww/pastewhat/tree/main/Sources/PasteWhat).
The exact source commit is recorded in `provenance.json`.
`provenance.json` records source hashes and the bounded AX reader hash. The wrapper
uses the actual field-to-surface projection and selection formatter. It supplies
no host name or bundle ID, and preserves a validated synthetic application category.

Call `project_context(context, capture=capture)` before student token budgeting
and before asking the teacher to label the visible input. `capture` is authoring
evidence kept in generation audits, not a student feature or a teacher-only hint.
The student still receives only the eight existing context fields.

```json
{
  "textWindow": "",
  "selectionLocation": 0,
  "selectionLength": 0,
  "nearbyText": ["Create a new resource with the endpoint request method."]
}
```

These keys are required exactly. Selection offsets are UTF-16 code units relative
to `textWindow`, not Python character indices. Use null for both offsets when
unknown. For a known range, the exact substring must equal `context.selectedText`;
an insertion point has length zero and empty selected text. Invalid or surrogate
splitting ranges are rejected. Unknown selection cannot contain selected text.
`context.surroundingText` must be empty when raw capture is supplied.

Authors can instead supply explicit literal boundaries, which avoids asking a
language model to calculate Unicode offsets:

```json
{"beforeSelection":"const method = '","afterSelection":"';","nearbyText":["Set the HTTP method to POST."]}
```

`context.selectedText` is the literal middle segment (empty for an insertion).
The Python adapter concatenates before + selected + after unchanged and computes
the UTF-16 offsets, then calls the same strict Swift implementation. These three
keys are the complete known-selection fragment form. The unknown-selection form
has only `textWindow` and `nearbyText`, with empty selected text. There is no
placeholder search, text rewriting, automatic truncation, or intention-based
position inference. The original author object is retained in the generation
audit and deterministically replayed; numeric capture remains supported for
earlier audits. `provenance.json` includes the adapter's source hash.

`textWindow` is the focused control's actual text window, at most 1,700 Swift
characters. `nearbyText` represents at most four static sibling labels/headings,
each at most 240 characters and at most 600 total. It cannot contain an imagined
user intention or text from another editable field. Native collection visits only
up to 32 children of one parent and up to three sibling positions on either side;
it never recursively scrapes the document. AX capture has a 30-call / 850 ms budget.
No AX access means no captured content or known selection. Secure input content
is rejected during synthetic authoring and cleared by the production projection.

The shared formatter stores JSON in `surroundingText`:

```json
{"format":"pastewhat-focus-v1","selectionKnown":true,"beforeSelection":"","afterSelection":"","nearbyText":["Create a new resource with the endpoint request method."]}
```

For an unknown range it uses `selectionKnown:false` and `textWindow` instead of
before/after fields. Strings are JSON-escaped. A fully empty observation stays
empty. Pasting replaces the actual selected range or inserts at the known caret;
there is no implicit deletion, quote insertion, placeholder search or caret move.
For example, bare `POST` fits an empty HTTP-method field but does not automatically
fit a JavaScript expression or an unselected `___` placeholder.

Omitting capture supports already-projected contexts and preserves idempotence;
new formal data must use the capture authoring contract. The adapter rejects extra
context/capture fields. Any authored `inputSurface` is ignored and recomputed by
the native implementation. After projection, the shared student preprocessor clips
to its token budgets; teachers see that exact clipped result, never the raw capture.

The JSONL CLI accepts either a projected context or an object containing only
`context` and `capture`. Compilation uses a process lock and a source-hashed cache
under ignored `local/`. It reads no clipboard history or running applications.
A Swift toolchain is needed only for generating new projected synthetic data.

## Candidate payload projection

Call `tools.project_candidates.project_candidates(entries)` alongside the context
projection, before token budgeting or teacher labeling. Each authoring entry has
exactly `id`, `sourceCategory`, and `payload`. The only payload variants are:

```json
{"type":"text","text":"git branch"}
{"type":"file","names":["Quarterly report.pdf","Outline.docx"]}
{"type":"image","width":1920,"height":1080}
```

The wrapper builds actual UTF-8 pasteboard data, synthetic file URL data, or a
blank grayscale PNG in memory. It passes these bytes to the same
`CandidateProjection.project` function the application uses. The result contains
only the deployed candidate fields: `id`, `text`, `kind`, `capabilities`, and
`sourceCategory`. Authoring payloads remain audit evidence and are not model
features. No real clipboard, user file, application, or image asset is read.

Text classification uses the application's existing rules, without a teacher
override. For example, `git branch` is `command`, while `cp a b`, an expression
such as `set(items)`, and `rgb(0, 0, 0)` are `text`. A filename typed into a text
payload remains text. File payload summaries come from actual URL basenames;
they do not include a made-up description or a hidden file path. Each file
fixture has 1–20 distinct basenames, at most 255 UTF-8 bytes per name, without
path separators or newlines. Synthetic URLs are never opened or created.

Image fixtures have positive integer dimensions, at most 8,192 per side and
16,777,216 pixels total. Core Graphics and ImageIO encode those blank pixels as
PNG. The production codec then reads the resulting PNG's metadata to obtain
the summary. The author cannot supply image semantics, an observed dimension
string, a kind, or capabilities. Actual application PNG/TIFF metadata can also
be missing, invalid, multi-frame, or inconsistent across representations; the
native implementation withholds dimensions in those cases. It never infers
the image's subject, performs OCR, or treats a PDF page size as pixel dimensions.

The payload protocol is `pastewhat-native-payload-v1`. The common provenance
file binds its Swift sources and `tools/project_candidates.py`. The JSONL CLI
accepts one array of authored entries per line and returns its projected array.
Audits replay the original fixture through the pinned code, budget the result,
and verify the exact student-visible fields used for the labels. Changing this
projection invalidates old labels unless that exact visible result is unchanged.
