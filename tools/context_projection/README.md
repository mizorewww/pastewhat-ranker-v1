# Production Swift projection

`Models.swift`, `RecommendationContext.swift` and `FocusText.swift` are unmodified
copies from [PasteWhat commit c0657c0](https://github.com/mizorewww/pastewhat/tree/c0657c0/Sources/PasteWhat).
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
