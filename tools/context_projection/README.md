# Production Swift projection

`Models.swift` and `RecommendationContext.swift` are unmodified copies from
[PasteWhat commit 64e8817](https://github.com/mizorewww/pastewhat/tree/64e8817/Sources/PasteWhat).
Their SHA-256 hashes are recorded in `provenance.json`. The wrapper calls the
actual `AppContext.modelContext` implementation; it does not reproduce its regular
expressions in Python or infer surface from the synthetic task's intent.

`ProjectSyntheticContext.swift` accepts a model-shaped synthetic context on each
stdin line, ignores any proposed `inputSurface`, and emits the production inferred
surface. It supplies no real or fake host name/bundle ID. A validated synthetic
application category is preserved after the production projection. Unknown extra
fields are rejected by the Python adapter. Without AX access, nonempty captured
fields are rejected. Secure fields pass through production text clearing.

The production projection itself does not clear arbitrary fabricated fields when
`hasAccessibility` is false; the actual AX reader never obtains them. The adapter
therefore rejects that impossible synthetic combination before teacher labeling.

From the repository root, use `from tools.project_context import project_context`
or `uv run python tools/project_context.py` for JSONL. A Mac Swift toolchain is
needed only to generate new projected synthetic data. Compilation uses a process
lock and a source-hashed cache under ignored `local/`; neither clipboard history
nor application state is read. The data generator then uses the student's shared
`Preprocessor` and obtains fresh teacher labels for the projected, budgeted input.
