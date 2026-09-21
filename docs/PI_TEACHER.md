# Pi / SWE-2 teacher transition

On 2026-09-21 the user explicitly requested the existing Pi `devin/swe-2`
provider after Kimi's five-hour quota was exhausted. New authorship, primary
labels, sampled blind reviews and hard-example verification use that provider.
The original run plan, conceptual-family assignments, sampling schedules,
preprocessing budgets and release criteria remain fixed. The append-only
[transition policy](../configs/teacher_transition_swe2.json) binds the change to
the original run-plan SHA-256.

Before the switch, 217 Train, 62 Dev, 58 Calibration and 67 Test episodes were
accepted. Those 404 rows retain their bytes, labels and Kimi audits. An intact
cached Kimi author draft may be reused with new SWE-2 labels; the author,
primary labeler and additional reviewer are attributed separately. The model
does not receive any of this provenance as input.

## Transport and isolation

The [teacher bridge](../tools/pi_teacher_extension.ts) delegates to the user's
existing `../pi-devin` extension. It uses the existing Pi/Devin login without
exporting credentials or changing the provider identity. Pi runs without
workspace context, session history, skills or tools. The bridge replaces the
provider context with the exact bound system prompt and one user message,
enforces the requested output-token ceiling, and permits only one provider
completion per process. Pi's own prompt and the workspace are excluded from
the teacher input.

Pi 0.85.1 and the provider source revision/hashes are recorded in
[runtime pins](../provenance/pi-swe2-runtime.json). The registered family model
is `swe-2`; the existing provider maps `medium`, `high` and `max` to
`swe-2-medium`, `swe-2-high` and `swe-2-max`. Requests previously asking for
disabled or low reasoning use the provider's supported `medium` setting.
Every attempt records its requested controls, effective variant, runtime,
bound request hash and raw JSON event/receipt hashes. Unsupported controls are
reported as unapplied, not silently claimed as enforced.

The authoritative completion is the single assistant `message_end` event.
Stream deltas and repeated end-of-agent messages are not additional billed
completions. Observed token counters are retained; absent or all-zero token
counters remain unknown. Pi's zero-valued cost metadata is not a verified
billing statement. Private raw responses and reasoning stay in ignored audit
storage and never enter student features or public model bundles.

The new provider has a separate local coordinator with at most six simultaneous
requests. Kimi producers are retired and their quota state is retained. There
is no automatic Kimi fallback. This concurrency cap is a local scheduling
choice, not a claim about the provider's account limits.

## Verification and release evidence

One actual isolated Pi call labeled six existing Train-only engineering cases
(four select, two abstain). All six outputs passed the schema and matched the
engineering reference. The provider reported 2,467 input and 213 output tokens.
The [protocol report](../reports/engineering/pi-swe2-protocol.json) records the
receipt, usage and raw-output hash. This verifies the transport on a small
convenience sample; it is not a teacher or student accuracy benchmark.

Formal data continue through native projection, student budgeting, compact
blind labeling, programmatic gates and sampled or uncertainty-triggered review.
The independent evaluator preserves Calibration/Test isolation and freezes the
transition policy and runtime pins before final Test. Public split manifests
report actual teacher sources by role. The model package includes the public
policy/pins and aggregate provenance, without private teacher requests.

Current Pi documentation was retrieved through Context7 before implementation:
[JSON output events](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/json.md)
and [CLI usage](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/quickstart.md).
The installed provider's actual model mapping and stream implementation were
also inspected and exercised. Pinning the client does not freeze the provider's
remote rolling model weights or make repeated generations deterministic.
