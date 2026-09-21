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

An actual label call at 07:12:43 UTC exposed an identity-check defect: Pi's
family was `swe-2`, but the live thinking map resolved `medium` to
`claude-opus-5-medium`. Matching the resolver to that same map did not enforce
the requested family. Four contributed Train rows are excluded before formal
training; the original response and its 5,818 reported tokens remain recorded.
The independent heldout receipt scan found no foreign UID.

The append-only [UID correction](../configs/teacher_correction_swe2_uid.json)
adds [v2 source pins](../provenance/pi-swe2-runtime-v2.json) and a
[new bridge](../tools/pi_teacher_extension_v2.ts). Both the catalog mapping and
resolved UID must equal the fixed `swe-2-medium`, `swe-2-high` or `swe-2-max`
for the requested effort before the provider is called. The validated map is
copied before delegation. Original policy, pins and bridge bytes remain intact.
The [offline regression](../reports/engineering/pi-exact-uid-guard.json)
reproduces the v1 defect and checks the v2 guard with zero real teacher calls.
Valid earlier caches retain their original source identity; their raw receipt
must prove the expected actual UID before reuse.

The new provider initially ran through a separate coordinator with at most six
simultaneous requests. Its first measured Train/Dev production window accepted
238 new Pi-labeled episodes in 12.53 minutes, reporting 368,783 tokens across all
observed attempts. Of those episodes, 217 were newly authored by SWE-2 and 21
reused intact Kimi authorship. These counts include the actual repair and review
costs and are not a dollar estimate or an accuracy result; see
[the production report](../reports/data/pi-initial-production-window.json).

The append-only [resource supplement](../configs/resource_supplement_swe2.json)
allows a local maximum of 12 after existing requests drain and activation is
recorded. It starts with 6 Train, 2 Dev, 2 Calibration and 2 Test workers, lets
Train borrow slots once other splits have frozen, and retains the same v0 gates
for hard-example scoring. A real 429 lowers the effective cap to the previously
healthy 6 and applies backoff; restart does not silently raise it again. Resource
settings are audit metadata and do not invalidate successful teacher caches.
These are local scheduling limits, not a claim about provider entitlement.
After a natural drain, production resumed at 07:30:49 UTC. The first measured
4.49-minute window accepted 208 new Train/Dev episodes, or 46.31 per minute,
with 267,948 reported tokens and no new completed attempts with unknown usage.
The prior six-slot window observed 18.99 episodes per minute. Different source
cohorts, queue carry-in and cache reuse limit the comparison; it is not a
controlled speedup experiment. See the [activation record](../reports/data/resource-activation.json),
[production window](../reports/data/resource-production-window.json) and the
independent [heldout verification](../reports/data/heldout-resource-window.json).

Kimi producers remain retired and their quota state is retained. There is no
automatic Kimi fallback.

## Local blocker notifications

`uv run python -m tools.production_watch --run-plan configs/run_plan_efficient.json`
observes only aggregate progress, coordinator state and reported pipeline errors.
It calls no teacher, reads no examples, and never changes the coordinator or
restarts production. It posts a local macOS notification for a provider block,
at least five minutes of provider backoff, persistent concurrency reduction,
reported training/evaluation/publication failure, exhausted finite data backfill
or hard-example confirmation with a remaining deficit, and completed publication.
Routine batches and an intentional maintenance drain do not produce alerts.
Notification deduplication persists across observer restarts and rearms after
the condition clears. Command failures can retry after five minutes.

Its current summary and delivery-attempt records are private under
`local/monitor/ranker-v1-efficient-20260921/`. Only fixed messages enter
AppleScript arguments; provider responses, example text and credentials do not.
An accepted notification command does not prove the user saw a banner: display
is controlled by [macOS notification preferences](https://developer.apple.com/library/archive/documentation/LanguagesUtilities/Conceptual/MacAutomationScriptingGuide/DisplayNotifications.html).
The observer reports recorded failures; it cannot diagnose every silently hung
process or post a new message into an inactive Codex conversation.

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
