The active run is `ranker-v1-efficient-20260921`, registered in
`configs/run_plan_efficient.json`: 20,000 Train and 2,000 Dev episodes, with
5,000 and 10,000 immutable Train subsets for the learning curve. The independent
evaluation owner produces 2,000 Calibration and 3,000 Test episodes. These are
targets, not completed counts. Earlier runs and their audits remain engineering
history and are excluded from this run.

After the user's explicit provider change, future calls use Pi's `devin/swe-2`
through the frozen `configs/teacher_transition_swe2.json` policy. Existing
accepted Kimi rows stay unchanged; successful cached Kimi author fixtures may be
reused, with new primary/review decisions supplied by SWE-2. New rows record the
actual teacher separately for each role. The source sampling, native projection,
student budget, label contract and split quotas are unchanged.
Pi runs in a private empty directory with tools, sessions, extensions other than
the pinned teacher bridge, skills and workspace context disabled. The bridge
replaces the provider input with exactly the declared system prompt and one user
message, forces the output budget, and limits each process to one provider call.
The raw bound request, event stream and isolation receipt are retained with
hashes. This provider supports medium/high/max; prior off/disabled/low requests
map explicitly to medium and the mapping participates in cache identity.
Pi's coordinator and provider-aware cache are independent of Kimi's preserved
quota state. There is no Kimi fallback or automatic Kimi restart.
Observed completion usage survives invalid labels, receipt failures and nonzero
process exits; all-zero provider counters mean unknown usage, and cost metadata
zeros are not a billing statement. Known temporary transport failures receive
bounded backoff with the same pending source and cache. Authentication, account
restrictions or unclassified configuration failures remain visible for repair.

`teacher-episodes-v7-batched-decisions` batches ten complete paste decisions in
one author call. Code projects the author fixtures through the unchanged native
candidate/context adapters and the student's token budget before one independent
blind Kimi decision-label call. The response gives the complete acceptable-ID
set or an abstention reason. Neither candidate IDs, mother-task identity, planned
action nor unavailable context is evidence supplied to this labeler.

A valid author list with 1–20 entries is retained in full, even when its actual
count differs slightly from the independently sampled target. Provenance records
both counts and the difference; no candidate is inserted or removed to fit a
number. Empty lists and lists above twenty require the one allowed author repair.
Count distributions must be reported by action and language to expose bias.
The author is asked for short helper text, but admission follows the existing
native limit of 240 characters per static string and 600 characters across the
captured nearby strings, including fixed profile help. Text between the former
180-character author preference and this native limit is preserved verbatim.

A deterministic hash preselects approximately 10% of complete source batches for
a second independent label after candidate IDs and ordering change. Programmatic
concerns, such as a Python positive that does not compile at the visible insertion
point or clipped input, also trigger this batch review. The second labeler sees
only the same student-visible input; it does not receive the source operation.
Disagreements are rejected without editing teacher labels. Accepted provenance
distinguishes `single_pass`, `sampled_reviewed` and `risk_reviewed`. This protocol
does not claim every row was independently reviewed twice, human validated, or
semantically correct merely because a teacher returned a valid answer.
Opaque IDs use `e0`/`c0` consistently. If the label response has malformed or
unmappable rows, valid rows are kept and only the invalid rows receive one
label-only format repair. Author text is not regenerated for an ID formatting
error. Previously cached drafts rejected solely by count or ID plumbing may be
newly projected and labeled once; prior semantic rejections are excluded from
this recovery. Their original records remain unchanged.

Conceptual operation families remain in the original frozen partition. Each
source batch records its operation, mother-task ID, data seed and visible-fixture
profile before authoring. Variants do not become new independent conceptual
families. Source boundaries are controlled by this registry and owner review;
v7 removes the repeated full-taxonomy classifier from every row. Every batch has
at most two author attempts, including one repair. Failed source tasks remain
retired records; any subsequent quota replacement requires a new situation/data
seed while preserving the original rejected content and labels in audit storage.

The registered sampling uses 70% selection, 20% no match and 10% missing or
ambiguous intention. Candidate count and language use independent seeds. Within
the last bucket, 40% have no accessibility observations and 20% retain only a
generic field. These give 4% and 2% of the whole run respectively. Code removes
unavailable fields before labels are requested; no hidden complete scene is
provided to the teacher. All candidates survive native projection and budgeting.

The first execution is bounded to Train100/Dev20 source slots, coordinated with
Calibration40/Test40 for 200 slots overall. Real yield and total teacher usage
must be inspected before scaling. `local/v7/<run_id>/` stores immutable source
sampling, resumable batch records, the rolling accepted corpus and isolated raw
teacher audits. Usage summaries include known usage from invalid responses and
repairs as well as successful authoring and labeling. Reasoning tokens are a
subset of completion tokens. Unknown transport-attempt usage remains unknown;
cache replay is not another billable request. No dollar estimate follows from a
token count without verified billing evidence.
If an observed invalid response is followed by another attempt with the same
request body, its complete response and usage remain in
`prior_observed_responses`. Each new HTTP completion has an attempt ID;
`observed_responses()` enumerates these once, while successful cache replay adds
none. Aggregate request-body record counts and observed completion counts are
reported separately. This forward fix cannot reconstruct an earlier response
that might already have been overwritten.

Run the bounded owned check with:

```sh
uv run python -m data_tools.generate_v7 --run-plan configs/run_plan_efficient.json --split train --max-batches 10 --workers 2
uv run python -m data_tools.generate_v7 --run-plan configs/run_plan_efficient.json --split dev --max-batches 2 --workers 1
```

These commands obey the same account-wide cap, provider cooldown and persistent
operator dispatch hold as the evaluator. They never restart legacy generators.

Without `--max-batches`, the owned runner continues through the registered source
schedule and up to seven finite replacement rounds (eight mother situations in
total per logical slot). Each failed logical quota
slot gets a new source situation, mother-task ID and data seed while preserving
its family, action, language, observation variant and preselected review cohort.
Only one accepted episode can fill each logical slot. Exhaustion reports the
actual deficit instead of silently changing labels or claimed target counts.

`data_tools.freeze_v7.try_freeze` publishes the exact family/action allocations
as soon as available. It preserves 5k ⊆ 10k ⊆ 20k with unchanged rows, verifies
budget idempotence and label mappings, binds source audit-file hashes, and writes
the manifest/fingerprint sidecars before atomically publishing JSONL as the GPU
readiness signal. Dev is independently frozen at 2k. Freeze completion is
distinct from a rolling accepted pool reaching an approximate count.
An optional 1,000-episode `throughput-train.jsonl` may appear first, using the
same exact family/action allocation. It must already cover every Train family,
candidate counts 1–20 and an actual text pair of at least 512 tokens. The GPU
pipeline otherwise waits for the 5k pilot. Once published, this early snapshot's
unchanged rows are required members of the pilot, diagnostic and main snapshots;
its purpose is the pipeline's single representative throughput measurement.

The bounded six-request low/high measurement is recorded in
`reports/data/v7-teacher-effort-low-high.json`. Both efforts matched the action
sets of all twenty independently reviewed engineering cases; high omitted four
required abstention reasons, while low returned valid labels for all twenty.
For the ten larger current Train episodes, low returned valid labels in156s and
high hit the old240s timeout. The large-group quality comparison is therefore
unobserved. This is limited engineering evidence, not a general equivalence
claim. After root approval, new v7 primary labels and their single format repair
use low; sampled/risk secondary reviews and post-mining confirmation use high.
Actual request effort is retained in raw audits and new episode provenance.
Existing accepted labels are unchanged. Later clients use a480s read timeout to
reduce retries after the teacher has already spent substantial computation;
retry count and the shared account cap are unchanged.

The active Pi teacher also enforces its actual provider model UID. The append-only
`configs/teacher_correction_swe2_uid.json` registers a separate v2 bridge and
runtime pins; the original teacher transition and v1 files remain unchanged.
Before dispatch, the bridge requires the exact requested `swe-2-medium`,
`swe-2-high` or `swe-2-max` UID. Successful response parsing, cache replay and
freezing check the same identity independently of the live model catalog.
Valid v1 responses with identical visible request bytes can be replayed without
a provider call and keep their original audit IDs. New requests bind the v2
correction in their cache identity and source evidence. One observed foreign-UID
completion contributed four Train rows, which were excluded without editing
their original labels or audit bytes; its 5,818 known tokens remain in cost
accounting. The aggregate record is `reports/data/pi-model-identity-quarantine.json`.

Pi provider errors can report a model rate limit and a reset countdown without
exposing an HTTP status. These explicit error events now receive a distinct
`provider_rate_limit` classification, a sticky local fallback to six requests,
and the reported reset countdown plus a five-second margin. A missing countdown
keeps bounded backoff. This does not change account entitlement, upgrade a plan,
or switch models. Multiple Pi error events from one CLI attempt are not counted
as additional provider calls when the single-call guard blocked them; unknown
usage stays unknown. The first observed incident and offline replay are recorded
in `reports/data/pi-provider-rate-limit-recovery.json`.
