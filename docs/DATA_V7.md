The active run is `ranker-v1-efficient-20260921`, registered in
`configs/run_plan_efficient.json`: 20,000 Train and 2,000 Dev episodes, with
5,000 and 10,000 immutable Train subsets for the learning curve. The independent
evaluation owner produces 2,000 Calibration and 3,000 Test episodes. These are
targets, not completed counts. Earlier runs and their audits remain engineering
history and are excluded from this run.

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
