# Efficient heldout production

The registered run is `ranker-v1-efficient-20260921`, bound by
`configs/run_plan_efficient.json`: Calibration 2,000 and Test 3,000, each with
250 episodes per existing conceptual family. The previous v6 run remains
historical and contributes no rows to this run. Test scoring still requires
the final deployment weights, precision, preprocessing, calibration and policy
freeze. Calibration keeps its existing four fit/four threshold families.

`evaluations/generate_v7.py` is a thin owner of the shared
`data_tools.v7.produce_batch`. It supplies only heldout mother-task recipes,
profiles and independent sampling plans, and stores batch data in ignored local
directories. An explicit `--max-batches` bounds the initial cost check; its
partial result cannot be used for formal scoring.

After the initial cost check, `evaluations.produce_v7` resumes all registered
sources with one worker per split. Its immutable local `sampling.json` records
every original logical slot before full production. Each rejected logical slot
can receive up to seven newly seeded source situations after the original,
for eight situations total. Each situation retains the shared one-author-repair
limit. Replacement registrations preserve family, action, observation, requested
language, count target and the original independent-review cohort. Exactly one
accepted situation can fill a logical slot. Exhausting the finite budget leaves
an explicit incomplete result; it does not silently lower quotas.

The producer resumes successful author responses from raw audits and obeys the
selected provider's shared account cooldown. It writes a final private JSONL only at complete
registered quotas, replays its sources/labels, and publishes aggregate manifests
and content fingerprints. This data freeze does not authorize student Test
scoring. Final inference still requires the separately authorized model freeze.

The append-only `configs/teacher_transition_swe2.json` records the user's switch
to Pi's `devin/swe-2` for subsequent calls. It leaves the original run, family
partition, sampling and quotas unchanged. The original 58 Calibration and 67
Test rows and their Kimi audits remain unchanged. A cached Kimi author fixture
may receive a new SWE-2 primary label or review; provenance records each role
separately. New calls have no Kimi fallback. Pi uses independent account state,
so changing providers does not clear or shorten Kimi's provider limit.

Pi requests run without workspace context, earlier turns or tools. The pinned
bridge supplies the exact declared system prompt and one bound user message at
the provider boundary. Audits preserve the bound request, complete JSON event
stream, single-call receipt and their hashes. Independent replay reconstructs
the normalized result from exactly one final assistant `message_end`, never
from streaming deltas. Effective medium/high/max model variants and runtime
source hashes are recorded; these are rolling remote models, not pinned weights.
The registered mapping turns disabled/low authoring or primary-label requests
into SWE-2 medium, while independent high-effort reviews use SWE-2 high.

Known usage is counted once for each observed completion, including replies
rejected by receipt, process or label validation. Timeout and other failed
attempts retain unknown usage explicitly when no counters were observed.
Reports separate provider/model counters and preserve raw cache counters;
they do not infer reasoning/cache inclusion or dollar cost from Pi metadata.

Each batch gets one author request and one independent compact blind label
request after native projection and student budgeting. Candidate IDs and order
are made opaque for labeling. A predetermined hash selects about 10% of whole
batches for independent label review; program-detected issues and truncation
also trigger review. Both decision passes receive only the masked, budgeted
visible episodes. Any sampled source-scope audit is a separate request that
outputs lineage judgments only, never decision labels; its source metadata is
not sent to either decision pass. The remaining accepted rows are
explicitly `single_pass`. No claim is made that they are all independently
double labeled. A batch receives at most one repair attempt; unresolved slots
must be replaced with new source situations and distinct lineage.

The label is exactly `decision`, `acceptable_ids`, and `abstain_reason`.
Program checks retain complete candidate sets, validate exact ID membership,
compare actual labels with registered action quotas, and reject duplicate or
already excluded content. Teacher rationales are not requested and never enter
student inputs. Source family and recipe are authoring/audit lineage, never
evidence for a user's intention in a blind label request.

Candidate count is a sampling target, not an exact-integer admission gate.
Complete author arrays containing 1–20 entries are retained unchanged and
labeled only after native projection/budgeting of that actual array. The record
stores planned count, actual count and their difference; empty or over-20 arrays
remain invalid. Previously unaccepted count/format failures may reuse their
original successful author response for a new blind label, with a separate
recovery record and no extra author call. Semantic rejections and existing
accepted rows are excluded from this path. Actual count coverage and its
relationship to action/language are reported.

Within the unchanged 10% missing-intent quota, the initial plan assigns 40% to
no accessibility and 20% to a generic field without task text. This means
Calibration 80/40 and Test 120/60. The remaining slots use the standard view.
These are initial v7 assignments; the old observation supplement is not applied.
Language and candidate count are independently shuffled, with the existing
finite RSVP vocabulary restriction.
Reports call the source language `requested_context_language`; they do not claim
to detect actual written language. Observation variants are reported directly
from per-row provenance. Unknown legacy factors remain explicitly unspecified.

The heldout recipes require visible, determined intent for select/no-match
tasks. Personal preferences that imply opposite actions are not interchangeable
positives. A sole available candidate or a generic RSVP prompt does not supply
the missing choice. The author must put the already communicated decision in a
real observable source field; otherwise the episode belongs to missing intent.

For final audit, retain and bind:

| Artifact | Required evidence |
|---|---|
| Owner binding | Run ID/hash, protocol, family partition, recipe and profile hashes |
| Batch record | Exact source spec/hash, planned IDs/counts/languages/observation states, accepted/rejected rows, at most two author attempts |
| Per-row provenance | Mother-task ID, source family/spec hash, author/label audit IDs, optional blind decision-review and separate source-audit IDs, native/preprocess/visible hashes, observation variant, quality path, actual per-role provider/model/effort |
| Teacher audits | Original request/response hashes and known usage, including invalid responses and unknown attempts; Pi bound prompt, raw events and single-call receipt |
| Teacher transition | Append-only policy, pinned bridge/provider sources and per-role aggregate source distributions; historical Kimi rows are not rewritten |
| Aggregate audit | Complete split and actual action/observation counts, duplicate check, raw author→native→budget replay, compact label mapping, sampled/risk review agreement and source-scope checks |

Final freeze uses a protocol branch: v7 binds these batch/owner/audit artifacts
and shared source files rather than requiring the v6 per-row classifier and
exhaustive verdict fields. The old branch stays available for historical v6
audits. All quality, calibration and paired-comparison thresholds stay unchanged.
The final freeze also binds the teacher-transition policy and Pi source pins.
Split manifests and the release data manifest retain actual author, primary
and review source distributions, and Test reporting includes primary-model
slices without using them to select a threshold or checkpoint. Public packaging
copies the policy, pins and aggregates; private teacher requests remain private.
Semantic sampling reports its actual numerator and denominator; it is not a
human-validation or real-world accuracy claim.

The durable `evaluations.release_pipeline --run-plan configs/run_plan_efficient.json`
waits for complete data manifests/audits and the genuine completed training
handoff. Parent authorization of 2026-09-21 is conditional on these actual
prerequisites and applies only to plan SHA
`e5491476a01f3cd3d1b3f3778ac90f4e1dbeadca0e001a078003633266b7963f`.
It waits for the training process to release its lock before using the GPU,
verifies parity/performance, fits the fixed Calibration family partition,
checks calibrated parity, and installs the calibrated policy before final
freeze. The selected weights and preprocessing remain unchanged.

Only after that freeze does it run one Test pass per ranker, pinned Laya and
pinned Jev client, then produce the paired report. A partial interrupted Test
is not automatically rerun. Failed data/training proof or numerical parity
produces an incomplete diagnostic state without a publication handoff. A
completed quality miss produces a diagnostic handoff with the failed criteria;
it never changes Test labels, the checkpoint or the threshold.

Aggregate progress and final publication inputs live under
`local/evaluator-release/ranker-v1-efficient-20260921/` as `status.json` and
`ready-for-publication.json`. The latter contains run binding, status and
`artifacts.{reference,deployment,freeze,metrics,parity,performance,calibration_report,data_manifest}`.
Files use `{path,sha256}`. Model directories additionally use an external
`manifest_path` containing exactly the relative-path-to-SHA mapping;
`sha256` equals that manifest's hash and `weight_sha256` binds its weights.
Root alone assembles and publishes the public bundle.
