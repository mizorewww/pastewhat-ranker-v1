# Preregistered evaluation protocol

This protocol is recorded before Calibration or Test generation and before any
student result is observed. The independent evaluator owns both splits. The
training and training-data agents may inspect this protocol and aggregate
manifests, but not their examples or labels. The parent integrator may not use
Test examples for tuning. The older 120-case PasteWhat benchmark is already
exposed and is not this release's held-out test.

## Data isolation

The shared `data_tools/family_partition.json` assigns complete conceptual
operations to Train, Dev, Calibration, and Test. An operation includes its
paraphrases, entity changes, counterfactuals, languages, and candidate
permutations. Task domains and representation kinds may recur across splits;
the decision operation must not. Mixed-operation examples crossing partitions
are rejected. A hash and Git commit of the partition precede generation.

The original suggested counts are 20,000 Train, 1,000 Dev, 1,000 Calibration,
and 2,000 Test. Actual production sizes must be registered separately under
`RUN_PLAN_FORMAT.md` before formal training, calibration, or scoring. No size is
selected implicitly by importing code. Every formal evaluator command requires
`--run-plan`; the unchanged conceptual partition still covers eight Calibration
operations and twelve Test operations. `run_contract.family_quotas` balances the
registered total in fixed family order, and `action_quotas` apportions global
70/20/10 selectable/no-match/missing-intent counts, including whole-episode
rounding. Missing intent pools ambiguity and insufficient information. Formal
replay verifies every original sampling slot and actual accepted label bucket.
Manifests, calibration, scores and the final freeze bind the same `run_id` and
`run_plan_sha256`; changed plans cannot resume existing caches. Private formal
data lives only at the plan's `local/evaluator-heldout/<run_id>/` paths. Explicit
`--staging` authoring probes are not registered production data and cannot be
scored as formal Calibration or Test. These are planned counts, not achieved facts.

The first production plan is `configs/run_plan.json`, run
`ranker-v1-local-20260921`, registered before formal scoring with SHA-256
`0095dcd6c15c7cdb28a8d1ee71443783e9d2f3ec666ca225c377521313c8f2f6`.
It specifies Train 1,000, Dev 200, Calibration 400, Test 600 and a 500-episode
Train pilot. Calibration has 50 episodes per family, split 200 fit / 200 threshold;
Test has 50 per family including 35 planned selectable cases. All quality gates,
including 30 answerable cases per critical Test family and at least 25 Calibration
recommendations, remain unchanged. This run does not claim completion of the
original suggested 20,000 / 1,000 / 1,000 / 2,000 data scale.

Only synthetic content is used. The vendored native Swift context projection
first derives input surface from actual AX metadata; generated surface guesses
are ignored, and inaccessible contexts cannot contain AX-only field content.
Formal generation uses the `pastewhat-capture-authoring-v1` evidence contract:
the actual focused text window, valid UTF-16 selection range or explicitly unknown
range, and at most four bounded neighboring static strings. The authoring context
has no free-form `surroundingText`. Shared production Swift converts this evidence
to `pastewhat-focus-v1` before student preprocessing. Raw capture remains only in
generation provenance; it cannot give the teacher extra information. Unversioned
earlier free-form drafts remain outside the formal data state.
Formal v6 uses `teacher-episodes-v6-compact-verdicts`. Its compact author supplies
only slot, zero to two short displayed guidance strings, the selected complete
old field value (or empty), and whole candidate payloads. Twenty evaluator-owned
field profiles are fixed in `evaluations/authoring-profiles.json` before generation.
Its v2 revision makes implementation-dependent runtime information explicit in
the same observable helper text for every label bucket. Superseded v1-profile rows
were quarantined with their unchanged inputs, labels and audits before any student
scoring; the registered episode counts and native model-input contract did not
change. This is an input-quality revision, not a model-result-based label change.
The shared `compact-literal-fixture-v1` builder constructs the full observable
context and capture without inspecting a label or inferred intent. An empty
selection means an empty field with a known insertion point. Whole-field
replacement has empty before/after boundaries. Candidates supply payload
fixtures instead of declared types: a literal text body, synthetic file
basenames, or blank PNG dimensions.
The same production Swift codec derives each candidate's text, kind and
capabilities before any tokenizer budget or teacher label. Coarse native kinds
are preserved even when a code/color string is classified as text. File/image
summaries do not imply visible contents or a textual clipboard representation.
Raw fixture hashes and shared native-source provenance accompany each episode;
the independent audit replays both context and candidate projection exactly.
Earlier rows with declared candidate metadata remain isolated with their original
teacher audits and labels; none are silently moved into a registered v6 run.
The synthetic authoring domain excludes the exact pattern in which a nonempty
selected value is duplicated as the entire unselected prefix with an empty suffix,
or as the entire suffix with an empty prefix. Authors can mistakenly represent a
whole-field replacement this way, producing two old values before the paste.
Such rows are quarantined unchanged and regenerated, rather than repairing their
capture or labels. This is a corpus constraint; the real AX projection continues
to accept legitimate repeated text. Whole-field replacements use empty boundaries.
The actual production `Preprocessor` then budgets and clips every episode before
two blind teacher labeling calls, with independently shuffled candidate order
and rewritten opaque candidate IDs in the second call. Labels must agree after
mapping back to the original IDs. Proposed labels or generation rationales do
not enter those calls. The teacher sees exactly the prepared context and
candidate records, without family, intended label, audit evidence, or source
application identity. A separate blind classifier receives the fixed 68-operation
taxonomy and visible input, with no expected family, answer, or split name. Its
observed operation must equal the assigned family, with no separately requested
secondary operation, and its deployment-realism review must pass. Earlier
confirmation-style family reviews are retained as audit history but are not
sufficient for release. Neither the two labelers nor this classifier sees a
proposed answer. Disputed slots are replaced by newly generated episodes and
independently labeled again. The approximately 70/20/10 quota is enforced on
actual independently agreed labels: a valid episode in another sampling bucket
is retained in a local quarantine with its unchanged label and replaced for the
release dataset. Review is by agents and the teacher;
it must never be described as human validation. The report records rejected,
disputed, repaired, and accepted episode counts and the responding teacher
model, rather than assuming an API alias identifies fixed weights.

Both blind labels use `independent-candidate-verdicts-v1`: every candidate gets
an independent usable boolean and short audit evidence. The standard acceptable
ID set is derived from all true verdicts. A complete positive ID/text quotation
set must match the actual input, and identical plaintext cannot receive
contradictory usability. Canonical wording, ordering or brevity is not an implicit
preference. Evidence remains outside student inputs. Native file/image summaries
are never treated as textual clipboard contents without text capability. The
provider's `response_format=json_object` constrains JSON syntax; it does not prove
semantic correctness. Agreement between two teacher passes remains a fallible
quality filter and is followed by independent evaluator review.

When an unlabelled compact draft has an incorrect candidate count, the author may
return a complete corrected array while preserving its visible guidance and
selection exactly. Both author responses and their lineage stay in the audit.
The repaired draft then goes through native projection, preprocessing, two blind
labels and blind family review from the beginning. Code never truncates, pads or
edits an accepted candidate set to satisfy a count.

Within each fixed family allocation, candidate counts (integers 1–20, except
the predeclared finite RSVP status operation with 1–4 candidates),
languages, and requested label buckets are shuffled with separate fixed seed
streams. Sharing a modulo schedule among these fields would create a label
shortcut. A one-candidate ambiguity request becomes insufficient information in
the same abstention bucket without changing candidate count. The accepted labels,
counts, and languages are reported from the completed data, not merely the plan.

Literal paste usability is part of both blind labeling passes and deployment
review. Only the observed selection is replaced; otherwise the entry is inserted
unchanged, including whitespace, quotes, newlines, and escaping. The labeler may
not assume an unselected placeholder is replaced, an invisible cursor is moved,
or missing syntax is supplied. Existing accepted rows from an earlier labeling
prompt must pass both updated blind labels with their original labels unchanged;
disagreements are quarantined and the slot is regenerated. Confirmed evaluator
semantic rejections are retained with original teacher provenance and cannot be
readmitted through an ID or order change. All such checks occur before any student
Test prediction; they are not corrections based on model errors.

The one permitted label disagreement is `ambiguous` versus
`insufficient_context` when both independent passes return `abstain` with an
empty acceptable set. Both supervise the same abstention action and belong to
the preregistered 10% sampling bucket. The first stored label is not changed;
both observed reasons and `reason_agreement: false` remain in provenance. These
episodes are reported under a combined ambiguity/insufficient-information group,
without a fine-grained reason accuracy claim. Disagreement between `no_match`
and either other reason remains a rejection, as does any select/action or
acceptable-candidate disagreement. This policy was fixed before student Test
scoring, based on annotation semantics rather than student performance.

## Calibration

Only the final intended deployment weights and precision are calibrated.
Calibration's eight conceptual families are deterministically sorted by
`sha256("pastewhat-calibration-v1:" + family_id)`; the first four are fit
families and the other four threshold-selection families. Registered family
quotas determine the two role counts (500/500 under the original suggestion),
without correlated family variants straddling the two roles.
The allocation and hashes are published before model scoring.

A `StandardScaler` and L2-regularized `LogisticRegression(C=1, solver="lbfgs",
max_iter=2000, random_state=42)` are fitted on the fit half only, with no
class-weight adjustment. The four features are:

1. Highest candidate score minus abstain score.
2. Highest candidate score minus second-highest candidate score, substituting
   the abstain score for the second score in a one-candidate episode.
3. Candidate count, without a hidden log transform.
4. A missing-context flag, true when field label and selected text are blank and
   the surrounding observation has no actual text after preprocessing. In a valid
   native `pastewhat-focus-v1` envelope, only nonblank before/after-selection text
   (or unknown-position textWindow) and nearbyText strings count; `format` and
   `selectionKnown` metadata alone do not count. Ordinary legacy strings and
   malformed/truncated envelopes remain literal text. This corrected feature is
   versioned as `pastewhat-calibrator-v2` so old coefficients cannot silently use
   the changed interpretation.

The target is one exactly when the episode is selectable and the highest-scored
candidate is in `acceptable_ids`. One shared implementation computes training
of this calibrator and its deployment probabilities. If the fit set lacks both
target classes, calibration fails explicitly; a constant all-reject predictor
is not a successful calibration.

Candidates only qualify for recommendation when their top score is strictly
greater than the abstain score. The threshold-selection half chooses the
threshold with maximum coverage among thresholds yielding at least 95%
observed recommendation precision and at least 25 recommendations. This absolute
evidence requirement stays unchanged if a smaller run is registered. Equal coverage is broken by higher precision, then
the higher threshold. If no threshold meets those requirements, report target
failure and retain the best-precision threshold with at least 25 recommendations
as a diagnostic candidate, not an accepted release. If fewer than 25 examples
ever qualify, target failure is explicit. A Wilson 95% interval and every
precision–coverage point are reported. Meeting an observed 95% target is not a
95% population guarantee.

Safe/empty bypasses, malformed responses, missing/nonfinite candidate scores,
or model failures do not silently become correct abstentions. Bypasses have
their own outcome and model failures stay in all dataset denominators. Raw
scores, feature rows, fitted coefficients, scaler, threshold, deployment hash,
and code revision are retained. Calibration does not choose model checkpoints.

## Freeze and final acceptance

The evaluator runs final Test only after the parent explicitly freezes model
weights, tokenizer, production preprocessing, deployment precision, calibration
parameters/threshold, and baseline implementation. The freeze manifest records
file hashes, the Test hash, the registered plan itself and `run_contract.py`.
The runner refuses to proceed if any frozen
artifact has changed. No test-derived error analysis, label correction,
threshold change, or checkpoint selection may feed this version's release.
Corrections require a versioned explanation and a new, independently held-out
test for subsequent acceptance.

The freeze also requires `pastewhat-training-handoff-v2`: the complete pilot,
all three main seeds, executed hardening, actual optimizer steps and episode
exposures, original encoder initializations, and Dev-only selection. Fifteen stage
summary/config/manifest files, ten aggregate evidence files, the handoff and the
reference weight file are individually hash-bound. The evaluator and release
assembler share one verifier; it never follows evidence links into Train/Dev
example files. Missing stages, changed evidence, or a non-Dev winner prevent
final acceptance.

The paired baseline is PasteWhat's existing Laya workflow at commit
`87f9c09` (full commit and engine file hashes resolved in the run manifest).
Both systems receive exactly the same prepared, category-only context and all
1–20 candidates. The baseline may apply its production prefilter; the new ranker
encodes every candidate. Input preparation runs once, before either system.
Jev, if available, is an additional remote baseline and never substitutes for
the Kimi teacher or the primary paired comparison.

The primary quality target is at least **+5 percentage points** in answerable
Top-1 over the production baseline. This is the fraction of selectable episodes
actually recommended with an acceptable ID; abstaining on a selectable episode
is a miss. All predetermined conceptual families are key groups. A material
group regression is a decline greater than 5 percentage points on a group with
at least 30 selectable examples. The no-regression gate passes only when every
preregistered critical family has enough evidence and none materially regresses.
An empty set of sufficient groups, or any insufficient critical group without an
observed regression, is `inconclusive` and cannot qualify a release. Report all
smaller groups without claiming
that low counts establish parity. Also report:

- Raw ranking Top-1, separately labeled and never substituted for actual
  user-visible answerable Top-1.
- Decision accuracy, recommendation precision together with coverage, and
  false-promotion rates for no match, insufficient context, and ambiguity.
- Recommendation precision intervals, multi-positive outcomes, same-kind
  negatives, language, candidate-count, and field/category-conflict slices.
- Paired gains/regressions and a 2,000-replicate family-block bootstrap interval
  for the primary delta (fixed bootstrap seed 423190).
- All inference failures and protocol errors, with denominators unchanged.
- Actual-Mac cold startup/load, warm p50/p95 latency and peak memory for 1, 5,
  10, and 20 candidates. GPU measurements run without concurrent training or
  other evaluation inference. Candidate pair token lengths accompany results;
  N candidates means N encoder text pairs, not one sequence.
- PyTorch/MLX FP16 score differences and decision agreement, plus permutation,
  padding, and save/reload invariants established before final Test.

The separate, unlabeled numerical regression input file has 16 fixed cases:
1, 5, 10, and 20 candidates at four text-length profiles, with multilingual
content and intentionally identical interchangeable entries. Its predeclared
maximum absolute PyTorch/MLX score difference is 0.05 and mean difference 0.01.
Within-backend permutation, text-padding, and mixed-candidate-batch differences
must be at most 0.02 for MLX FP16 and 0.001 for PyTorch FP32. Reloading the same
saved artifact must reproduce scores exactly. Padded candidate scores must be
negative infinity. Raw actions must agree except for records with identical
payload and metadata; the same check is repeated using the final calibrator.
These checks establish numerical behavior, not semantic accuracy. The report
retains failures and actual errors instead of silently loosening tolerances.

Failing any quality gate is reported plainly. Synthetic-only measurements make
claims only about the generated and audited synthetic distribution; no real
clipboard or user acceptance rate can be inferred from them.

## Documentation checked before implementation

Context7 resolved `/websites/scikit-learn_stable` and retrieved the current
`LogisticRegression` fitting/probability API. The official [calibration
documentation](https://scikit-learn.org/stable/modules/calibration.html)
describes probability reliability and calibration curves. [Guo et al.
(2017)](https://proceedings.mlr.press/v70/guo17a.html) motivates checking
calibration empirically; it does not establish that this four-feature custom
calibrator will work for clipboard decisions.
