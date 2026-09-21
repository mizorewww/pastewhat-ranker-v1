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

The intended counts are 20,000 Train, 1,000 Dev, 1,000 Calibration, and 2,000
Test. Calibration has eight operations with 125 episodes each. Test has twelve
operations: the first eight in the fixed manifest have 167 episodes and the
remaining four have 166. Candidate counts cover 1–20. The intended distribution
is approximately 70% selectable, 20% no match, and 10% insufficient information
or ambiguous. At least half of selectable episodes contain a negative candidate
of the same content kind. These are targets, not achieved facts.

Only synthetic content is used. The actual production `Preprocessor` budgets
and clips every episode before a fresh teacher labeling call. Proposed labels
or generation rationales do not enter that call. Candidate IDs are opaque and
shuffled independently. The teacher sees exactly the prepared context and
candidate records, without family, intended label, audit evidence, or source
application identity. A validation pass checks allowed IDs, label consistency,
family membership and visible evidence. Review is by agents and the teacher;
it must never be described as human validation. The report records rejected,
disputed, repaired, and accepted episode counts and the responding teacher
model, rather than assuming an API alias identifies fixed weights.

## Calibration

Only the final intended deployment weights and precision are calibrated.
Calibration's eight conceptual families are deterministically sorted by
`sha256("pastewhat-calibration-v1:" + family_id)`; the first four are fit
families and the other four threshold-selection families. Thus 500/500 examples
are allocated without correlated family variants straddling the two roles.
The allocation and hashes are published before model scoring.

A `StandardScaler` and L2-regularized `LogisticRegression(C=1, solver="lbfgs",
max_iter=2000, random_state=42)` are fitted on the fit half only, with no
class-weight adjustment. The four features are:

1. Highest candidate score minus abstain score.
2. Highest candidate score minus second-highest candidate score, substituting
   the abstain score for the second score in a one-candidate episode.
3. Candidate count, without a hidden log transform.
4. A missing-context flag, true when field label, selected text, and surrounding
   text are all empty after preprocessing.

The target is one exactly when the episode is selectable and the highest-scored
candidate is in `acceptable_ids`. One shared implementation computes training
of this calibrator and its deployment probabilities. If the fit set lacks both
target classes, calibration fails explicitly; a constant all-reject predictor
is not a successful calibration.

Candidates only qualify for recommendation when their top score is strictly
greater than the abstain score. The threshold-selection half chooses the
threshold with maximum coverage among thresholds yielding at least 95%
observed recommendation precision and at least 25 recommendations (5% of the
intended threshold split). Equal coverage is broken by higher precision, then
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
file hashes and the Test hash. The runner refuses to proceed if any frozen
artifact has changed. No test-derived error analysis, label correction,
threshold change, or checkpoint selection may feed this version's release.
Corrections require a versioned explanation and a new, independently held-out
test for subsequent acceptance.

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
at least 30 selectable examples. Report all smaller groups without claiming
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
