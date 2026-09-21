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

Within the unchanged 10% missing-intent quota, the initial plan assigns 40% to
no accessibility and 20% to a generic field without task text. This means
Calibration 80/40 and Test 120/60. The remaining slots use the standard view.
These are initial v7 assignments; the old observation supplement is not applied.
Language and candidate count are independently shuffled, with the existing
finite RSVP vocabulary restriction.

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
| Per-row provenance | Mother-task ID, source family/spec hash, author/label audit IDs, optional blind decision-review and separate source-audit IDs, native/preprocess/visible hashes, observation variant, quality path, actual model alias |
| Teacher audits | Original request/response hashes and known usage, including invalid responses and unknown attempts |
| Aggregate audit | Complete split and actual action/observation counts, duplicate check, raw author→native→budget replay, compact label mapping, sampled/risk review agreement and source-scope checks |

Final freeze uses a protocol branch: v7 binds these batch/owner/audit artifacts
and shared source files rather than requiring the v6 per-row classifier and
exhaustive verdict fields. The old branch stays available for historical v6
audits. All quality, calibration and paired-comparison thresholds stay unchanged.
Semantic sampling reports its actual numerator and denominator; it is not a
human-validation or real-world accuracy claim.
