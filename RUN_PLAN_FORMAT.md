# Registering a production run

The initial data sizes are recommendations. A different size must be selected
before formal training, calibration and Test scoring, with its measured cost
evidence and reason recorded. A smaller run must never be reported as completing
the original 20,000 / 1,000 / 1,000 / 2,000 recommendation.

`run_contract.py` provides `load_run_plan(path)`, `family_quotas(partition, split,
total)` and `action_quotas(family_counts)`. Importing it selects no plan. It reads
only the plan, fixed source contracts and bound aggregate cost reports; it never
opens examples or labels. The original `data_tools/family_partition.json` stays
unchanged so the completed engineering checks retain their provenance.

A production plan has these required keys:

- `version`: `pastewhat-run-plan-v1`
- `run_id`: a new lowercase identifier, at most 64 letters, digits, `_` or `-`
- `split_targets`: explicit positive counts for Train, Dev, Calibration and Test
- `pilot_episodes`: a subset of the main Train target
- `hardening`: separate `pool_episodes`, `review_nominations`, `accepted_new` and
  `retained_original` counts; a candidate pool is not an accepted hard dataset
- `training_seeds`: `[42, 43, 44]`
- `epochs`: `{"pilot":2,"main":2,"hardening":1}`
- `head_warmup_steps`: explicitly recorded because its relative exposure changes
  when a dataset is smaller
- `effective_batch_episodes`: `16`
- `family_partition_sha256` and `projection_provenance_sha256`: fixed source hashes
- `teacher_contract_version`: the exact generation/labeling protocol in use
- `registered_at`, `registration_reason`: timestamp and evidence-based decision
- `cost_evidence`: one or more `{"path":"relative aggregate report","sha256":"..."}`
- `quality_gates`: the unchanged quality requirements shown below

The optional `diagnostic_episodes` field registers an intermediate Train subset
strictly larger than the pilot and smaller than main Train. It uses the same
seed and training hyperparameters as the pilot, starts from the same original
initialization, and is compared on the same Dev snapshot. It does not add unique
examples to the main Train count. Plans without this field preserve their prior
behavior and hashes.

```json
{
  "recommendation_precision": 0.95,
  "minimum_calibration_recommendations": 25,
  "top1_improvement": 0.05,
  "maximum_group_regression": 0.05,
  "minimum_answerable_per_test_family": 30,
  "minimum_dev_group_episodes": 20
}
```

Family quotas use division with remainder in the original partition order. The
action allocator apportions exact global 70/20/10 buckets, with whole-episode
rounding and largest-remainder distribution. The final bucket combines ambiguous
and insufficient-context cases. The data owner still independently samples
candidate counts and language; labels cannot determine the candidate count.
Every declared family remains covered. The original 2,000-row Test allocation is
preserved by the same quota function.

The returned `RunPlan` exposes a copied `document`, `target(split)`, `run_id`,
`binding()` and `verify_unchanged()`. Its binding contains `run_id` and
`run_plan_sha256`; store both in generation caches, data manifests, training
configs, stage handoffs, calibration and the final freeze. Bind the plan file
itself in the final artifact manifest. Refuse resume when the registered hash
differs. Reuse the completed 32-example engineering check without changing its
configuration; each formal run still starts its encoder from the original
checkpoint rather than from that engineering checkpoint or the pilot.

`data_path(stage)` provides stable paths under `data/frozen/<run_id>/` for
`pilot`, `diagnostic`, `train`, `dev` and `hardening`. Heldout paths remain under
`local/evaluator-heldout/<run_id>/`. `pipeline_directory`, `checkpoint_directory`
and `report_directory` separate execution state and artifacts by run identity.
These are repository-relative paths; the existing commands run from the repository
root. Full generated JSONL files remain ignored by Git.

Insufficient calibration evidence or fewer than 30 answerable cases in a
registered critical Test family yields an inconclusive quality gate, not a pass.
Overall metrics and diagnostic model artifacts can still be reported honestly.
No plan schema check or example allocation is itself a registered production run.
