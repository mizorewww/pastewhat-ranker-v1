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

A deterministic hash preselects approximately 10% of complete source batches for
a second independent label after candidate IDs and ordering change. Programmatic
concerns, such as a Python positive that does not compile at the visible insertion
point or clipped input, also trigger this batch review. The second labeler sees
only the same student-visible input; it does not receive the source operation.
Disagreements are rejected without editing teacher labels. Accepted provenance
distinguishes `single_pass`, `sampled_reviewed` and `risk_reviewed`. This protocol
does not claim every row was independently reviewed twice, human validated, or
semantically correct merely because a teacher returned a valid answer.

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
