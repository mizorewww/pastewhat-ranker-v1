# Teacher decision-label distillation

The targets are 20,000 Train, 1,000 Dev, 1,000 Calibration and 2,000 Test episodes.
These are quotas, not completed-data claims. The actual accepted count, label
distribution, family count, candidate-count distribution and SHA-256 are recorded
in each data manifest. Train/Dev are authored by the data agent; a separate
evaluation agent owns Calibration/Test and their labels. The training agent does
not read those held-out examples.

The 68 conceptual operation families in `data_tools/family_partition.json` were
assigned and approved before generation (commit `c3d1517`). An operation and all
its paraphrases, literal substitutions, counterfactuals and candidate permutations
stay in one split. Shared application categories and broad domains are allowed;
the decision operation itself is held out. There are 40 Train, 8 Dev, 8 Calibration
and 12 Test families. A thousand literal substitutions are not a thousand
independent families. Synthetic-only performance is not real-world accuracy.

`kimi-for-coding` is the teacher. The current API returns that rolling model alias,
not an immutable underlying weight revision. We retain response model strings,
request bodies, timestamps, raw response hashes and token usage. It is possible
to reproduce the dataset bytes and training; it is not possible to promise that
repeating a rolling teacher request will reproduce its completion.

## The release gate for every Train/Dev episode

1. Kimi authors the complete visible context and all 1–20 candidates. Generation
   plans ask for approximately 70% select, 20% no-match and 10% ambiguous or
   insufficient-context decisions. These plans are never sent to the labeler.
   Observed labels, not requested labels, determine the reported distribution.
   If an independently agreed label misses the planned sampling bucket, its
   original label stays in quarantine and that slot receives a newly authored
   example. The two ambiguity reasons may share the final 10% bucket.
2. The actual pinned PasteWhat Swift projection determines the input surface from
   field metadata. The generation model cannot provide an intent-based surface
   oracle. Impossible no-AX contexts containing captured field data are rejected.
3. The shared student `Preprocessor` sanitizes and clips the episode. Metadata and
   special tokens have 64 tokens, visible context 448, and candidate text 512.
   All candidates are retained, and each final pair is at most 1,024 tokens.
4. A new Kimi request labels only those exact projected, clipped context/entries.
   Teacher labels are select with a set of interchangeable acceptable IDs, or
   abstain with no-match, ambiguous, or insufficient-context reason. Quoted
   selected candidate text checks ID mapping. There are no soft-logit targets.
5. Another blind Kimi request sees the same information with episode ordering,
   candidate ordering and candidate IDs changed. The complete action sets must
   agree after remapping. No-match versus other abstention reasons is also a
   rejection. If both passes abstain with empty positives and disagree only
   between ambiguous and insufficient-context, their common ABSTAIN training
   target is accepted in the shared 10% bucket. The first label remains unchanged;
   both reasons and `reason_agreement:false` are recorded. Such rows are excluded
   from reason-specific accuracy claims. Other disagreements are retained in
   audit and excluded; new episodes replace them. A majority vote never silently
   changes a disputed label.
6. A separate teacher call receives all 68 operation descriptions without split
   assignments and infers the observed operation from visible input. It sees no
   expected family or proposed label. A primary family mismatch or required
   secondary operation rejects the slot. It also reviews realistic field
   visibility and payload metadata. Schema,
   candidate counts, native projection, preprocessing idempotence, visible hashes
   and label membership are also checked by the production tools.
7. Only accepted episodes enter the current Train/Dev JSONL. Training uses frozen
   snapshots, never a file that the generator is still updating. The 5,000 pilot
   rows must be a recorded subset of the final 20,000 Train rows. Any hard-case
   augmentation is restricted to new examples from Train operation families and
   requires a frozen ranker-v0; Test is never an error-mining source.

The unreleased v1 probe exposed a wrong teacher ID selection, out-of-family
operations and impossible field metadata. The v2 probe additionally exposed
equivalent candidates omitted from positive sets, unusable replacement ranges,
authoring contamination from negative operation lists and expected-family reviewer
agreement bias. Those main-data probes are isolated in ignored
`local/initial-v1-unreleased/` and `local/initial-v2-unreleased/`. Production v3 is
`teacher-episodes-v3-visible-evidence`; it uses the four calls above, replacing
confirmatory family review with blind classification. Authors receive only the
positive target operation, while reviewers retain the complete taxonomy.

The 32-row engineering seed is a separately documented exception: the root agent
independently reviewed the original 32, rejected six, and then approved six new
Kimi-authored replacements after the same label gates. Its frozen provenance is
`data/train_overfit.review.json`. It covers multi-positive and abstain examples
but is not a representative accuracy benchmark. These inspections are agent
reviews, not human validation. Repeated teacher agreement and programmatic checks
reduce errors; they do not establish that every synthetic label is correct.

An additional first-batch v3 agent review found that code placeholders such as
unselected `___` had been treated as question answering: a bare `POST` token
would not replace that placeholder or supply missing JavaScript quotes. Similar
unselected existing-code replacements and one incomplete positive set were
quarantined without rewriting their labels. The content fingerprints and
findings are in `data/train_dev.placement_review.json`; raw original episodes
remain in ignored quarantine. Frozen production snapshots reject these contents
even if their IDs or candidate order change. Their four teacher passes had
agreed, illustrating why teacher consensus alone is not a correctness guarantee.
The already independently reviewed engineering seed is unchanged.

## Running and auditing

The repository environment is managed by `uv`. The tokenizer is the frozen
non-quantized Laya-multilingual tokenizer, and a local Mac Swift toolchain is used
to compile the original context projection once.

```sh
uv run python -m data_tools.produce --train-workers 4 --dev-workers 2
uv run python -m data_tools.provenance --split train --write
uv run python -m data_tools.provenance --split dev --write
```

The supervisor persists the full 20,000/1,000 targets, restarts failed work with
backoff, reports accepted/rejected counts, actual request usage and observed-rate
ETA, and freezes the 5k pilot, 20k Train and 1k Dev snapshots as each becomes ready.
It does not stop at the pilot quota. `local/production-v3/progress.json` is the
latest aggregate status; JSONL progress and process logs retain its history.

Each completed batch is resumable, and partial accepted rows and repair counters
survive failures. Generation and freezing use one candidate-ID/order-independent
content fingerprint. An atomic Train/Dev registry prevents concurrent duplicate
acceptance; recovery quarantines a later duplicate slot and regenerates that
slot. Existing frozen pilot rows cannot be changed by duplicate repair.
Raw audits and caches stay under ignored `local/`; public
manifests contain hashes, model names, parameters and usage, not credentials or
reasoning targets. Candidate IDs, episode IDs, family IDs, teacher evidence,
requested scenario type and every provenance field are excluded from student
features by the shared preprocessor.

Credentials are read into memory from `KIMI_API_KEY` or the restricted file
`~/Library/Application Support/PasteWhat/credentials/kimi.key` (mode 0600). The
client identifies itself truthfully as `PasteWhat-Ranker/0.1`. The client shares
`local/kimi-account-rate/state.json` and a process lock across Train/Dev and the
independent evaluator. At most six HTTP requests are in flight, with at least
0.5 seconds between starts; this is a conservative local setting, not a claimed
server entitlement. The state contains PID leases, times and aggregate errors,
never prompts, labels or credentials. HTTP 429 sets an account-wide cooldown and
honors numeric or HTTP-date Retry-After without capping a longer server delay.
Retryable 5xx errors use bounded exponential backoff.

Current official HTTP 403 semantics distinguish quota exhaustion from concurrent
account restriction. A 5-hour/weekly/monthly quota response pauses new requests
until Retry-After, or conservatively a complete 5-hour/7-day/31-day window from
the error if no reset time is supplied. Unknown-reset quotas, concurrent account
restrictions, other permission errors, authentication errors and membership
errors persistently block new requests pending an external account-state change.
The supervisor reports and respects that shared state instead of restarting
workers through it. Neither alternate client identities nor alternate endpoints
are used to avoid restrictions.

Current official documentation was retrieved with Context7 before implementation:
the [Kimi Code API](https://www.kimi.com/code/docs/en/) specifies the OpenAI-compatible
endpoint and `kimi-for-coding` model identifier; the
[error reference](https://www.kimi.com/code/docs/en/kimi-code/error-reference.html)
documents distinct 403 quotas/account restrictions, 429 and transient 5xx
behavior. The current primary error page was checked because the Context7 index
contained older rate-limit descriptions. The official
[Kimi provider implementation](https://github.com/moonshotai/kimi-code/blob/main/packages/kosong/src/providers/kimi.ts)
defines `thinking.type=disabled`. Actual API probes confirmed generation can use
that mode with temperature 0.6, while the default reasoning mode accepts 1.0.
Labeling, blind verification and semantic review keep reasoning enabled. API
errors and mode probes are recorded rather than silently substituted.
