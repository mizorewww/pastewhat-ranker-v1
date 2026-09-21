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
   candidate ordering and candidate IDs changed. The two complete action sets
   and abstention reasons must agree after remapping. Disagreements are retained
   in audit and excluded; new episodes replace them. A majority vote never
   silently changes a disputed label.
6. A separate teacher call reviews conceptual-family adherence, realistic field
   visibility and payload metadata. It sees no proposed decision label. Schema,
   candidate counts, native projection, preprocessing idempotence, visible hashes
   and label membership are also checked by the production tools.
7. Only accepted episodes enter the current Train/Dev JSONL. Training uses frozen
   snapshots, never a file that the generator is still updating. The 5,000 pilot
   rows must be a recorded subset of the final 20,000 Train rows. Any hard-case
   augmentation is restricted to new examples from Train operation families and
   requires a frozen ranker-v0; Test is never an error-mining source.

The initial, unreleased single-label probe exposed a wrong teacher ID selection,
out-of-family operations, and impossible field metadata. It was isolated under
ignored `local/initial-v1-unreleased/` before training. The stronger pipeline is
named `teacher-episodes-v2-blind-consensus`. This inspection was performed by an
agent, not a human. Repeated teacher agreement and programmatic checks reduce
errors; they do not establish that every synthetic label is correct.

## Running and auditing

The repository environment is managed by `uv`. The tokenizer is the frozen
non-quantized Laya-multilingual tokenizer, and a local Mac Swift toolchain is used
to compile the original context projection once.

```sh
uv run python -m data_tools.generate --split train --limit 20000 --workers 4
uv run python -m data_tools.generate --split dev --limit 1000 --workers 2
uv run python -m data_tools.provenance --split train --write
uv run python -m data_tools.provenance --split dev --write
```

Each completed batch is resumable, and partial accepted rows survive semantic
repair attempts. Raw audits and caches stay under ignored `local/`; public
manifests contain hashes, model names, parameters and usage, not credentials or
reasoning targets. Candidate IDs, episode IDs, family IDs, teacher evidence,
requested scenario type and every provenance field are excluded from student
features by the shared preprocessor.

Credentials are read into memory from `KIMI_API_KEY` or the restricted file
`~/Library/Application Support/PasteWhat/credentials/kimi.key` (mode 0600). The
client identifies itself truthfully as `PasteWhat-Ranker/0.1`. HTTP 429 and
retryable 5xx errors use bounded exponential backoff and numeric Retry-After;
authentication failures do not trigger prompt-repair loops. No client spoofing
or rate-limit circumvention is used.

Current official documentation was retrieved with Context7 before implementation:
the [Kimi Code API](https://www.kimi.com/code/docs/en/) specifies the OpenAI-compatible
endpoint and `kimi-for-coding` model identifier; the
[error reference](https://www.kimi.com/code/docs/en/kimi-code/error-reference.html)
documents 429 and transient 5xx behavior. The official
[Kimi provider implementation](https://github.com/moonshotai/kimi-code/blob/main/packages/kosong/src/providers/kimi.ts)
defines `thinking.type=disabled`. Actual API probes confirmed generation can use
that mode with temperature 0.6, while the default reasoning mode accepts 1.0.
Labeling, blind verification and semantic review keep reasoning enabled. API
errors and mode probes are recorded rather than silently substituted.
