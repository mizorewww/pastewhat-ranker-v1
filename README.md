# PasteWhat-Ranker-v1

A candidate-aware clipboard ranker distilled into the original, non-quantized **Laya-multilingual encoder**. Data production now uses Pi's **`devin/swe-2`** under an explicit teacher transition; earlier valid **`kimi-for-coding`** data retain their original attribution. It scores 1–20 existing clipboard entries and can abstain. It does not generate paste content or chain-of-thought targets.

**Training and data production are in progress. No trained, calibrated or accepted release is claimed yet.** The implemented model and passing engineering checks below establish that the pipeline runs, not that recommendation accuracy has improved. See [execution status](RUN_STATUS.md) for completed and remaining work.

[AppKit application](https://github.com/mizorewww/pastewhat) · [Hugging Face model repository](https://huggingface.co/aac6fef/PasteWhat-Ranker-v1) · [data production](docs/DATA_PRODUCTION.md) · [independent evaluation protocol](docs/EVALUATION_PROTOCOL.md)

## Model

Each context–candidate pair passes through the shared encoder; a new MLP scores its first-token representation. A separate abstention head sees the mean and maximum candidate representations plus `log(1 + candidate_count)`. It therefore considers the complete candidate group.

The group loss is `logsumexp(all valid candidate and abstain scores) - logsumexp(acceptable actions)`, averaged once per episode. Multiple genuinely interchangeable candidates can all be positive. Padding never enters the action distribution. The upstream typed-decision/action heads and calibration are removed; all encoder parameters are strictly loaded and then fully fine-tuned. No LoRA, QLoRA, RL or training from scratch is used.

| Initialization | Frozen value |
|---|---|
| Source | `convaiinnovations/laya-multilingual` |
| Revision | `052592a15d198d9ad47da779604259b10b47b7aa` |
| Encoder | mmBERT / ModernBERT, 22 layers, hidden size 768 |
| Encoder parameters | 306,939,648 |
| New head parameters | 591,106 |
| Original storage | FP16, non-quantized upstream weights |
| Training arithmetic | FP32 master parameters and optimizer state, MPS BF16 autocast |
| Deployment | MLX FP16, separately verified and calibrated |

Weight/config/tokenizer hashes and the exact encoder loading audit are in [initialization provenance](provenance/initialization.json). The original encoder's FP32 hidden states match the adapted encoder exactly on the recorded preservation check.

## Input and data isolation

Inputs contain `applicationCategory`, observable focused-field metadata, selected/surrounding text, and candidate text, kind, payload capabilities and source category. Real app name, bundle ID, PID and window title do not enter the student. Candidate IDs map output scores only; their names, family IDs, labels and teacher audit text never become features.

The shared [preprocessor](src/pastewhat_ranker/preprocess.py) budgets each pair to at most 1,024 tokens: 448 context, 512 candidate text, and 64 metadata/special tokens. Synthetic contexts first pass through the [actual production Swift projection](tools/context_projection/README.md), then token budgeting, **then** blind teacher labeling. The teacher cannot use evidence that was truncated away from the student.

Three agents own separate work streams: Train/Dev data; model training/export; and Calibration/Test/evaluation. Conceptual families are assigned before generation. Paraphrases, entity substitutions, counterfactuals and permutations remain within a partition. The training code never inspects Calibration/Test. The efficient production protocol uses batched authorship followed by native projection, token budgeting and one independent compact blind-label pass. Programmatic checks precede labeling. A preselected 10% sample and uncertain cases receive an additional blind review; manifests distinguish single-pass and reviewed examples. Retries are bounded and disagreements are quarantined rather than silently declared correct. This is agent/teacher review, **not human validation**.

| Partition / stage | Target episodes | Purpose |
|---|---:|---|
| Train | 20,000 | Weight updates; nested 5k and 10k subsets measure the learning curve |
| Dev | 2,000 | Checkpoint and training decisions |
| Calibration | 2,000 | Separate 1,000-example fit and 1,000-example threshold selection |
| Test | 3,000 | Final frozen paired evaluation only |
| Hard-example round | 5,000 new + 5,000 original | One additional training round, accepted on Dev evidence |

These are targets for `ranker-v1-efficient-20260921`, not completed counts. The unique main/release target is 32,000 episodes, including 5,000 new hard examples; pilot subsets and reused original examples are not counted twice. The larger hard-mining proposal pool is separate. [The immutable run plan](configs/run_plan_efficient.json), [sample-size rationale](reports/data/sample-size-decision.json), and [size calculation](docs/DATA_SCALE.md) record the scope. 20k is a practical starting estimate for adapting the pretrained 307M encoder, not a measured optimum. Expansion to 50k depends on meaningful fixed-Dev learning-curve improvements and added task coverage, never final Test scores.

The earlier 1,000/200/400/600 run is preserved in [its original plan](configs/run_plan.json). It was superseded before formal training or Test scoring after costly authoring/labeling iterations; its remaining 39/9/2/10 rows are historical engineering data and do not count toward the new run. All 68 conceptual families remain allocated. The new Test target is 250 episodes per family, with 175 scheduled selectable cases. Quality gates remain unchanged; inadequate group evidence is reported as inconclusive. Frozen manifests record actual accepted counts, hashes, lineage, audit coverage and the absence of human validation. Model inference failures remain visible in evaluation denominators.

## Reproduce the engineering setup

The measured machine is an Apple M3 Max with 128 GB unified memory. Use the committed uv lockfile and Python 3.12:

```bash
uv sync --extra eval
uv run pastewhat-ranker-init --output checkpoints/initial
```

Initialization resolves only the frozen upstream revision and checks weight/config/tokenizer hashes. If it is already cached locally, `--source /path/to/original/checkpoint` avoids a download. The initializer rejects rounded MLX exports as initialization sources.

New teacher calls use the existing Pi/Devin login through an isolated, single-completion bridge. [The teacher transition](docs/PI_TEACHER.md) documents the exact runtime/model mapping, retained Kimi data, per-role source attribution and token accounting. No workspace context or tools enter teacher requests. Kimi production was retired after the user's provider switch; its existing credentials and quota state are not used as an automatic fallback. Never put credentials in command-line arguments, this repository or dataset artifacts. Both teachers are rolling services; actual responses, model strings and observed token usage are audited.

The first actual full-encoder engineering run has passed: 32 independently reviewed Train episodes were fitted in six epochs / 24 updates, with 32/32 correct decisions in both PyTorch and the converted MLX FP16 model. The maximum cross-backend score difference was 0.0282. This is **same-training-set fit**, not benchmark accuracy; the checkpoint is not a publishable ranker. [The complete report](reports/training/overfit-and-mlx-parity.json) includes counts, timings, hashes and per-episode conversion differences. The short-sample throughput measurement selected micro-batch 4 with effective batch 16; realistic training duration must be remeasured as full-length examples arrive.

Training stages are described by [overfit](configs/overfit.yaml), [pilot](configs/pilot.yaml), [main](configs/main.yaml) and [hardening](configs/hardening.yaml) configurations. The pipeline waits for independently reviewed immutable Train/Dev snapshots:

```bash
uv run python scripts/train_pipeline.py --run-plan configs/run_plan_efficient.json
```

The existing 32-episode full-encoder overfit proof is retained. The pipeline measures practical micro-batch sizes on representative new Train data, runs the 5k pilot and 10k diagnostic from identical seed 42 initialization, then independently initializes 20k main runs for seeds 42/43/44. A fixed Dev set measures the 5k/10k/20k learning curve; final checkpoint selection uses Dev only. Each seed has separately initialized task heads and identical upstream encoder weights; see [seed provenance](reports/training/seed-initializations.json).

A new 10,000-episode training pool supplies up to 6,000 nominations for blind review. The hard-example round requires 5,000 accepted new episodes mixed with 5,000 original Train episodes. The pipeline exports the Dev-selected candidate and stops at the independent calibration handoff. Progress, source/data hashes, RNG state and optimizer checkpoints support recovery. Every formal snapshot and stage is bound to one run-plan hash; a partial dataset cannot satisfy a full stage. GPU work runs serially.

`uv run python -m tools.mine_training_pool --help` describes the later Train-only hard-example proposal tool. It rejects reused original examples, binds inference to the Dev-selected v0, and prioritizes disagreements and close decisions for **blind teacher review**. A disagreement is not automatically labeled as a student error, and its output cannot be used as a frozen training snapshot.

## Deployment and acceptance

After training, the evaluator verifies PyTorch → MLX parity on a fixed regression set and measures 1/5/10/20-candidate latency/memory. Calibration fits a small logistic model using score margins, candidate count and missing-context status. Threshold selection targets observed 95% recommendation precision while maximizing coverage on its separate calibration partition. A high score is not inherently a probability of correctness.

The final Test is evaluated only after the weights, precision, preprocessing and policy are frozen. The preregistered quality target is +5 percentage points in actual answerable Top-1 versus the existing PasteWhat workflow, without a material key-family regression. Precision, coverage, abstention errors, raw ranking, latency and failures are also reported; mostly returning null cannot be presented as a successful ranker. Jev is an additional remote comparison, not the primary acceptance comparator or label source.

The [AppKit adapter](https://github.com/mizorewww/pastewhat/blob/main/engine/ranker.py) passes all candidates through the same model preprocessor and calibration function. It refuses incomplete or mismatched artifacts and calibration policies that did not meet the registered observed target. The final model deliverable will include reference weights, MLX weights, tokenizer, configuration, preprocessing, calibrator, training/data manifests, metrics and model card. Repository creation alone does not establish that this deliverable exists.

After final evaluation, `uv run python -m tools.package_release --help` describes the artifact assembler. It checks completion of the pilot, all three main seeds and the hardening round, along with their exact optimizer updates, initialization/data hashes and Dev-only selection. These aggregate training records are frozen before Test and bundled with a provenance index. The assembler also checks conversion and calibration hashes, full split counts, frozen evaluation provenance, parity and measured deployment performance before copying an explicit inference-file allowlist to a new bundle. It never modifies the frozen inference directory. If quality targets are not met, an explicitly requested diagnostic package is labeled as a research candidate; initialized or engineering-only checkpoints cannot be packaged as a release.

The durable `uv run python -m evaluations.release_pipeline --run-plan configs/run_plan_efficient.json` waits for all registered training and heldout proofs before independently verifying, calibrating, freezing and evaluating the final model. Its completed, hash-bound handoff is consumed by `uv run python -m tools.publish_pipeline --run-plan configs/run_plan_efficient.json`. The publisher rechecks that evidence, uploads the explicit bundle to the existing public Hugging Face repository in one commit, and verifies every remote file hash at the returned revision. It checks the AppKit worker with public unlabeled inputs when calibration is usable and enables the app only if all release criteria pass. A research publication retains the current app backend. Waiting processes do not imply a completed release; their private status records distinguish readiness, validation, publication and failures.

All training/calibration/test data in this version are synthetic. Their results, once measured, apply only to the tested synthetic distribution. Real-user accuracy and independent human label agreement remain unmeasured.

## License and attribution

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [initialization provenance](provenance/initialization.json) for Laya, mmBERT/ModernBERT and the existing MLX/Core ML implementations. Credentials, real clipboard history and private application context are not part of this repository.
