# PasteWhat-Ranker-v1

A candidate-aware clipboard ranker distilled from **`kimi-for-coding`** into the original, non-quantized **Laya-multilingual encoder**. It scores 1–20 existing clipboard entries and can abstain. It does not generate paste content or chain-of-thought targets.

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

Three agents own separate work streams: Train/Dev data; model training/export; and Calibration/Test/evaluation. Conceptual families are assigned before generation. Paraphrases, entity substitutions, counterfactuals and permutations remain within a partition. The training code never inspects Calibration/Test. Accepted examples require two blind label passes with reordered/reidentified candidates, an independent family/deployment review, and programmatic schema/budget checks. Teacher disagreements are quarantined or regenerated, not silently declared correct. This is agent/teacher review, **not human validation**.

| Partition / stage | Target episodes | Purpose |
|---|---:|---|
| Train | 20,000 | Weight updates, including a frozen 5,000-example pilot subset |
| Dev | 1,000 | Checkpoint and training decisions |
| Calibration | 1,000 | Separate 500-example fit and 500-example threshold selection |
| Test | 2,000 | Final frozen paired evaluation only |
| Hard-example round | 5,000 new + 5,000 original | One additional training round, accepted on Dev evidence |

These numbers are targets, not completed data counts. Frozen manifests record actual accepted counts, hashes, distribution, lineage, audit coverage and the absence of human validation. Model inference failures remain visible in evaluation denominators.

## Reproduce the engineering setup

The measured machine is an Apple M3 Max with 128 GB unified memory. Use the committed uv lockfile and Python 3.12:

```bash
uv sync --extra eval
uv run pastewhat-ranker-init --output checkpoints/initial
```

Initialization resolves only the frozen upstream revision and checks weight/config/tokenizer hashes. If it is already cached locally, `--source /path/to/original/checkpoint` avoids a download. The initializer rejects rounded MLX exports as initialization sources.

Teacher credentials are supplied through `KIMI_API_KEY` or the private local credential file described in [data production](docs/DATA_PRODUCTION.md). Never put keys in command-line arguments, this repository or dataset artifacts. Kimi requests retain the client's real identity. `kimi-for-coding` is the requested model ID; real responses and token usage are audited because the service behind an alias may change.

The first actual full-encoder engineering run has passed: 32 independently reviewed Train episodes were fitted in six epochs / 24 updates, with 32/32 correct decisions in both PyTorch and the converted MLX FP16 model. The maximum cross-backend score difference was 0.0282. This is **same-training-set fit**, not benchmark accuracy; the checkpoint is not a publishable ranker. [The complete report](reports/training/overfit-and-mlx-parity.json) includes counts, timings, hashes and per-episode conversion differences. The short-sample throughput measurement selected micro-batch 4 with effective batch 16; realistic training duration must be remeasured as full-length examples arrive.

Training stages are described by [overfit](configs/overfit.yaml), [pilot](configs/pilot.yaml), [main](configs/main.yaml) and [hardening](configs/hardening.yaml) configurations. The pipeline waits for independently reviewed immutable Train/Dev snapshots:

```bash
uv run python scripts/train_pipeline.py
```

It measures practical micro-batch sizes while preserving the effective episode batch, checks overfitting on 32 reviewed training examples, runs the 5k pilot, reinitializes for 20k main runs with seeds 42/43/44, and selects only on Dev. Each main seed has separately initialized new task heads with exactly the same upstream encoder weights; see [seed provenance](reports/training/seed-initializations.json). It waits for a new training-pool hard-example round, exports the selected candidate and stops at the independent calibration handoff. Progress, source/data hashes, RNG state and optimizer checkpoints allow recovery; it never substitutes a smaller run for a planned full stage. GPU work runs serially.

`uv run python -m tools.mine_training_pool --help` describes the later Train-only hard-example proposal tool. It rejects reused original examples, binds inference to the Dev-selected v0, and prioritizes disagreements and close decisions for **blind teacher review**. A disagreement is not automatically labeled as a student error, and its output cannot be used as a frozen training snapshot.

## Deployment and acceptance

After training, the evaluator verifies PyTorch → MLX parity on a fixed regression set and measures 1/5/10/20-candidate latency/memory. Calibration fits a small logistic model using score margins, candidate count and missing-context status. Threshold selection targets observed 95% recommendation precision while maximizing coverage on its separate calibration partition. A high score is not inherently a probability of correctness.

The final Test is evaluated only after the weights, precision, preprocessing and policy are frozen. The preregistered quality target is +5 percentage points in actual answerable Top-1 versus the existing PasteWhat workflow, without a material key-family regression. Precision, coverage, abstention errors, raw ranking, latency and failures are also reported; mostly returning null cannot be presented as a successful ranker. Jev is an additional remote comparison, not the primary acceptance comparator or label source.

The [AppKit adapter](https://github.com/mizorewww/pastewhat/blob/main/engine/ranker.py) passes all candidates through the same model preprocessor and calibration function. It refuses incomplete or mismatched artifacts and calibration policies that did not meet the registered observed target. The final model deliverable will include reference weights, MLX weights, tokenizer, configuration, preprocessing, calibrator, training/data manifests, metrics and model card. Repository creation alone does not establish that this deliverable exists.

After final evaluation, `uv run python -m tools.package_release --help` describes the artifact assembler. It checks training completion, conversion and calibration hashes, full split counts, frozen evaluation provenance, parity and measured deployment performance before copying an explicit inference-file allowlist to a new bundle. It never modifies the frozen inference directory. If quality targets are not met, an explicitly requested diagnostic package is labeled as a research candidate; initialized or engineering-only checkpoints cannot be packaged as a release.

All training/calibration/test data in this version are synthetic. Their results, once measured, apply only to the tested synthetic distribution. Real-user accuracy and independent human label agreement remain unmeasured.

## License and attribution

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [initialization provenance](provenance/initialization.json) for Laya, mmBERT/ModernBERT and the existing MLX/Core ML implementations. Credentials, real clipboard history and private application context are not part of this repository.
