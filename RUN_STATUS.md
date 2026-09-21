# Execution status

The persistent task is active. **No trained/accepted release is claimed by this status file.**

Completed foundations:

- Public [GitHub repository](https://github.com/mizorewww/pastewhat-ranker-v1) and [Hugging Face model repository](https://huggingface.co/aac6fef/PasteWhat-Ranker-v1), with an explicitly pending model card.
- Separate agents for Train/Dev generation, student training/export, and independent Calibration/Test/evaluation. Conceptual task families were partitioned before generation; related variants remain within their partition.
- uv-locked Python environment; original non-quantized Laya-multilingual revision and hashes verified.
- Strict encoder initialization, new candidate/group-abstention heads and multi-positive group loss; PyTorch MPS BF16 training and initial MLX FP16 export exercised.
- Preflight checks for parameter loading, gradient flow, candidate ordering, padding, save/reload and conversion agreement. Details in `reports/training/preflight.json`.
- Independently reviewed and frozen 32-episode **Train-only engineering set**, with 21 selectable cases, 11 abstentions, four multi-positive cases and 1–20 candidates. Root review rejected six initial proposals and approved six independently regenerated replacements; no labels were manually changed.
- Actual full-encoder MPS training fitted those 32 examples in six epochs / 24 optimizer updates (62.49 seconds including evaluation/checkpoint work). The learned checkpoint's MLX FP16 decisions matched PyTorch on all 32. **This is a training-fit and conversion check, not generalization accuracy.** See `reports/training/overfit-and-mlx-parity.json`.
- Real short-sample throughput selected micro-batch 4 with effective batch 16, approximately 0.127 seconds per episode. The measured MPS driver allocation was about 8.25 GB of unified memory; full-dataset throughput remains to be measured.
- Separate untrained initialization snapshots for seeds 42/43/44: all 134 encoder tensors match exactly, while new-head weights differ. Each formal run will independently vary head initialization, episode shuffle and dropout.
- Native Swift context projection shared with the generator. It now preserves actual UTF-16 insertion boundaries and bounded adjacent static text (`pastewhat-focus-v1`); synthetic authoring can use explicit before/selected/after fragments, compiled without changing any text. Teacher labeling sees only the same budgeted representation as the student. See `tools/context_projection/README.md` and `reports/engineering/context-projection.json`.
- Native Swift candidate projection shared with the app at commit `737c736`: authors supply text, synthetic file-name fixtures or actual PNG fixtures, and the deployed codec derives text, kind and capabilities. Image dimensions are read from bytes rather than accepted as an author's claim. App debug/release builds and 35 actual-payload checks passed; temporary scaffolds were removed. The authoring adapter also passed mapping checks for 1/5/10/20 candidates, real file/PNG projection and invalid-fixture rejection. See `reports/engineering/candidate-projection.json`.
- A 20-case Train-only paired teacher-effort probe found identical, independently reviewed action sets for `high` and `max`. Across four requests per condition, `high` used 2,336 completion tokens versus 4,054 and summed request duration was 51.50 versus 84.12 seconds. The approved choice is `high`. This small, explicit engineering set does not establish broader teacher quality or production throughput. Its candidate fields were replayed through the native payload adapter and remained identical. See `data/train_teacher_effort.report.json`.
- An independent evaluator separately checked the trained engineering checkpoint on 16 fixed, unlabeled conversion inputs: all raw decisions agreed, maximum absolute score difference 0.0228583, mean difference 0.00283274. Permutation, padding, mixed-candidate masking and reload checks passed. Final deployment weights must repeat this verification; this remains an engineering check, not Test accuracy.
- Real Jev API integration in [PasteWhat](https://github.com/mizorewww/pastewhat), with a documented 120-case synthetic regression; it is separate from this model's final Test.

Currently executing:

- The revised run is `ranker-v1-efficient-20260921`, registered in `configs/run_plan_efficient.json`: 20,000 Train, 2,000 Dev, 2,000 Calibration, 3,000 Test, plus 5,000 accepted new hard examples. Nested 5k/10k Train subsets provide a learning curve before judging whether expansion to 50k is worthwhile. The user clarified 100k was illustrative, not a minimum. This is a planning estimate; no optimum has been measured.
- The previous run stopped before any formal training or final Test. Its net retained rows were 39 Train, 9 Dev, 2 Calibration and 10 Test. Old protocol data and usage audits remain historical; none is relabeled as new-run production.
- The earlier four-pass protocol spent most observed teacher tokens on repeated labeling and review. Version 7 is being implemented as batched authorship, native projection/student budgeting, one compact independent blind-label pass, programmatic gates, and sampled or uncertainty-triggered secondary review. Every accepted record will expose its actual audit coverage. No new corpus or model quality is claimed yet.
- Separate data, training and evaluation agents are adapting the existing pipeline. Old generators and training waiters are retired before switching run namespaces. Teacher spending will resume under the user's authorization after the new compact protocol is ready; old producers will not restart. Provider-enforced quota limits remain respected.
- The original encoder, student architecture, 1024-token preprocessing, native app projections and 32-episode engineering proof remain unchanged. Training seeds 42/43/44, full-encoder optimization,5k new/5k original hard-example round, MLX parity and final calibration/Test steps remain required.
- Test remains evaluator-only and unscored until final artifacts and recommendation policy are frozen. Independent heldout authoring retains the original conceptual-family partition and now includes missing-observation variants from the initial sampling plan.

Still required for the new model release:

1. Complete actual data quotas and freeze manifests, rather than report attempted or template-expanded counts as validated examples.
2. Finish pilot, main training and the registered seed comparison; select on Dev only.
3. Perform a new training-pool hard-example round and accept it only on Dev evidence.
4. Export the selected model to MLX FP16, measure parity/latency/memory, fit and select the deployment calibration policy using Calibration only.
5. Freeze artifacts and threshold, then run the independent final Test paired with the existing PasteWhat workflow.
6. Publish weights, tokenizer, preprocessing, calibration, provenance and honest metrics/model limitations as one deliverable.

The app's dedicated ranker adapter already exists, but it refuses uncalibrated or mismatched artifacts. A repository, initialized model or passing engineering check does not establish ranking quality. Synthetic-only results will be labeled as such.

The final training handoff now proves completion of the pilot, all three main seeds and the hardening round, including exact update/exposure counts, immutable data/configuration hashes and Dev-only selection. The evaluator freezes every referenced proof file before Test; the release packager includes those records with an index and binds their actual Train/Dev hashes to the published data manifest. These checks are implemented and verified with temporary metadata fixtures; **the formal stages themselves are still pending data completion**.
