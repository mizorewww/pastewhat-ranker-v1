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
- Native Swift context projection shared with the generator. Teacher labeling uses the same budgeted representation as the student.
- Real Jev API integration in [PasteWhat](https://github.com/mizorewww/pastewhat), with a documented 120-case synthetic regression; it is separate from this model's final Test.

Currently executing:

- Native Kimi episode generation, production projection/truncation, independent decision labeling, permutation verification and task-family review. Initial teacher errors were identified and quarantined before release to training.
- Full Train/Dev data production and release. The persistent training pipeline has passed the small-sample gate and is waiting for the frozen 5,000-episode pilot and 1,000-episode Dev snapshots before beginning the next stage.
- Independent Calibration/Test creation and benchmark tooling. No final Test model scores have been inspected.

Still required for the new model release:

1. Complete actual data quotas and freeze manifests, rather than report attempted or template-expanded counts as validated examples.
2. Finish pilot, main training and the registered seed comparison; select on Dev only.
3. Perform a new training-pool hard-example round and accept it only on Dev evidence.
4. Export the selected model to MLX FP16, measure parity/latency/memory, fit and select the deployment calibration policy using Calibration only.
5. Freeze artifacts and threshold, then run the independent final Test paired with the existing PasteWhat workflow.
6. Publish weights, tokenizer, preprocessing, calibration, provenance and honest metrics/model limitations as one deliverable.

The app's dedicated ranker adapter already exists, but it refuses uncalibrated or mismatched artifacts. A repository, initialized model or passing engineering check does not establish ranking quality. Synthetic-only results will be labeled as such.
