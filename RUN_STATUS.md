# Execution status

The persistent task is active. **No trained/accepted release is claimed by this status file.**

Completed foundations:

- Public [GitHub repository](https://github.com/mizorewww/pastewhat-ranker-v1) and [Hugging Face model repository](https://huggingface.co/aac6fef/PasteWhat-Ranker-v1), with an explicitly pending model card.
- Separate agents for Train/Dev generation, student training/export, and independent Calibration/Test/evaluation. Conceptual task families were partitioned before generation; related variants remain within their partition.
- uv-locked Python environment; original non-quantized Laya-multilingual revision and hashes verified.
- Strict encoder initialization, new candidate/group-abstention heads and multi-positive group loss; PyTorch MPS BF16 training and initial MLX FP16 export exercised.
- Preflight checks for parameter loading, gradient flow, candidate ordering, padding, save/reload and conversion agreement. Details in `reports/training/preflight.json`.
- Native Swift context projection shared with the generator. Teacher labeling uses the same budgeted representation as the student.
- Real Jev API integration in [PasteWhat](https://github.com/mizorewww/pastewhat), with a documented 120-case synthetic regression; it is separate from this model's final Test.

Currently executing:

- Native Kimi episode generation, production projection/truncation, independent decision labeling, permutation verification and task-family review. Initial teacher errors were identified and quarantined before release to training.
- Train/Dev data release, small-sample overfit and realistic MPS throughput measurement, followed by the 5k pilot and 20k main runs.
- Independent Calibration/Test creation and benchmark tooling. No final Test model scores have been inspected.

Still required for the new model release:

1. Complete actual data quotas and freeze manifests, rather than report attempted or template-expanded counts as validated examples.
2. Finish overfit, pilot, main training and the registered seed comparison; select on Dev only.
3. Perform a new training-pool hard-example round and accept it only on Dev evidence.
4. Export the selected model to MLX FP16, measure parity/latency/memory, fit and select the deployment calibration policy using Calibration only.
5. Freeze artifacts and threshold, then run the independent final Test paired with the existing PasteWhat workflow.
6. Publish weights, tokenizer, preprocessing, calibration, provenance and honest metrics/model limitations as one deliverable.

The app's dedicated ranker adapter already exists, but it refuses uncalibrated or mismatched artifacts. A repository, initialized model or passing engineering check does not establish ranking quality. Synthetic-only results will be labeled as such.
