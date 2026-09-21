# Engineering-only independent export verification

This report checks the evaluator's numerical regression pipeline on the
32-episode engineering-overfit checkpoint. It is **not** final release acceptance,
Calibration inference, Test inference, or a measurement of generalization accuracy.
No training, Calibration, or Test examples were used as regression inputs.

The inputs are the 16 previously fixed, unlabeled records in
`evaluations/regression-inputs.jsonl`, covering 1, 5, 10, and 20 candidates,
multiple text lengths and languages, and identical-payload alternatives.
The training agent stopped its background pipeline and explicitly reserved the
GPU for this run. PyTorch MPS and MLX ran in separate sequential processes.

All predeclared checks passed. The maximum cross-backend score difference was
0.0228582621 and the mean was 0.0028327425. The raw action agreed on all 16
inputs, allowing an alternative ID only when its complete payload and metadata
were identical. Candidate permutation, extra padding, mixed candidate counts,
padding exclusion, and exact artifact reload checks passed.

`run-artifacts.json` binds the exact reference, MLX, preprocessing, runtime source,
and regression input files used. The reference weight SHA-256 starts with
`6ddd68f812a806a5`; the MLX weight SHA-256 starts with `959ff84fbb40ef18`.
Full hashes and individual numerical results are retained in the JSON artifacts.

The selected final release checkpoint must undergo the same fixed numerical
verification again. Its frozen calibrator must additionally preserve deployment
decisions across the two backends; that check has not been run for this engineering
checkpoint and cannot be inferred from these results.
