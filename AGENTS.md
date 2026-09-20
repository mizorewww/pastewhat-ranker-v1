# Research isolation and execution

- Read current documentation with Context7 CLI before library-specific implementation. Resolve a library ID first; use at most three commands per distinct question. Never send credentials or private context in documentation queries.
- Use uv and a committed lockfile. Record initialization revision, weight hashes, software versions, seeds, data hashes, and code commits.
- No TDD. Design and implement first, then verify meaningful invariants. Delete temporary scaffold tests after stabilization; retain requested benchmark/data/verification tools and reports.
- Commit atomic changes. Never commit credentials, caches, real clipboard history, or model weights to Git; publish model artifacts to Hugging Face only after provenance review.
- Root owns integration, Jev, publication, and coordination. The training agent owns `src/pastewhat_ranker/`, `pyproject.toml`, `uv.lock`, training/export configs and scripts. The data agent owns teacher client, Train/Dev generation and auditing. The evaluator owns Calibration/Test generation, evaluation, calibration, and frozen acceptance reports. Coordinate shared APIs explicitly.
- Training and data-generation agents must not inspect Calibration/Test examples or labels. Root must not use final Test examples for tuning. The evaluator must not modify training logic or select checkpoints using Test.
- Split by conceptual family, including paraphrases, entity substitutions, counterfactuals, and permutations. Namespacing IDs alone is insufficient isolation. Publish a shared family partition manifest before generation.
- The same preprocessing implementation must budget and truncate input before teacher labeling and before student encoding. No teacher rationale, ID semantics, family ID, label, or inferred intent may enter student features.
- Model context contains applicationCategory, inputSurface, fieldRole, fieldLabel, selectedText, surroundingText, hasAccessibility, isSecure. No appName, bundleID, PID, or windowTitle in model features. Candidate metadata is kind, capabilities, sourceCategory. Candidate IDs only map outputs.
- Preserve all 1–20 candidates during ranker training. Mask padding from group softmax. Secure or empty requests bypass inference.
- Train full encoder, candidate head, and group-aware abstain head. Use multi-positive group logsumexp loss. No LoRA, RL, generated paste content, or chain-of-thought targets.
- Final Test runs only after weights, preprocessing and deployment calibrator/threshold are frozen. Quality targets: +5 percentage points answerable Top-1 over existing production workflow, no material key-group regression; calibration targets 95% recommendation precision with maximum observed coverage. Report misses honestly; never tune on Test or relabel to favor the model.
- Synthetic-only data supports claims only on the measured synthetic distribution. Agent review is not human validation. Inference failures remain visible in metric denominators.
