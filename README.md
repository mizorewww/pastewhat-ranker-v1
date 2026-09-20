# PasteWhat-Ranker-v1

Work in progress: a candidate-aware clipboard ranker distilled from `kimi-for-coding` into the non-quantized Laya-multilingual encoder. It scores existing clipboard candidates and may abstain; it never generates paste content.

The planned pipeline is full PyTorch fine-tuning on Apple GPU, followed by MLX FP16 deployment, calibration, and an independently held-out benchmark. Actual results and release status will be recorded as work completes. A repository or model name is not evidence that training or acceptance has completed.

Application identities are projected to categories. Only deployment-visible context and candidate data enter the model. Credentials, real clipboard history, and private contexts are excluded from this repository.
