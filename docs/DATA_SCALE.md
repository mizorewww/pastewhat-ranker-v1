# Choosing the first training-set size

The first release targets **20,000 base Train episodes plus 5,000 new hard
episodes**. Dev has 2,000 episodes, Calibration 2,000, and the independent final
Test 3,000. That is **32,000 distinct release-data episodes**. The 5k pilot and
10k diagnostic are nested subsets of the 20k Train set; the 5k original examples
reused during hardening are not additional data. Hard-pool proposals that are not
accepted for training are counted separately.

These are registered targets, not completed production counts or a measured
optimum. The user clarified that 100k was illustrative rather than a minimum.
The machine-readable calculation and immutable run binding are in
[the size decision](../reports/data/sample-size-decision.json) and
[the run plan](../configs/run_plan_efficient.json).

## What the model size tells us

The actual initialization contains 306,939,648 encoder parameters. Of those,
196,608,000 are word embeddings and 110,331,648 are the rest of the encoder.
The two new task heads contain 591,106 parameters, giving **307,530,754 total**.
All encoder parameters are fine-tuned; the heads' small size does not mean that
only the heads are trained.

The [upstream mmBERT model card](https://huggingface.co/jhu-clsp/mmBERT-base)
describes the pretrained 307M encoder, including its 110M non-embedding
parameters. Existing multilingual representations make this an adaptation
problem. A parameter-to-token rule for language-model pretraining does not
determine the number of clipboard decisions needed for adaptation.

## Comparable training scales

The planning calculation assumes an average of 10.5 candidates per episode,
two full-encoder epochs, effective batch size 16 episodes, and 40 Train operation
families. Actual candidate counts and sequence lengths will be measured from
the produced corpus.

| Train episodes | Mean episodes per Train family | Pairs per epoch | Pair presentations over two epochs | Full-encoder updates per seed |
|---|---:|---:|---:|---:|
| 5,000 | 125 | 52,500 | 105,000 | 626 |
| 10,000 | 250 | 105,000 | 210,000 | 1,250 |
| 20,000 | 500 | 210,000 | 420,000 | 2,500 |
| 50,000 | 1,250 | 525,000 | 1,050,000 | 6,250 |
| 100,000 | 2,500 | 1,050,000 | 2,100,000 | 12,500 |

Each run also performs the registered 200 head-warmup updates. A context paired
with ten candidates is still one decision, not ten independent training
episodes. Candidate permutations and entity substitutions do not create new
operation families.

20k provides room for about 500 distinct situations per existing Train family,
including the registered mix of select, no-match, and missing/ambiguous intent.
The additional 5k examples target verified weaknesses on new Train-only
situations. This is a practical starting allocation, not a statistical proof
that every family needs exactly 500 examples.

The [InPars-v2 experiments](https://arxiv.org/html/2301.01820v4) demonstrate
reranker adaptation with 10k synthetic positive and 10k negative pairs per
collection. Their starting model and task differ from this project, so that
result supports trying moderate data sizes, not transferring an optimum to
PasteWhat. The [Sentence Transformers training overview](https://sbert.net/docs/cross_encoder/training_overview.html)
also emphasizes cross-encoder supervision and evaluation; its example dataset
sizes are not sample-complexity guarantees.

## How further expansion is decided

The pipeline compares 5k, 10k, and 20k nested Train subsets using the same seed
42 initialization, training recipe, and fixed 2k Dev set. It reports ranking,
abstention, group behavior, and the actual amount of training. Main training
additionally uses seeds 42, 43, and 44 to expose variation at the final scale.
This learning curve gives evidence about useful additional data; a single
small change in a point estimate is not proof of improvement.

Further expansion to 50k is justified when Dev improvements remain material
and new data adds useful situations or addresses verified failure groups.
If the curve flattens, the next work is on label quality, missing coverage, and
hard examples. The final Test is never used to choose the data size, adjust
training, or mine failures. There is no current evidence that 100k is required.

Teacher cost is tracked over all author, label, review, invalid, and repair
requests. Counts distinguish generated drafts, accepted episodes, and frozen
training data. Token totals are not converted into an account charge without
the applicable billing evidence. Training duration will use real candidate
counts and token lengths; the earlier short engineering examples do not supply
a reliable full-corpus time estimate.
