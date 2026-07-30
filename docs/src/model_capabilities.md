# Model capabilities and maturity

OnlineML implements incremental algorithms and mutable learned state. LearnAPI
adapters expose generic learning contracts, while MLJ is responsible for machines,
general composition, evaluation, and tuning. A capability listed here describes
verified OnlineML behavior; it does not introduce a public trait.

The reference set is deliberately small. New algorithms should not be
treated as reference models until they satisfy the same lifecycle and
model-specific contract tests.

## Reference capability matrix

| Model | Order-sensitive | Mergeable state | Bounded state for a fixed schema | New classes | Schema evolution |
|:--|:--|:--|:--|:--|:--|
| `Regression` | No, apart from floating-point effects | Yes, through `OnlineStats.LinReg` | Yes | Not applicable | No |
| `LogisticRegression` | Yes | No | Yes | No; binary labels only | No |
| `Perceptron` | Yes | No | Yes | No; binary labels only | No |
| `PassiveAggressive` | Yes | No | Yes | No; binary labels only | No |
| `GaussianNB` | No, apart from floating-point effects | Not exposed | Yes, proportional to classes × features | Yes | Partial; trailing features can be added |
| `BernoulliNB` | No, apart from floating-point effects | Not exposed | Yes, proportional to classes × features | Yes | Partial; trailing features can be added |
| `MultinomialNB` | No, apart from floating-point effects | Not exposed | Yes, proportional to classes × features | Yes | Partial; trailing features can be added |

All seven reference models support observation-wise incremental updates, delta
mini-batches through `fit_batch!`, single-pass non-reiterable input, and reset.
None currently supports sample weights.

These claims are exercised in
`test/contract/test_reference_models.jl`. In particular, the tests verify
delta-only updates, one-pass consumption, prediction without mutation, reset,
regression-state merging, Naive Bayes class discovery, and the order sensitivity
of the three online linear classifiers.

## Maturity interpretation

The reference models are integration targets, not declarations that every
statistical or numerical property is production-qualified:

- `Regression` delegates its sufficient statistics and merging behavior to
  `OnlineStats.LinReg`.
- `LogisticRegression`, `Perceptron`, and `PassiveAggressive` own
  order-sensitive state and must not be merged by averaging parameters.
- The Naive Bayes models discover integer classes incrementally. Their schema
  evolution is limited to appending trailing features; column reordering,
  removal, and semantic type changes are not supported.

The tree learners, ensembles, and drift detectors remain experimental until
their behavior is covered by corresponding capability entries and contract
tests. General pipelines, resampling, tuning, and batch evaluation belong to MLJ
rather than this capability layer.
