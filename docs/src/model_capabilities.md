# Model capabilities and maturity

OnlineML implements incremental algorithms and mutable learned state. LearnAPI
adapters expose generic learning contracts, while MLJ is responsible for machines,
general composition, evaluation, and tuning. A capability listed here describes
verified OnlineML behavior; it does not introduce a public trait.

The initial reference set is deliberately small. New algorithms should not be
treated as reference models until they satisfy the same lifecycle and
model-specific contract tests.

## Reference capability matrix

| Capability | `Regression` | `LogisticRegression` | `GaussianNB` |
|:--|:--|:--|:--|
| Observation-wise incremental update | Yes | Yes | Yes |
| Delta mini-batches through `fit_batch!` | Yes | Yes | Yes |
| Single-pass, non-reiterable input | Yes | Yes | Yes |
| Resettable | Yes | Yes | Yes |
| Order-sensitive | No, apart from floating-point effects | Yes | No, apart from floating-point effects |
| Mergeable state | Yes, through `OnlineStats.LinReg` | No | Not exposed |
| Bounded state for a fixed schema | Yes | Yes | Yes, proportional to classes × features |
| New classes | Not applicable | No; binary labels only | Yes |
| Schema evolution | No | No | Partial; trailing features can be added |
| Sample weights | No | No | No |

These claims are exercised in
`test/contract/test_reference_models.jl`. In particular, the tests verify
delta-only updates, one-pass consumption, prediction without mutation, reset,
regression-state merging, GaussianNB class discovery, and logistic-regression
order sensitivity.

## Maturity interpretation

The three reference models are integration targets, not declarations that every
statistical or numerical property is production-qualified:

- `Regression` delegates its sufficient statistics and merging behavior to
  `OnlineStats.LinReg`.
- `LogisticRegression` owns an order-sensitive optimizer state and must not be
  merged by averaging parameters.
- `GaussianNB` discovers integer classes incrementally. Its schema evolution is
  limited to appending trailing features; column reordering, removal, and semantic
  type changes are not supported.

Other exported models remain experimental until their behavior is covered by a
corresponding capability entry and contract tests. General pipelines, resampling,
tuning, and batch evaluation belong to MLJ rather than this capability layer.
