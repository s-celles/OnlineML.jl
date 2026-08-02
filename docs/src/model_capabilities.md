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

## Reference instance and clustering capabilities

| Model | Order-sensitive | Mergeable state | Bounded state | New classes | Schema evolution |
|:--|:--|:--|:--|:--|:--|
| `StreamingKMeans` | Yes | No | Yes, proportional to clusters × features | Not applicable | No |
| `KNN` | Yes | No | Yes, proportional to `window_size` × features | Yes, while represented in the window | No |

Both models support observation-wise updates, delta mini-batches, single-pass
input, prediction without mutation, and reset. `StreamingKMeans` retains at most
`k` centroids. `KNN` retains at most `window_size` labeled observations, while
`nobs` continues to report the total number of observations consumed.

The assertions live in
`test/contract/test_instance_cluster_capabilities.jl`. They also verify
constructor constraints, explicit rejection of feature-dimension changes,
KNN window eviction, and order-sensitive state. Neither model exposes a
mathematically valid state merge, and KNN distance weighting is not equivalent
to per-observation sample weights.

## Experimental tree capabilities

| Model | Order-sensitive | Mergeable state | Bounded memory | New classes | Schema evolution |
|:--|:--|:--|:--|:--|:--|
| `HoeffdingTree` | Yes | No | Only with a finite `max_depth` | Yes | Partial; trailing features can be added |
| `ExtremelyFastTree` | Yes | No | Only with a finite `max_depth` | Yes | Partial; trailing features can be added |
| `HoeffdingAdaptiveTree` | Yes | No | Only with a finite `max_depth` | Yes | Partial; trailing features can be added |

These tree learners support observation-wise updates, delta mini-batches,
single-pass input, and reset. The lifecycle assertions live in
`test/contract/test_tree_capabilities.jl`. They remain experimental because
these assertions do not yet qualify split correctness, drift adaptation,
long-run memory behavior, or predictive quality. Tree states must not be merged,
and parameter averaging is not a valid distributed training strategy.

## Experimental drift-detector capabilities

| Detector | Input | Retained state | Warning | Automatic state reset | Maturity |
|:--|:--|:--|:--|:--|:--|
| `ADWIN` | Finite real-valued | Exact adaptive window | No | Shrinks on drift | Cut-point lifecycle qualified |
| `DDM` | Binary error | Constant | Yes | On drift | Deterministic lifecycle qualified |
| `EDDM` | Binary error | Constant | Yes | On drift | Inter-error statistics qualified |
| `PageHinkley` | Real-valued | Constant | No | On drift | Deterministic lifecycle qualified |
| `KSWIN` | Real-valued | Sliding window bounded by `window_size` | Yes | No | Deterministic lifecycle qualified |

The deterministic assertions live in
`test/contract/test_drift_capabilities.jl`. They verify reset and status
invariants, DDM behavior on a perfect error stream followed by an abrupt error
increase, Page-Hinkley behavior after an internal reset, and KSWIN window
bounds and response to a separated deterministic distribution.

`ADWIN` evaluates every admissible cut point using a variance-sensitive bound
and discards obsolete prefixes. It retains exact observations rather than the
paper's bucket-compressed optimization; its memory therefore follows the
adaptive window and can grow on a stationary stream. EDDM's running mean and
population standard deviation are qualified against explicit inter-error
distances; its warning/drift timing still lacks a cross-implementation oracle.
For DDM, EDDM, and Page-Hinkley, `nobs` describes the
current post-reset segment because their state is cleared after a detected
drift. For ADWIN, `nobs` is the lifetime count and `window_size` is the retained
adaptive window.

Detector states are order-sensitive and do not expose a merge operation.
Merging detector parameters or summaries would not reproduce processing the
original event order.

## Experimental ensemble capabilities

| Ensemble | Order-sensitive | Mergeable state | Memory bound | Drift adaptation | Maturity |
|:--|:--|:--|:--|:--|:--|
| `Bagging` | Yes, stochastic | No | Depends on base learners | No | Lifecycle qualified |
| `HighRatePoissonBagging` (`LeveragingBagging` alias) | Yes, stochastic | No | Depends on base learners | No | Lifecycle-qualified Poisson resampling; not Leveraging Bagging |
| `DriftAwareBagging` (`AdaptiveRandomForest` alias) | Yes, stochastic | No | Depends on base learners and background models | Yes | Lifecycle qualified; not ARF |

The assertions in `test/contract/test_ensemble_capabilities.jl` verify
constructor constraints, one-pass delta consumption, prediction without
training-state mutation, normalized probability aggregation, and prequential
test-then-train errors for drift detectors. Supplying independent RNG instances
with identical initial states gives reproducible stochastic updates. `reset!`
clears learned models and counters but does not rewind the RNG stream.

`HighRatePoissonBagging` implements higher-rate Poisson resampling but not
the random output codes, per-learner ADWIN detectors, or worst-learner
replacement of Leveraging Bagging. The historical `LeveragingBagging` name is
retained as an alias and must not be interpreted as algorithmic conformance.
`DriftAwareBagging` adds
per-estimator detectors and background-model replacement, but it does not
implement random feature subspaces. The historical `AdaptiveRandomForest` name
is retained as an alias and must not be interpreted as full algorithmic
conformance.

Ensemble states must not be merged or replaced by averaged parameters.
Distribution can assign independently owned estimators to workers only with an
explicit prediction-aggregation and checkpoint protocol.

## Experimental anomaly-detector capabilities

| Detector | Order-sensitive | Mergeable state | Memory bound | Score contract | Maturity |
|:--|:--|:--|:--|:--|:--|
| `HalfSpaceTrees` | Yes, stochastic | No | Fixed trees and depth | Normalized to `[0, 1]` | Lifecycle qualified |
| `GaussianProjectionDetector` (`LODA` alias) | Yes, stochastic | No | Projections × features | Normalized to `[0, 1]` | Lifecycle qualified Gaussian approximation |
| `RandomCutForestApproximation` (`RobustRandomCutForest` alias) | Yes, stochastic | No | Trees × `tree_size` | Raw tree score of retained points | Non-conforming approximation |

The assertions in `test/contract/test_anomaly_capabilities.jl` verify
constructor and fixed-schema constraints, one-pass delta consumption,
non-mutating scoring, reset, deterministic construction from explicit RNG
state, and configured memory bounds. `reset!` clears learned state but does not
rewind an RNG stream.

`GaussianProjectionDetector` uses dense Gaussian random projections and running
variance. The legacy `LODA` binding is only a compatibility alias; it does not
claim conformance with LODA's sparse projections, adaptive histogram density
estimator, or log-density score.
`RandomCutForestApproximation.score` can only score a point retained in its
sliding window and throws for an absent point; callers must use `fit_score!` to
insert and then score a new point. Its leaves do not retain point coordinates,
insertion does not implement the geometric RRCT construction, and its score is
not independently qualified as RRCF codisplacement. The historical
`RobustRandomCutForest` binding is only a compatibility alias.

Anomaly-detector states are order-sensitive and expose no mathematically valid
merge. Distributed execution therefore requires single-owner partitioning or
explicit ensemble aggregation, not parameter averaging.

Julia `Serialization` round trips preserve representative ensemble and anomaly
states, including explicit RNG continuation, within the same Julia runtime. This
is exercised in `test/contract/test_checkpoint_capabilities.jl`; it is not a
portable or version-stable checkpoint format.

General pipelines, resampling, tuning, and batch evaluation belong to MLJ
rather than this capability layer.
