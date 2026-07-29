# Incremental learning contract

`fit!` consumes one observation. `fit_batch!` consumes an iterable of observations
once and in iteration order. A supervised observation is an `(x, y)` tuple; an
unsupervised observation is a feature value.

Batch updates are deliberately non-transactional: each successful observation is
committed immediately. If observation `k` fails, observations before `k` remain in
the learner state. Callers requiring atomicity must validate or stage input before
calling `fit_batch!`.

This contract does not imply that learners are mergeable. It also makes no promise
that reordering observations produces the same model.

## Current capability inventory

This table is documentation, not a public trait API.

| Learner family | Incremental | Order-sensitive | Bounded memory | Windowed | New classes | Schema evolution | Merge |
|---|---:|---:|---:|---:|---:|---:|---|
| LogisticRegression/Perceptron/PassiveAggressive | yes | yes | yes, fixed dimension | no | binary labels only | no | none |
| Regression | yes | no mathematically; floating-point order effects | yes, fixed dimension | no | n/a | no | exact sufficient-statistic merge |
| Gaussian/Bernoulli/Multinomial NB | yes | mostly no | classes × features | no | yes | partial; names are not tracked | none implemented |
| Hoeffding/EFDT/adaptive trees | yes | yes | not generally bounded | no | yes | no | none |
| KNN | yes | yes | yes | yes | yes | no | none |
| StreamingKMeans | yes | yes | yes, fixed `k` and dimension | no | n/a | no | none |
| Drift detectors | yes | yes | detector-specific | detector-specific | n/a | n/a | none demonstrated |
| Adaptive ensembles | yes | yes, including RNG | member-specific | member-specific | member-specific | no | none |

Serialization currently relies on Julia object serialization and is not a stable
cross-version checkpoint format. Sample weights are not part of the common learner
contract. Distributed training is unsupported unless a specific state supplies and
tests a mathematically valid merge operation.

## LearnAPI integration

Loading LearnAPI activates an optional package extension. `learnapi` converts a
supported, unfitted OnlineML learner into a LearnAPI learner configuration:

```julia
using LearnAPI
using OnlineML
using OnlineML.Bayes: GaussianNB

learner = learnapi(GaussianNB())
model = LearnAPI.fit(learner, first_observation_stream)
LearnAPI.update_observations(model, new_observation_stream)
predictions = LearnAPI.predict(model, LearnAPI.Point(), features)
```

`update_observations` consumes only the supplied delta. It mutates and returns the
same fitted model, preserving the OnlineML state accumulated from earlier calls.
Both fitting operations consume their iterable once and in iteration order.

The extension currently supports `GaussianNB` classification and `Regression`.
The numeric type of the OnlineML configuration is preserved in the fitted state.
Passing an already-fitted OnlineML object to `learnapi` is rejected, because a
LearnAPI learner represents configuration rather than fitted state.
