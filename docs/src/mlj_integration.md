# MLJ integration

Loading `OnlineML` and `MLJBase` activates the experimental
`OnlineMLMLJExt` package extension. It provides MLJ model interfaces for online
logistic regression, perceptron, Gaussian naive Bayes, Hoeffding tree, and KNN.

The interfaces accept continuous Tables.jl inputs and categorical targets. They
preserve the complete categorical pool in deterministic and probabilistic
predictions and encode target levels as the integer labels required by the
OnlineML core algorithms.

## Incremental observations

`MLJBase.update` is not overloaded as an observation-delta operation. In MLJ,
its data arguments represent the machine's complete training data and the method
supports efficient retraining after model changes.

For an already fitted interface, `update_observations!` consumes only a new
mini-batch and preserves the existing OnlineML learner:

```julia
using MLJBase
using OnlineML

ext = Base.get_extension(OnlineML, :OnlineMLMLJExt)
model = ext.OnlineNaiveBayes()

X1 = MLJBase.table([0.0 0.0; 1.0 1.0])
y1 = MLJBase.categorical(["negative", "positive"])
fitresult, _, _ = MLJBase.fit(model, 0, X1, y1)

X2 = MLJBase.table([0.1 0.2; 0.9 0.8])
y2 = MLJBase.categorical(["negative", "positive"])
ext.update_observations!(model, fitresult, X2, y2)
```

Every observed target in a delta must belong to the categorical pool established
at initial fit. Unsupported classes are rejected before the learner is mutated.

## Current limitations

- Inputs are materialized with `MLJBase.matrix`; column names are not retained
  in the OnlineML learned state.
- Feature schema evolution is not supported by these interfaces.
- Sample weights are not supported.
- The extension is not yet registered in the MLJ model registry.
- The Julia fitresult can be serialized within the same runtime contract as the
  underlying OnlineML learner; no portable checkpoint format is promised.
