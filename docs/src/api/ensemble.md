# Ensemble Methods

Online ensemble learning algorithms.

## Bagging

```@docs
OnlineML.Ensemble.Bagging
```

## Drift-Aware Bagging

```@docs
OnlineML.Ensemble.DriftAwareBagging
```

## Leveraging Bagging

```@docs
OnlineML.Ensemble.LeveragingBagging
```

## Examples

### Basic Bagging

```julia
using OnlineML.Ensemble: Bagging
using OnlineML.Linear: LogisticRegression

# Create ensemble of 10 logistic regression models
model = Bagging(() -> LogisticRegression(), n_estimators=10)

for (x, y) in data_stream
    fit!(model, (x, y))
end

y_pred = predict(model, x_new)
```

### Drift-Aware Bagging

```julia
using OnlineML.Drift: DDM
using OnlineML.Ensemble: DriftAwareBagging
using OnlineML.Trees: HoeffdingTree

# Each base learner is monitored and replaced after detected drift.
model = DriftAwareBagging(
    () -> HoeffdingTree(),
    () -> DDM(),
    n_estimators=10,
    lambda=6.0  # Poisson resampling weight
)

for (x, y) in data_stream
    fit!(model, (x, y))
end
```
