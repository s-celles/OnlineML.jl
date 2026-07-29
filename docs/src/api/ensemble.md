# Ensemble Methods

Online ensemble learning algorithms.

## Bagging

```@docs
OnlineML.Ensemble.Bagging
```

## Adaptive Random Forest

```@docs
OnlineML.Ensemble.AdaptiveRandomForest
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

### Adaptive Random Forest

```julia
using OnlineML.Ensemble: AdaptiveRandomForest

# ARF automatically handles concept drift
model = AdaptiveRandomForest(
    n_estimators=10,
    lambda=6.0  # Poisson resampling weight
)

for (x, y) in data_stream
    fit!(model, (x, y))
end
```
