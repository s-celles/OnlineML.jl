# Linear Models

Linear models for online regression and classification.

## Regression

```@docs
OnlineML.Linear.Regression
```

## Classification

```@docs
OnlineML.Linear.LogisticRegression
OnlineML.Linear.Perceptron
OnlineML.Linear.PassiveAggressiveClassifier
OnlineML.Linear.PassiveAggressiveRegressor
```

## Examples

### Logistic Regression

```julia
using OnlineML.Linear: LogisticRegression

model = LogisticRegression(lr=0.01, lambda=0.001)

for (x, y) in data_stream
    fit!(model, (x, y))
end

y_pred = predict(model, x_new)
probs = predict_proba(model, x_new)
```

### Perceptron

```julia
using OnlineML.Linear: Perceptron

model = Perceptron(lr=1.0)

for (x, y) in data_stream
    fit!(model, (x, y))
end
```
