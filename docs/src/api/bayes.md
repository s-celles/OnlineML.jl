# Naive Bayes

Online Naive Bayes classifiers.

## Gaussian Naive Bayes

```@docs
OnlineML.Bayes.GaussianNB
```

## Multinomial Naive Bayes

```@docs
OnlineML.Bayes.MultinomialNB
```

## Bernoulli Naive Bayes

```@docs
OnlineML.Bayes.BernoulliNB
```

## Examples

```julia
using OnlineML.Bayes: GaussianNB

model = GaussianNB()

for (x, y) in data_stream
    fit!(model, (x, y))
end

y_pred = predict(model, x_new)
probs = predict_proba(model, x_new)
```
