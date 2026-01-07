# Instance-Based Learning

K-Nearest Neighbors and related algorithms.

## KNN Classifier

```@docs
OnlineML.Instance.KNNClassifier
```

## Examples

```julia
using OnlineML.Instance: KNNClassifier

model = KNNClassifier(k=5, window_size=1000)

for (x, y) in data_stream
    fit!(model, (x, y))
end

y_pred = predict(model, x_new)
```
