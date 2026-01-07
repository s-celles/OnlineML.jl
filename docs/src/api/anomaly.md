# Anomaly Detection

Online anomaly detection algorithms.

## Half-Space Trees

```@docs
OnlineML.Anomaly.HalfSpaceTrees
```

## LODA

```@docs
OnlineML.Anomaly.LODA
```

## Examples

```julia
using OnlineML.Anomaly: HalfSpaceTrees

model = HalfSpaceTrees(n_trees=10, max_depth=8, window_size=250)

for x in data_stream
    score = fit_score!(model, x)
    if is_anomaly(model, x, threshold=0.8)
        println("Anomaly detected!")
    end
end
```
