# Anomaly Detection

Online anomaly detection algorithms.

## Half-Space Trees

```@docs
OnlineML.Anomaly.HalfSpaceTrees
```

## Gaussian projection detector

```@docs
OnlineML.Anomaly.GaussianProjectionDetector
```

`LODA` is retained as a compatibility alias. It does not currently implement
the histogram-based LODA algorithm described by Pevný.

## Random-cut forest approximation

```@docs
OnlineML.Anomaly.RandomCutForestApproximation
```

`RobustRandomCutForest` is retained as a compatibility alias. The current
implementation is not a conforming geometric RRCF.

## Examples

```julia
using OnlineML.Anomaly: HalfSpaceTrees

model = HalfSpaceTrees(n_trees=10, height=8, window_size=250)

for x in data_stream
    score = fit_score!(model, x)
    if is_anomaly(model, x, threshold=0.8)
        println("Anomaly detected!")
    end
end
```
