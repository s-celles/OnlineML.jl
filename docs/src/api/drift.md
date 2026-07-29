# Drift Detection

Concept drift detection algorithms.

## ADWIN

```@docs
OnlineML.Drift.ADWIN
```

## DDM

```@docs
OnlineML.Drift.DDM
```

## EDDM

```@docs
OnlineML.Drift.EDDM
```

## PageHinkley

```@docs
OnlineML.Drift.PageHinkley
```

## Drift Wrapper

```@docs
OnlineML.Drift.DriftRetrainingWrapper
```

## Examples

### Manual Drift Detection

```julia
using OnlineML.Drift: ADWIN

detector = ADWIN(delta=0.002)

for error in error_stream
    status = update!(detector, error)
    if status == DriftDetected
        println("Drift detected! Resetting model...")
        reset!(model)
    elseif status == Warning
        println("Warning: potential drift")
    end
end
```

### Automatic Model Reset

```julia
using OnlineML.Drift: DriftRetrainingWrapper, DDM
using OnlineML.Linear: LogisticRegression

wrapped = DriftRetrainingWrapper(LogisticRegression(), DDM())

for (x, y) in data_stream
    fit!(wrapped, (x, y))
end

println("Drifts detected: ", wrapped.n_drifts)
```
