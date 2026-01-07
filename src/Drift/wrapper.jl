# DriftRetrainingWrapper - Automatic model retraining on drift

"""
    DriftRetrainingWrapper{L,D} <: Learner

Wraps a learner with a drift detector to automatically reset
the model when concept drift is detected.

# Parameters
- `learner::L` - The wrapped learner
- `detector::D` - The drift detector
- `reset_on_warning::Bool = false` - Reset on warning (not just drift)

# Example
```julia
model = DriftRetrainingWrapper(
    LogisticRegression(),
    DDM()
)

for (x, y) in stream
    y_pred = predict(model, x)
    error = y_pred != y ? 1.0 : 0.0
    fit!(model, (x, y), error)
end

println("Drifts detected: ", drift_count(model))
```
"""
mutable struct DriftRetrainingWrapper{L,D} <: OnlineStat{Any}
    learner::L
    detector::D
    reset_on_warning::Bool
    n::Int
    n_drifts::Int
    n_warnings::Int

    function DriftRetrainingWrapper(learner::L, detector::D;
        reset_on_warning::Bool = false
    ) where {L,D}
        new{L,D}(learner, detector, reset_on_warning, 0, 0, 0)
    end
end

# Fit the wrapper - takes (x, y) for learner and error for detector
function OnlineStatsBase._fit!(w::DriftRetrainingWrapper, data)
    # Data can be ((x, y), error) or just (x, y) where we compute error internally
    if data isa Tuple && length(data) == 2 && data[1] isa Tuple
        # ((x, y), error) format
        xy, error = data
        fit!(w.learner, xy)
        fit!(w.detector, error)
    else
        # (x, y) format - fit learner only
        fit!(w.learner, data)
    end

    w.n += 1

    # Check drift status
    drift_status = status(w.detector)

    if drift_status == DriftDetected
        w.n_drifts += 1
        reset!(w.learner)
    elseif drift_status == Warning
        w.n_warnings += 1
        if w.reset_on_warning
            reset!(w.learner)
        end
    end

    return w
end

OnlineStatsBase.value(w::DriftRetrainingWrapper) = value(w.learner)
OnlineStatsBase.nobs(w::DriftRetrainingWrapper) = w.n

function reset!(w::DriftRetrainingWrapper)
    reset!(w.learner)
    reset!(w.detector)
    w.n = 0
    w.n_drifts = 0
    w.n_warnings = 0
    return w
end

# Forward predict to learner
import ..OnlineML: predict, predict_proba

predict(w::DriftRetrainingWrapper, x) = predict(w.learner, x)
predict_proba(w::DriftRetrainingWrapper, x) = predict_proba(w.learner, x)

# Inspection methods
drift_count(w::DriftRetrainingWrapper) = w.n_drifts
warning_count(w::DriftRetrainingWrapper) = w.n_warnings
detector_status(w::DriftRetrainingWrapper) = status(w.detector)

# Export
export DriftRetrainingWrapper, drift_count, warning_count, detector_status
