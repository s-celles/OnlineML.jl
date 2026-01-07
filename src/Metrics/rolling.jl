# Rolling window metrics

using DataStructures: CircularBuffer

"""
    RollingMetric{M<:Metric}

Wraps any metric to compute it over a rolling window.

# Parameters
- `metric::M` - The base metric type
- `window_size::Int = 100` - Size of the rolling window

# Example
```julia
rolling_acc = RollingMetric(Accuracy(), window_size=100)
for (y_pred, y_true) in data
    fit!(rolling_acc, y_pred, y_true)
    println("Rolling accuracy: ", value(rolling_acc))
end
```
"""
mutable struct RollingMetric{M} <: OnlineStat{Tuple}
    metric_type::Type{M}
    window_size::Int
    buffer::CircularBuffer{Tuple{Any,Any}}
    current_metric::M
    n::Int

    function RollingMetric(::Type{M}; window_size::Int=100) where {M}
        new{M}(M, window_size, CircularBuffer{Tuple{Any,Any}}(window_size), M(), 0)
    end
end

# Convenience constructor with instance
RollingMetric(m::M; window_size::Int=100) where {M} = RollingMetric(typeof(m); window_size=window_size)

function OnlineStatsBase._fit!(rm::RollingMetric{M}, xy::Tuple) where {M}
    push!(rm.buffer, xy)
    rm.n += 1

    # Recompute metric from buffer
    rm.current_metric = rm.metric_type()
    for item in rm.buffer
        fit!(rm.current_metric, item)
    end

    return rm
end

OnlineStatsBase.value(rm::RollingMetric) = value(rm.current_metric)
OnlineStatsBase.nobs(rm::RollingMetric) = rm.n

function reset!(rm::RollingMetric{M}) where {M}
    empty!(rm.buffer)
    rm.current_metric = rm.metric_type()
    rm.n = 0
    return rm
end
