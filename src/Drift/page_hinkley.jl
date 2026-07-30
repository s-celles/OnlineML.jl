# PageHinkley - Page-Hinkley Test for change detection

"""
    PageHinkley <: Detector

Page-Hinkley Test for change detection in data streams.

Monitors cumulative sum of differences from mean and detects
drift when sum exceeds threshold.

# Parameters
- `min_instances::Int = 30` - Minimum observations before detection
- `delta::Float64 = 0.005` - Magnitude of changes to allow (tolerance)
- `threshold::Float64 = 50.0` - Change detection threshold
- `alpha::Float64 = 1 - 0.0001` - Forgetting factor for mean

# Example
```jldoctest
julia> detector = PageHinkley()
PageHinkley: n=0 | value=0.0

julia> fit!(detector, 0.5);

julia> status(detector)
NoDrift::DriftStatus = 0

julia> nobs(detector)
1
```
"""
mutable struct PageHinkley <: Detector
    min_instances::Int
    delta::Float64
    threshold::Float64
    alpha::Float64

    # Statistics
    n::Int
    x_mean::Float64
    sum::Float64
    sum_min::Float64

    drift_status::DriftStatus

    function PageHinkley(;
        min_instances::Int = 30,
        delta::Float64 = 0.005,
        threshold::Float64 = 50.0,
        alpha::Float64 = 1 - 0.0001
    )
        min_instances > 0 || throw(ArgumentError("min_instances must be positive"))
        delta >= 0 || throw(ArgumentError("delta must be non-negative"))
        threshold > 0 || throw(ArgumentError("threshold must be positive"))
        0 <= alpha < 1 || throw(ArgumentError("alpha must be in [0, 1)"))
        new(min_instances, delta, threshold, alpha,
            0, 0.0, 0.0, 0.0, NoDrift)
    end
end

function OnlineStatsBase._fit!(ph::PageHinkley, x::Real)
    x = Float64(x)
    ph.n += 1
    ph.drift_status = NoDrift

    # Initialize a new segment from its first value. Starting from zero would
    # repeatedly report drift after an internal reset on a stable non-zero
    # signal.
    if ph.n == 1
        ph.x_mean = x
    else
        ph.x_mean = ph.alpha * ph.x_mean + (1 - ph.alpha) * x
    end

    # Update cumulative sum
    ph.sum = ph.sum + (x - ph.x_mean - ph.delta)

    # Update minimum
    ph.sum_min = min(ph.sum_min, ph.sum)

    # Check for drift after minimum instances
    if ph.n >= ph.min_instances
        if ph.sum - ph.sum_min > ph.threshold
            ph.drift_status = DriftDetected
            reset_stats!(ph)
        end
    end

    return ph
end

function reset_stats!(ph::PageHinkley)
    ph.n = 0
    ph.x_mean = 0.0
    ph.sum = 0.0
    ph.sum_min = 0.0
end

OnlineStatsBase.value(ph::PageHinkley) = ph.sum - ph.sum_min
OnlineStatsBase.nobs(ph::PageHinkley) = ph.n

function reset!(ph::PageHinkley)
    reset_stats!(ph)
    ph.drift_status = NoDrift
    return ph
end

status(ph::PageHinkley) = ph.drift_status

# update! is an alias for fit! for drift detectors
update!(ph::PageHinkley, x::Real) = (fit!(ph, x); status(ph))

# Check drift status
detected_drift(ph::PageHinkley) = ph.drift_status == DriftDetected
detected_warning(ph::PageHinkley) = false  # PageHinkley doesn't have warning level
