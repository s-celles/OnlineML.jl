# ADWIN - Adaptive Windowing

"""
    ADWIN <: Detector

Adaptive Windowing drift detector.

The detector retains its current adaptive window and evaluates every admissible
cut point. When the two sub-window means differ by more than the
variance-sensitive ADWIN bound, the older sub-window is discarded. `nobs`
reports the total number of observations consumed, while `window_size` reports
the number currently retained.

This implementation stores the adaptive window exactly. It favors a clear,
testable cut-point implementation over the bucket-compressed optimization
described in the original paper.

# Parameters
- `delta::Float64 = 0.002` - False-positive confidence parameter
- `min_window_length::Int = 5` - Minimum observations on each side of a cut

# Example
```jldoctest
julia> detector = ADWIN(delta=0.002)
ADWIN: n=0 | value=0.0

julia> fit!(detector, 0.5);

julia> status(detector)
NoDrift::DriftStatus = 0

julia> nobs(detector)
1
```
"""
mutable struct ADWIN <: Detector
    delta::Float64
    min_window_length::Int
    window::Vector{Float64}
    total::Float64
    variance::Float64
    width::Int
    n::Int
    drift_status::DriftStatus

    function ADWIN(; delta::Float64 = 0.002, min_window_length::Int = 5)
        0 < delta < 1 || throw(ArgumentError("delta must be in (0, 1)"))
        min_window_length > 0 ||
            throw(ArgumentError("min_window_length must be positive"))
        new(delta, min_window_length, Float64[], 0.0, 0.0, 0, 0, NoDrift)
    end
end

function OnlineStatsBase._fit!(adwin::ADWIN, x::Real)
    observation = Float64(x)
    isfinite(observation) || throw(ArgumentError("observation must be finite"))

    old_mean = adwin.width == 0 ? 0.0 : adwin.total / adwin.width
    push!(adwin.window, observation)
    adwin.total += observation
    adwin.width += 1
    adwin.n += 1
    if adwin.width > 1
        new_mean = adwin.total / adwin.width
        adwin.variance += (observation - old_mean) * (observation - new_mean)
    end

    adwin.drift_status = detect_and_reduce!(adwin) ? DriftDetected : NoDrift
    return adwin
end

OnlineStatsBase.value(adwin::ADWIN) =
    adwin.width > 0 ? adwin.total / adwin.width : 0.0
OnlineStatsBase.nobs(adwin::ADWIN) = adwin.n

function reset!(adwin::ADWIN)
    empty!(adwin.window)
    adwin.total = 0.0
    adwin.variance = 0.0
    adwin.width = 0
    adwin.n = 0
    adwin.drift_status = NoDrift
    return adwin
end

function recompute_statistics!(adwin::ADWIN)
    adwin.width = length(adwin.window)
    adwin.total = sum(adwin.window)
    if adwin.width < 2
        adwin.variance = 0.0
    else
        mean = adwin.total / adwin.width
        adwin.variance = sum(x -> abs2(x - mean), adwin.window)
    end
    return adwin
end

function cut_bound(adwin::ADWIN, n0::Int, n1::Int)
    n = n0 + n1
    variance = n > 1 ? adwin.variance / (n - 1) : 0.0
    inverse_harmonic_mean = inv(n0) + inv(n1)
    log_term = log(2 * log(max(n, 2)) / adwin.delta)
    return sqrt(2 * inverse_harmonic_mean * variance * log_term) +
           (2 / 3) * inverse_harmonic_mean * log_term
end

"""
Evaluate admissible cut points and discard an obsolete prefix after a change.

The scan is repeated after a cut because more than one obsolete segment can be
present in the retained window. Returns `true` when at least one cut is made.
"""
function detect_and_reduce!(adwin::ADWIN)
    minimum = adwin.min_window_length
    adwin.width < 2 * minimum && return false
    detected = false

    while adwin.width >= 2 * minimum
        prefix_total = 0.0
        cut_index = 0

        for i in 1:(adwin.width - minimum)
            prefix_total += adwin.window[i]
            i < minimum && continue

            n0 = i
            n1 = adwin.width - i
            mean0 = prefix_total / n0
            mean1 = (adwin.total - prefix_total) / n1
            if abs(mean0 - mean1) > cut_bound(adwin, n0, n1)
                cut_index = i
                break
            end
        end

        cut_index == 0 && break
        deleteat!(adwin.window, 1:cut_index)
        recompute_statistics!(adwin)
        detected = true
    end

    return detected
end

status(adwin::ADWIN) = adwin.drift_status
update!(adwin::ADWIN, x::Real) = (fit!(adwin, x); status(adwin))
detected_drift(adwin::ADWIN) = adwin.drift_status == DriftDetected
detected_warning(::ADWIN) = false
current_mean(adwin::ADWIN) = value(adwin)
window_size(adwin::ADWIN) = adwin.width
