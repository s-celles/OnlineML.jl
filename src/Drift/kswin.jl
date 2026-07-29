# KSWIN - Kolmogorov-Smirnov Windowed Drift Detector

using DataStructures: CircularDeque

"""
    KSWIN <: Detector

Kolmogorov-Smirnov Windowed (KSWIN) drift detector.

Uses the Kolmogorov-Smirnov statistical test to detect distribution changes
between two halves of a sliding window.

# Parameters
- `alpha::Float64 = 0.005` - Significance level for the KS test
- `window_size::Int = 100` - Total window size (split into two halves)
- `stat_size::Int = 30` - Size of reference window for comparison

# Example
```jldoctest
julia> detector = KSWIN(alpha=0.005, window_size=100)
KSWIN: n=0 | value=0.0

julia> fit!(detector, 0.5);

julia> status(detector)
NoDrift::DriftStatus = 0

julia> nobs(detector)
1
```

# Reference
Raab, C., Heusinger, M., & Schleif, F. M. (2020). Reactive Soft Prototype Computing
for Concept Drift Streams. Neurocomputing.
"""
mutable struct KSWIN <: Detector
    alpha::Float64
    window_size::Int
    stat_size::Int
    window::CircularDeque{Float64}
    n::Int
    drift_status::DriftStatus

    function KSWIN(;
        alpha::Float64 = 0.005,
        window_size::Int = 100,
        stat_size::Int = 30
    )
        alpha > 0 && alpha < 1 || throw(ArgumentError("alpha must be in (0, 1)"))
        window_size > 0 || throw(ArgumentError("window_size must be positive"))
        stat_size > 0 || throw(ArgumentError("stat_size must be positive"))
        stat_size <= window_size ÷ 2 || throw(ArgumentError("stat_size must be <= window_size/2"))

        new(alpha, window_size, stat_size, CircularDeque{Float64}(window_size), 0, NoDrift)
    end
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(detector::KSWIN, x::Real)
    x = Float64(x)

    # Add to window
    if length(detector.window) >= detector.window_size
        popfirst!(detector.window)
    end
    push!(detector.window, x)
    detector.n += 1

    # Check for drift when we have enough data
    if length(detector.window) >= detector.window_size
        detector.drift_status = check_ks_drift(detector)
    else
        detector.drift_status = NoDrift
    end

    return detector
end

OnlineStatsBase.value(detector::KSWIN) = length(detector.window) > 0 ? last(detector.window) : 0.0
OnlineStatsBase.nobs(detector::KSWIN) = detector.n

function reset!(detector::KSWIN)
    empty!(detector.window)
    detector.n = 0
    detector.drift_status = NoDrift
    return detector
end

"""
Perform KS test between reference window and recent window.
"""
function check_ks_drift(detector::KSWIN)
    n = length(detector.window)
    if n < detector.window_size
        return NoDrift
    end

    # Split window into reference (older) and recent (newer) portions
    window_arr = collect(detector.window)
    stat_size = detector.stat_size

    # Reference: first stat_size elements
    reference = window_arr[1:stat_size]
    # Recent: last stat_size elements
    recent = window_arr[end-stat_size+1:end]

    # Compute KS statistic
    ks_stat = ks_statistic(reference, recent)

    # Critical value based on alpha and sample sizes
    critical = ks_critical_value(stat_size, stat_size, detector.alpha)

    if ks_stat > critical
        return DriftDetected
    elseif ks_stat > critical * 0.8  # Warning threshold at 80% of critical
        return Warning
    end

    return NoDrift
end

"""
Compute the Kolmogorov-Smirnov statistic between two samples.
Returns the maximum absolute difference between empirical CDFs.
"""
function ks_statistic(sample1::AbstractVector, sample2::AbstractVector)
    n1 = length(sample1)
    n2 = length(sample2)

    if n1 == 0 || n2 == 0
        return 0.0
    end

    # Sort both samples
    sorted1 = sort(sample1)
    sorted2 = sort(sample2)

    # Combine and sort all values
    all_values = sort(vcat(sorted1, sorted2))

    # Compute maximum difference between ECDFs
    max_diff = 0.0

    for x in all_values
        # ECDF of sample1 at x
        ecdf1 = count(v -> v <= x, sorted1) / n1
        # ECDF of sample2 at x
        ecdf2 = count(v -> v <= x, sorted2) / n2

        diff = abs(ecdf1 - ecdf2)
        if diff > max_diff
            max_diff = diff
        end
    end

    return max_diff
end

"""
Compute the critical value for the KS test at given significance level.
Uses the asymptotic approximation.
"""
function ks_critical_value(n1::Int, n2::Int, alpha::Float64)
    # Asymptotic critical value: c(α) * sqrt((n1 + n2) / (n1 * n2))
    # where c(α) = sqrt(-0.5 * ln(α/2)) for the two-sample test
    c_alpha = sqrt(-0.5 * log(alpha / 2))
    return c_alpha * sqrt((n1 + n2) / (n1 * n2))
end

# Detector interface
status(detector::KSWIN) = detector.drift_status
update!(detector::KSWIN, x::Real) = (fit!(detector, x); status(detector))
detected_drift(detector::KSWIN) = detector.drift_status == DriftDetected
detected_warning(detector::KSWIN) = detector.drift_status == Warning

# Inspection methods
window_size(detector::KSWIN) = length(detector.window)
current_window(detector::KSWIN) = collect(detector.window)

function Base.show(io::IO, detector::KSWIN)
    v = length(detector.window) > 0 ? last(detector.window) : 0.0
    print(io, "KSWIN: n=$(nobs(detector)) | value=$(v)")
end

export KSWIN
