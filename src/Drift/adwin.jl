# ADWIN - Adaptive Windowing

"""
    ADWIN <: Detector

Adaptive Windowing drift detector.

!!! warning "Experimental approximation"
    This implementation compresses observations into buckets but does not yet
    perform ADWIN cut-point evaluation or shrink its logical window after a
    detected change. It must not be treated as a conforming ADWIN
    implementation.

# Parameters
- `delta::Float64 = 0.002` - Confidence parameter

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
    bucket_row::Vector{Vector{Float64}}  # Buckets organized by size
    bucket_count::Vector{Int}
    total::Float64
    variance::Float64
    width::Int
    n::Int
    drift_status::DriftStatus

    function ADWIN(; delta::Float64 = 0.002)
        0 < delta < 1 || throw(ArgumentError("delta must be in (0, 1)"))
        new(delta, [Float64[] for _ in 1:32], zeros(Int, 32),
            0.0, 0.0, 0, 0, NoDrift)
    end
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(adwin::ADWIN, x::Real)
    x = Float64(x)

    # Add to smallest bucket
    push!(adwin.bucket_row[1], x)
    adwin.bucket_count[1] += 1

    # Update statistics
    if adwin.width > 0
        inc_variance = (adwin.width * (x - adwin.total / adwin.width)^2) / (adwin.width + 1)
        adwin.variance += inc_variance
    end
    adwin.total += x
    adwin.width += 1
    adwin.n += 1

    # Compress buckets
    compress_buckets!(adwin)

    # Check for drift
    adwin.drift_status = detect_change(adwin)

    return adwin
end

OnlineStatsBase.value(adwin::ADWIN) = adwin.width > 0 ? adwin.total / adwin.width : 0.0
OnlineStatsBase.nobs(adwin::ADWIN) = adwin.n

function reset!(adwin::ADWIN)
    for row in adwin.bucket_row
        empty!(row)
    end
    fill!(adwin.bucket_count, 0)
    adwin.total = 0.0
    adwin.variance = 0.0
    adwin.width = 0
    adwin.n = 0
    adwin.drift_status = NoDrift
    return adwin
end

# Compress buckets when they exceed capacity
function compress_buckets!(adwin::ADWIN)
    max_buckets = 5  # Maximum buckets per row

    for i in 1:length(adwin.bucket_row)-1
        if adwin.bucket_count[i] >= max_buckets
            # Merge two smallest buckets into next row
            if length(adwin.bucket_row[i]) >= 2
                b1 = popfirst!(adwin.bucket_row[i])
                b2 = popfirst!(adwin.bucket_row[i])
                push!(adwin.bucket_row[i+1], (b1 + b2) / 2)
                adwin.bucket_count[i] -= 2
                adwin.bucket_count[i+1] += 1
            end
        end
    end
end

# Detect change using statistical test
function detect_change(adwin::ADWIN)
    if adwin.width < 10
        return NoDrift
    end

    # Simple mean-based change detection
    # Full ADWIN uses more sophisticated bucket comparison
    n = adwin.width
    mean = adwin.total / n
    var = adwin.variance / n

    if var <= 0
        return NoDrift
    end

    # Hoeffding bound
    eps = sqrt(2 * var * log(2 / adwin.delta) / n)

    # Check if recent values differ significantly
    # Simplified: use standard deviation threshold
    std_dev = sqrt(var)
    if std_dev > eps * 10
        return DriftDetected
    elseif std_dev > eps * 5
        return Warning
    end

    return NoDrift
end

# Get current drift status
status(adwin::ADWIN) = adwin.drift_status

# update! is an alias for fit! for drift detectors
update!(adwin::ADWIN, x::Real) = (fit!(adwin, x); status(adwin))

# Check drift status
detected_drift(adwin::ADWIN) = adwin.drift_status == DriftDetected
detected_warning(adwin::ADWIN) = adwin.drift_status == Warning

# Get current window mean
current_mean(adwin::ADWIN) = adwin.width > 0 ? adwin.total / adwin.width : 0.0

# Get window size
window_size(adwin::ADWIN) = adwin.width
