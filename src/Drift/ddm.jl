# DDM - Drift Detection Method
# EDDM - Early Drift Detection Method

"""
    DDM <: Detector

Drift Detection Method (DDM) for concept drift detection.

Monitors the error rate and detects drift when error rate
increases significantly.

# Parameters
- `warning_level::Float64 = 2.0` - Standard deviations for warning
- `drift_level::Float64 = 3.0` - Standard deviations for drift
- `min_instances::Int = 30` - Minimum instances before detection

# Example
```jldoctest
julia> detector = DDM()
DDM: n=0 | value=0.0

julia> fit!(detector, 0.0);

julia> status(detector)
NoDrift::DriftStatus = 0

julia> nobs(detector)
1
```
"""
mutable struct DDM <: Detector
    warning_level::Float64
    drift_level::Float64
    min_instances::Int

    # Statistics
    n::Int
    p::Float64  # Error rate
    s::Float64  # Standard deviation

    # Minimum values (best performance)
    p_min::Float64
    s_min::Float64
    n_min::Int

    drift_status::DriftStatus

    function DDM(;
        warning_level::Float64 = 2.0,
        drift_level::Float64 = 3.0,
        min_instances::Int = 30
    )
        warning_level > 0 || throw(ArgumentError("warning_level must be positive"))
        drift_level > warning_level ||
            throw(ArgumentError("drift_level must be greater than warning_level"))
        min_instances > 0 || throw(ArgumentError("min_instances must be positive"))
        new(warning_level, drift_level, min_instances,
            0, 0.0, 0.0, Inf, Inf, 0, NoDrift)
    end
end

function OnlineStatsBase._fit!(ddm::DDM, x::Real)
    x = Float64(x)
    x in (0.0, 1.0) || throw(ArgumentError("DDM expects a binary error indicator"))
    ddm.n += 1

    # Update error rate estimate
    ddm.p = ddm.p + (x - ddm.p) / ddm.n
    ddm.s = sqrt(ddm.p * (1 - ddm.p) / ddm.n)

    ddm.drift_status = NoDrift

    if ddm.n < ddm.min_instances
        return ddm
    end

    # Update minimum
    if ddm.p + ddm.s < ddm.p_min + ddm.s_min
        ddm.p_min = ddm.p
        ddm.s_min = ddm.s
        ddm.n_min = ddm.n
    end

    # Check for drift
    if ddm.p + ddm.s > ddm.p_min + ddm.drift_level * ddm.s_min
        ddm.drift_status = DriftDetected
        # Reset after drift
        reset_stats!(ddm)
    elseif ddm.p + ddm.s > ddm.p_min + ddm.warning_level * ddm.s_min
        ddm.drift_status = Warning
    end

    return ddm
end

function reset_stats!(ddm::DDM)
    ddm.n = 0
    ddm.p = 0.0
    ddm.s = 0.0
    ddm.p_min = Inf
    ddm.s_min = Inf
    ddm.n_min = 0
end

OnlineStatsBase.value(ddm::DDM) = ddm.p
OnlineStatsBase.nobs(ddm::DDM) = ddm.n

function reset!(ddm::DDM)
    reset_stats!(ddm)
    ddm.drift_status = NoDrift
    return ddm
end

status(ddm::DDM) = ddm.drift_status

# update! is an alias for fit! for drift detectors
update!(ddm::DDM, x::Real) = (fit!(ddm, x); status(ddm))

# Check drift status
detected_drift(ddm::DDM) = ddm.drift_status == DriftDetected
detected_warning(ddm::DDM) = ddm.drift_status == Warning


"""
    EDDM <: Detector

Early Drift Detection Method.

Similar to DDM but uses distance between errors instead of error rate,
which can detect drift earlier.

The mean and population standard deviation are updated online over the
observed distances between consecutive errors. Detection starts after
`min_instances` errors, as specified by the original algorithm.

# Parameters
- `warning_level::Float64 = 0.95` - Threshold for warning (as ratio of max)
- `drift_level::Float64 = 0.90` - Threshold for drift (as ratio of max)
- `min_instances::Int = 30` - Minimum instances before detection

# Example
```jldoctest
julia> detector = EDDM()
EDDM: n=0 | value=0.0

julia> fit!(detector, 1.0);

julia> status(detector)
NoDrift::DriftStatus = 0

julia> nobs(detector)
1
```
"""
mutable struct EDDM <: Detector
    warning_level::Float64
    drift_level::Float64
    min_instances::Int

    # Statistics
    n::Int
    n_errors::Int
    last_error_idx::Int
    mean_distance::Float64
    std_distance::Float64

    # Maximum values
    mean_max::Float64
    std_max::Float64

    drift_status::DriftStatus

    function EDDM(;
        warning_level::Float64 = 0.95,
        drift_level::Float64 = 0.90,
        min_instances::Int = 30
    )
        0 < drift_level < warning_level < 1 ||
            throw(ArgumentError("levels must satisfy 0 < drift_level < warning_level < 1"))
        min_instances > 0 || throw(ArgumentError("min_instances must be positive"))
        new(warning_level, drift_level, min_instances,
            0, 0, 0, 0.0, 0.0, 0.0, 0.0, NoDrift)
    end
end

function OnlineStatsBase._fit!(eddm::EDDM, x::Real)
    x = Float64(x)
    x in (0.0, 1.0) || throw(ArgumentError("EDDM expects a binary error indicator"))
    eddm.n += 1
    eddm.drift_status = NoDrift

    if x > 0.5  # Error occurred
        eddm.n_errors += 1

        if eddm.n_errors > 1
            distance = eddm.n - eddm.last_error_idx
            n_distances = eddm.n_errors - 1

            # Welford update over inter-error distances. Reconstructing M2 from
            # the stored population standard deviation keeps the state compact.
            old_mean = eddm.mean_distance
            old_m2 =
                n_distances > 1 ? eddm.std_distance^2 * (n_distances - 1) : 0.0
            eddm.mean_distance = old_mean + (distance - old_mean) / n_distances
            new_m2 = old_m2 +
                (distance - old_mean) * (distance - eddm.mean_distance)
            eddm.std_distance = sqrt(max(new_m2 / n_distances, 0.0))
        end

        eddm.last_error_idx = eddm.n

        # Check for drift after minimum instances
        if eddm.n_errors >= eddm.min_instances
            current_metric = eddm.mean_distance + 2 * eddm.std_distance

            # Update maximum
            if current_metric > eddm.mean_max + 2 * eddm.std_max
                eddm.mean_max = eddm.mean_distance
                eddm.std_max = eddm.std_distance
            end

            max_metric = eddm.mean_max + 2 * eddm.std_max

            if max_metric > 0
                ratio = current_metric / max_metric

                if ratio < eddm.drift_level
                    eddm.drift_status = DriftDetected
                    reset_stats!(eddm)
                elseif ratio < eddm.warning_level
                    eddm.drift_status = Warning
                end
            end
        end
    end

    return eddm
end

function reset_stats!(eddm::EDDM)
    eddm.n = 0
    eddm.n_errors = 0
    eddm.last_error_idx = 0
    eddm.mean_distance = 0.0
    eddm.std_distance = 0.0
    eddm.mean_max = 0.0
    eddm.std_max = 0.0
end

OnlineStatsBase.value(eddm::EDDM) = eddm.mean_distance
OnlineStatsBase.nobs(eddm::EDDM) = eddm.n

function reset!(eddm::EDDM)
    reset_stats!(eddm)
    eddm.drift_status = NoDrift
    return eddm
end

status(eddm::EDDM) = eddm.drift_status

# update! is an alias for fit! for drift detectors
update!(eddm::EDDM, x::Real) = (fit!(eddm, x); status(eddm))

# Check drift status
detected_drift(eddm::EDDM) = eddm.drift_status == DriftDetected
detected_warning(eddm::EDDM) = eddm.drift_status == Warning
