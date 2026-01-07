# LODA - Lightweight Online Detector of Anomalies
# Based on: Pevný, T. (2016). Loda: Lightweight on-line detector of anomalies.
# Machine Learning, 102(2), 275-304.

using OnlineStats: Variance

"""
    LODA{T<:AbstractFloat} <: UnsupervisedLearner{AbstractVector{T}}

Lightweight Online Detector of Anomalies using random projections
and Gaussian density estimation.

# Parameters
- `n_projections::Int = 10` - Number of random projections

# Algorithm
LODA works by:
1. Projecting high-dimensional data onto random one-dimensional subspaces
2. Maintaining Gaussian statistics (mean, variance) for each projection
3. Computing anomaly scores as average Mahalanobis distance across projections

# Example
```julia
model = LODA(n_projections=20)
for x in normal_data
    fit!(model, x)
end
score = score_one(model, suspicious_point)  # Higher score = more anomalous
```
"""
mutable struct LODA{T<:AbstractFloat} <: UnsupervisedLearner{AbstractVector{T}}
    n_projections::Int
    projections::Vector{Vector{T}}
    stats::Vector{Variance}  # Running mean and variance for each projection
    n::Int
    initialized::Bool
    rng::Random.AbstractRNG

    function LODA{T}(;
        n_projections::Int=10,
        rng::Random.AbstractRNG=Random.default_rng()
    ) where {T<:AbstractFloat}
        new{T}(
            n_projections,
            Vector{T}[],
            Variance[],
            0,
            false,
            rng
        )
    end
end

LODA(; kwargs...) = LODA{Float64}(; kwargs...)

# Initialize projections when we see the first data point
function initialize!(loda::LODA{T}, d::Int) where {T}
    # Generate random sparse projections
    # Each projection is a random direction
    for _ in 1:loda.n_projections
        # Random unit vector
        proj = randn(loda.rng, T, d)
        proj ./= sqrt(sum(proj.^2) + eps(T))  # Normalize
        push!(loda.projections, proj)
        # Create variance tracker for this projection
        push!(loda.stats, Variance(T))
    end
    loda.initialized = true
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(loda::LODA{T}, x) where {T}
    x = collect(T, x)
    d = length(x)

    if !loda.initialized
        initialize!(loda, d)
    end

    # Project onto each random direction and update statistics
    for i in 1:loda.n_projections
        projected = dot(loda.projections[i], x)
        fit!(loda.stats[i], projected)
    end

    loda.n += 1
    return loda
end

OnlineStatsBase.value(loda::LODA) = (projections=loda.projections, stats=loda.stats)
OnlineStatsBase.nobs(loda::LODA) = loda.n

function reset!(loda::LODA{T}) where {T}
    empty!(loda.projections)
    empty!(loda.stats)
    loda.n = 0
    loda.initialized = false
    return loda
end

# Import mean and var functions
using OnlineStats: mean, var

# Anomaly score functions

"""
    score_one(loda::LODA, x) -> Float64

Compute anomaly score for observation x.
Higher scores indicate more anomalous observations.
Returns a score in [0, 1] range.
"""
function score_one(loda::LODA{T}, x::AbstractVector) where {T}
    if !loda.initialized || loda.n < 2
        return 0.5  # Return neutral score for uninitialized model
    end

    x = collect(T, x)
    total_z_score = zero(T)

    for i in 1:loda.n_projections
        projected = dot(loda.projections[i], x)
        m = mean(loda.stats[i])
        v = var(loda.stats[i])

        if v > eps(T)
            # Z-score (standardized distance from mean)
            z = abs(projected - m) / sqrt(v)
            total_z_score += z
        end
    end

    # Average z-score across projections
    avg_z = total_z_score / loda.n_projections

    # Convert to [0, 1] range using sigmoid
    # Values beyond 3 standard deviations should score close to 1
    return 1.0 / (1.0 + exp(-avg_z + 2.0))
end

# Dot product helper
using LinearAlgebra: dot

export LODA
