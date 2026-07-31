# Experimental random-cut-forest approximation

using Random
using DataStructures: CircularBuffer

"""
    RandomCutForestApproximation <: UnsupervisedLearner{AbstractVector{Float64}}

Experimental random-cut-forest approximation for streaming anomaly detection.

Uses an ensemble of Random Cut Trees with CoDisplacement scoring
to detect anomalies in streaming data. Maintains a sliding window
of recent points.

This implementation is not a conforming Robust Random Cut Forest. In
particular, its leaves do not retain point coordinates, so insertion does not
implement the geometric RRCT construction from Guha et al. The
`RobustRandomCutForest` name remains available as a compatibility alias.

# Parameters
- `n_trees::Int = 100` - Number of trees in the forest
- `tree_size::Int = 256` - Maximum number of points per tree
- `seed::Union{Int, Nothing} = nothing` - Random seed for reproducibility

# Example
```jldoctest
julia> rrcf = RandomCutForestApproximation(n_trees=50, tree_size=100)
RandomCutForestApproximation: n=0 | trees=50 | tree_size=100

julia> fit!(rrcf, [1.0, 2.0, 3.0]);

julia> s = score(rrcf, [1.0, 2.0, 3.0])
0.0

julia> nobs(rrcf)
1
```

# Reference
Guha, S., Mishra, N., Roy, G., & Schrijvers, O. (2016).
Robust Random Cut Forest Based Anomaly Detection On Streams. ICML.
"""
mutable struct RandomCutForestApproximation <:
               UnsupervisedLearner{AbstractVector{Float64}}
    n_trees::Int
    tree_size::Int
    trees::Vector{CutTree}
    point_buffer::CircularBuffer{Vector{Float64}}
    point_indices::CircularBuffer{Int}
    next_idx::Int
    n::Int
    rng::Random.AbstractRNG

    function RandomCutForestApproximation(;
        n_trees::Int = 100,
        tree_size::Int = 256,
        seed::Union{Int, Nothing} = nothing
    )
        n_trees > 0 || throw(ArgumentError("n_trees must be positive"))
        tree_size > 0 || throw(ArgumentError("tree_size must be positive"))

        rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
        trees = [CutTree(rng=Random.MersenneTwister(rand(rng, UInt64))) for _ in 1:n_trees]
        point_buffer = CircularBuffer{Vector{Float64}}(tree_size)
        point_indices = CircularBuffer{Int}(tree_size)

        new(n_trees, tree_size, trees, point_buffer, point_indices, 1, 0, rng)
    end
end

"""
    RobustRandomCutForest

Compatibility alias for [`RandomCutForestApproximation`](@ref).

New code should use the explicit approximation name until the geometric RRCT
insertion and externality score are implemented and independently qualified.
"""
const RobustRandomCutForest = RandomCutForestApproximation

# OnlineStatsBase interface
function OnlineStatsBase._fit!(
    rrcf::RandomCutForestApproximation,
    x::AbstractVector,
)
    point = collect(Float64, x)
    if !isempty(rrcf.point_buffer)
        expected = length(first(rrcf.point_buffer))
        length(point) == expected ||
            throw(DimensionMismatch("expected $expected features, got $(length(point))"))
    end
    point_idx = rrcf.next_idx
    rrcf.next_idx += 1

    # If buffer is full, remove oldest point from all trees
    if isfull(rrcf.point_buffer)
        old_idx = first(rrcf.point_indices)
        for tree in rrcf.trees
            delete_point!(tree, old_idx)
        end
    end

    # Add new point to buffer
    push!(rrcf.point_buffer, point)
    push!(rrcf.point_indices, point_idx)

    # Insert into all trees
    for tree in rrcf.trees
        insert_point!(tree, point, point_idx)
    end

    rrcf.n += 1
    return rrcf
end

OnlineStatsBase.value(rrcf::RandomCutForestApproximation) = rrcf.trees
OnlineStatsBase.nobs(rrcf::RandomCutForestApproximation) = rrcf.n

function reset!(rrcf::RandomCutForestApproximation)
    for tree in rrcf.trees
        reset!(tree)
    end
    empty!(rrcf.point_buffer)
    empty!(rrcf.point_indices)
    rrcf.next_idx = 1
    rrcf.n = 0
    return rrcf
end

"""
    score(rrcf::RandomCutForestApproximation, x::AbstractVector) -> Float64

Compute the anomaly score for a point.
Returns the average CoDisplacement across all trees.
Higher scores indicate more anomalous points.

Note: The point must have been previously inserted into the forest.
For scoring new points, use `fit_score!`.
"""
function score(rrcf::RandomCutForestApproximation, x::AbstractVector)
    # Find the point index by matching
    point = collect(Float64, x)
    if !isempty(rrcf.point_buffer)
        expected = length(first(rrcf.point_buffer))
        length(point) == expected ||
            throw(DimensionMismatch("expected $expected features, got $(length(point))"))
    end

    for (i, (stored_point, idx)) in enumerate(zip(rrcf.point_buffer, rrcf.point_indices))
        if stored_point ≈ point
            return score_by_index(rrcf, idx)
        end
    end

    throw(ArgumentError(
        "point is not retained by the forest; use fit_score! to insert and score a new point",
    ))
end

"""
Compute score for a point by its internal index.
"""
function score_by_index(rrcf::RandomCutForestApproximation, point_idx::Int)
    if rrcf.n == 0
        return 0.0
    end

    total_score = 0.0
    for tree in rrcf.trees
        total_score += codisplacement(tree, point_idx)
    end

    return total_score / rrcf.n_trees
end

"""
    score_one(rrcf::RandomCutForestApproximation, x::AbstractVector) -> Float64

Alias for `score`.
"""
score_one(rrcf::RandomCutForestApproximation, x::AbstractVector) = score(rrcf, x)

"""
    fit_score!(rrcf::RandomCutForestApproximation, x::AbstractVector) -> Float64

Insert a point into the forest and return its anomaly score.
"""
function fit_score!(rrcf::RandomCutForestApproximation, x::AbstractVector)
    point = collect(Float64, x)
    point_idx = rrcf.next_idx

    # Insert the point
    fit!(rrcf, x)

    # Score the just-inserted point
    return score_by_index(rrcf, point_idx)
end

"""
    is_anomaly(rrcf::RandomCutForestApproximation, x::AbstractVector; threshold::Float64=3.0) -> Bool

Check if a point is an anomaly based on its score.
Uses a threshold relative to the expected depth.

# Arguments
- `threshold::Float64 = 3.0` - Score threshold for anomaly detection
"""
function is_anomaly(
    rrcf::RandomCutForestApproximation,
    x::AbstractVector;
    threshold::Float64=3.0,
)
    s = score(rrcf, x)
    # Normalize by expected depth
    expected_depth = log2(max(1.0, Float64(length(rrcf.point_buffer))))
    normalized = s / max(1.0, expected_depth)
    return normalized > threshold
end

# Inspection methods
n_trees(rrcf::RandomCutForestApproximation) = rrcf.n_trees
tree_size(rrcf::RandomCutForestApproximation) = rrcf.tree_size
current_size(rrcf::RandomCutForestApproximation) = length(rrcf.point_buffer)

function Base.show(io::IO, rrcf::RandomCutForestApproximation)
    print(io, "RandomCutForestApproximation: n=$(nobs(rrcf)) | trees=$(n_trees(rrcf)) | tree_size=$(tree_size(rrcf))")
end

export RandomCutForestApproximation, RobustRandomCutForest
export score, score_one, fit_score!, is_anomaly
export n_trees, tree_size, current_size
