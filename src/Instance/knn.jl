# K-Nearest Neighbors with sliding window

using DataStructures: CircularBuffer

"""
    KNN{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}

Online K-Nearest Neighbors classifier with sliding window.

# Parameters
- `k::Int = 5` - Number of neighbors
- `window_size::Int = 1000` - Maximum observations to retain
- `weighted::Bool = true` - Use distance-weighted voting
- `distance::Distances.Metric = Euclidean()` - Distance metric

# Example
```jldoctest
julia> model = KNN(k=5, window_size=500)
KNN: n=0 | value=(X = Vector{Float64}[], y = Int64[])

julia> fit!(model, ([1.0, 2.0], 1));

julia> predict(model, [1.0, 2.0])
1

julia> nobs(model)
1
```
"""
mutable struct KNN{T<:AbstractFloat, M<:Distances.Metric} <: Learner{AbstractVector{T}, Int}
    k::Int
    window_size::Int
    weighted::Bool
    distance::M
    X_buffer::Vector{Vector{T}}
    y_buffer::Vector{Int}
    n::Int

    function KNN{T}(;
        k::Int = 5,
        window_size::Int = 1000,
        weighted::Bool = true,
        distance::Distances.Metric = Euclidean()
    ) where {T<:AbstractFloat}
        new{T, typeof(distance)}(
            k,
            window_size,
            weighted,
            distance,
            Vector{T}[],
            Int[],
            0
        )
    end
end

KNN(; kwargs...) = KNN{Float64}(; kwargs...)

# OnlineStatsBase interface
function OnlineStatsBase._fit!(knn::KNN{T}, xy::Tuple) where {T}
    x, y = xy
    x = collect(T, x)
    y = Int(y)

    # Add to buffers
    push!(knn.X_buffer, x)
    push!(knn.y_buffer, y)

    # Maintain window size
    while length(knn.X_buffer) > knn.window_size
        popfirst!(knn.X_buffer)
        popfirst!(knn.y_buffer)
    end

    knn.n += 1
    return knn
end

OnlineStatsBase.value(knn::KNN) = (X=knn.X_buffer, y=knn.y_buffer)
OnlineStatsBase.nobs(knn::KNN) = knn.n

function reset!(knn::KNN{T}) where {T}
    empty!(knn.X_buffer)
    empty!(knn.y_buffer)
    knn.n = 0
    return knn
end

# Learner interface
function predict(knn::KNN{T}, x::AbstractVector) where {T}
    if isempty(knn.X_buffer)
        return 0
    end

    x = collect(T, x)

    # Find k nearest neighbors
    neighbors, distances = find_neighbors(knn, x)

    # Vote
    return vote(knn, neighbors, distances)
end

function predict_proba(knn::KNN{T}, x::AbstractVector) where {T}
    if isempty(knn.X_buffer)
        return Dict{Int, T}()
    end

    x = collect(T, x)
    neighbors, distances = find_neighbors(knn, x)

    # Compute class probabilities from neighbor votes
    return vote_proba(knn, neighbors, distances)
end

# Find k nearest neighbors
function find_neighbors(knn::KNN{T}, x::Vector{T}) where {T}
    n_samples = length(knn.X_buffer)
    k_actual = min(knn.k, n_samples)

    # Compute all distances
    dists = [Distances.evaluate(knn.distance, x, xi) for xi in knn.X_buffer]

    # Find k smallest
    sorted_idx = sortperm(dists)
    neighbor_idx = sorted_idx[1:k_actual]
    neighbor_dists = dists[neighbor_idx]

    return neighbor_idx, neighbor_dists
end

# Majority voting
function vote(knn::KNN, neighbors::Vector{Int}, distances::Vector{<:Real})
    vote_counts = Dict{Int, Float64}()

    for (i, idx) in enumerate(neighbors)
        y = knn.y_buffer[idx]

        if knn.weighted
            # Inverse distance weighting
            d = distances[i]
            w = d > 0 ? 1.0 / d : 1e10  # Handle zero distance
        else
            w = 1.0
        end

        vote_counts[y] = get(vote_counts, y, 0.0) + w
    end

    if isempty(vote_counts)
        return 0
    end

    return dict_argmax(vote_counts)
end

# Probability voting
function vote_proba(knn::KNN{T}, neighbors::Vector{Int}, distances::Vector{<:Real}) where {T}
    vote_counts = Dict{Int, T}()
    total_weight = zero(T)

    for (i, idx) in enumerate(neighbors)
        y = knn.y_buffer[idx]

        if knn.weighted
            d = distances[i]
            w = d > 0 ? T(1.0) / T(d) : T(1e10)
        else
            w = one(T)
        end

        vote_counts[y] = get(vote_counts, y, zero(T)) + w
        total_weight += w
    end

    # Normalize to probabilities
    if total_weight > 0
        for k in keys(vote_counts)
            vote_counts[k] /= total_weight
        end
    end

    return vote_counts
end

