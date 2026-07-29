# Streaming K-Means

using Distances

"""
    StreamingKMeans{T<:AbstractFloat} <: UnsupervisedLearner{AbstractVector{T}}

Online K-Means clustering using mini-batch updates.

# Parameters
- `k::Int = 3` - Number of clusters
- `learning_rate::Float64 = 0.1` - Learning rate for centroid updates

# Example
```jldoctest
julia> model = StreamingKMeans(k=3)
StreamingKMeans: n=0 | value=Vector{Float64}[]

julia> fit!(model, [1.0, 2.0]);

julia> predict(model, [1.0, 2.0])
1

julia> nobs(model)
1
```
"""
mutable struct StreamingKMeans{T<:AbstractFloat} <: UnsupervisedLearner{AbstractVector{T}}
    k::Int
    learning_rate::T
    centroids::Vector{Vector{T}}
    counts::Vector{Int}
    n::Int

    function StreamingKMeans{T}(; k::Int=3, learning_rate::Real=0.1) where {T<:AbstractFloat}
        new{T}(k, T(learning_rate), Vector{T}[], zeros(Int, k), 0)
    end
end

StreamingKMeans(; kwargs...) = StreamingKMeans{Float64}(; kwargs...)

# OnlineStatsBase interface
function OnlineStatsBase._fit!(km::StreamingKMeans{T}, x) where {T}
    x = collect(T, x)

    if isempty(km.centroids)
        # Initialize first centroid
        push!(km.centroids, copy(x))
        km.counts[1] = 1
    elseif length(km.centroids) < km.k
        # Add new centroid if we haven't reached k yet
        push!(km.centroids, copy(x))
        km.counts[length(km.centroids)] = 1
    else
        # Find nearest centroid and update
        idx = find_nearest(km, x)
        update_centroid!(km, idx, x)
    end

    km.n += 1
    return km
end

OnlineStatsBase.value(km::StreamingKMeans) = km.centroids
OnlineStatsBase.nobs(km::StreamingKMeans) = km.n

function reset!(km::StreamingKMeans{T}) where {T}
    empty!(km.centroids)
    fill!(km.counts, 0)
    km.n = 0
    return km
end

# Find nearest centroid
function find_nearest(km::StreamingKMeans{T}, x::Vector{T}) where {T}
    min_dist = typemax(T)
    min_idx = 1

    for (i, c) in enumerate(km.centroids)
        d = Distances.evaluate(Euclidean(), x, c)
        if d < min_dist
            min_dist = d
            min_idx = i
        end
    end

    return min_idx
end

# Update centroid with learning rate
function update_centroid!(km::StreamingKMeans{T}, idx::Int, x::Vector{T}) where {T}
    km.counts[idx] += 1
    lr = km.learning_rate

    c = km.centroids[idx]
    for i in eachindex(c)
        c[i] = (one(T) - lr) * c[i] + lr * x[i]
    end
end

# Predict cluster assignment
function predict(km::StreamingKMeans{T}, x::AbstractVector) where {T}
    if isempty(km.centroids)
        return 0
    end
    x = collect(T, x)
    return find_nearest(km, x)
end

# Get all centroids
centroids(km::StreamingKMeans) = km.centroids

# Get cluster sizes
cluster_sizes(km::StreamingKMeans) = km.counts[1:length(km.centroids)]
