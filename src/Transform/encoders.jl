# Online encoders for categorical features

using OnlineStats: CountMap

"""
    OneHotEncoder <: OnlineStat{Any}

Online one-hot encoder that dynamically expands to new categories.

# Example
```jldoctest
julia> encoder = OneHotEncoder()
OneHotEncoder: n=0 | value=Any[]

julia> fit!(encoder, :a);

julia> fit!(encoder, :b);

julia> fit!(encoder, :c);

julia> transform(encoder, :b)
3-element Vector{Float64}:
 0.0
 1.0
 0.0

julia> nobs(encoder)
3
```
"""
mutable struct OneHotEncoder <: OnlineStat{Any}
    categories::Vector{Any}
    category_map::Dict{Any, Int}
    n::Int

    function OneHotEncoder()
        new(Any[], Dict{Any, Int}(), 0)
    end
end

function OnlineStatsBase._fit!(enc::OneHotEncoder, x)
    if !haskey(enc.category_map, x)
        push!(enc.categories, x)
        enc.category_map[x] = length(enc.categories)
    end
    enc.n += 1
    return enc
end

OnlineStatsBase.value(enc::OneHotEncoder) = enc.categories
OnlineStatsBase.nobs(enc::OneHotEncoder) = enc.n

function reset!(enc::OneHotEncoder)
    empty!(enc.categories)
    empty!(enc.category_map)
    enc.n = 0
    return enc
end

function transform(enc::OneHotEncoder, x)
    n_cats = length(enc.categories)
    result = zeros(Float64, n_cats)
    if haskey(enc.category_map, x)
        result[enc.category_map[x]] = 1.0
    end
    return result
end

"""
    OrdinalEncoder <: OnlineStat{Any}

Online ordinal encoder that maps categories to integers.

# Example
```jldoctest
julia> encoder = OrdinalEncoder()
OrdinalEncoder: n=0 | value=Any[]

julia> fit!(encoder, :low);

julia> fit!(encoder, :medium);

julia> fit!(encoder, :high);

julia> transform(encoder, :medium)
2.0

julia> nobs(encoder)
3
```
"""
mutable struct OrdinalEncoder <: OnlineStat{Any}
    categories::Vector{Any}
    category_map::Dict{Any, Int}
    n::Int

    function OrdinalEncoder()
        new(Any[], Dict{Any, Int}(), 0)
    end
end

function OnlineStatsBase._fit!(enc::OrdinalEncoder, x)
    if !haskey(enc.category_map, x)
        push!(enc.categories, x)
        enc.category_map[x] = length(enc.categories)
    end
    enc.n += 1
    return enc
end

OnlineStatsBase.value(enc::OrdinalEncoder) = enc.categories
OnlineStatsBase.nobs(enc::OrdinalEncoder) = enc.n

function reset!(enc::OrdinalEncoder)
    empty!(enc.categories)
    empty!(enc.category_map)
    enc.n = 0
    return enc
end

function transform(enc::OrdinalEncoder, x)
    if haskey(enc.category_map, x)
        return Float64(enc.category_map[x])
    else
        return 0.0  # Unknown category
    end
end

"""
    TargetEncoder{Y} <: OnlineStat{Tuple{Any, Y}}

Online target encoder with smoothing for categorical features.

Encodes categories based on the mean target value for that category,
with Bayesian smoothing to handle rare categories.

# Parameters
- `smoothing::Float64 = 1.0` - Smoothing parameter (higher = more regularization)

# Example
```jldoctest
julia> encoder = TargetEncoder{Float64}(smoothing=10.0)
TargetEncoder: n=0 | value=(sums = Dict{Any, Float64}(), counts = Dict{Any, Int64}())

julia> fit!(encoder, (:a, 1.0));

julia> fit!(encoder, (:b, 0.0));

julia> nobs(encoder)
2
```
"""
mutable struct TargetEncoder{Y} <: OnlineStat{Tuple{Any, Y}}
    smoothing::Float64
    global_sum::Float64
    global_count::Int
    category_sums::Dict{Any, Float64}
    category_counts::Dict{Any, Int}
    n::Int

    function TargetEncoder{Y}(; smoothing::Float64=1.0) where {Y}
        new{Y}(smoothing, 0.0, 0, Dict{Any, Float64}(), Dict{Any, Int}(), 0)
    end
end

TargetEncoder(; kwargs...) = TargetEncoder{Float64}(; kwargs...)

function OnlineStatsBase._fit!(enc::TargetEncoder, xy::Tuple)
    x, y = xy
    y_val = Float64(y)

    # Update global statistics
    enc.global_sum += y_val
    enc.global_count += 1

    # Update category statistics
    if haskey(enc.category_counts, x)
        enc.category_sums[x] += y_val
        enc.category_counts[x] += 1
    else
        enc.category_sums[x] = y_val
        enc.category_counts[x] = 1
    end

    enc.n += 1
    return enc
end

OnlineStatsBase.value(enc::TargetEncoder) = (sums=enc.category_sums, counts=enc.category_counts)
OnlineStatsBase.nobs(enc::TargetEncoder) = enc.n

function reset!(enc::TargetEncoder)
    enc.global_sum = 0.0
    enc.global_count = 0
    empty!(enc.category_sums)
    empty!(enc.category_counts)
    enc.n = 0
    return enc
end

function transform(enc::TargetEncoder, x)
    if enc.global_count == 0
        return 0.0
    end

    global_mean = enc.global_sum / enc.global_count

    if haskey(enc.category_counts, x)
        cat_count = enc.category_counts[x]
        cat_mean = enc.category_sums[x] / cat_count
        # Bayesian smoothing: weighted average of category mean and global mean
        weight = cat_count / (cat_count + enc.smoothing)
        return weight * cat_mean + (1 - weight) * global_mean
    else
        return global_mean  # Unknown category gets global mean
    end
end

export OneHotEncoder, OrdinalEncoder, TargetEncoder
