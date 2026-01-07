# Online feature scalers

"""
    StandardScaler{T<:AbstractFloat} <: Transformer{AbstractVector{T}}

Online standardization (z-score normalization).

Transforms features to have zero mean and unit variance.

# Example
```jldoctest
julia> scaler = StandardScaler()
StandardScaler: n=0 | value=Variance[]

julia> fit!(scaler, [1.0, 2.0]);

julia> transform(scaler, [1.0, 2.0])
2-element Vector{Float64}:
 0.0
 0.0

julia> nobs(scaler)
1
```
"""
mutable struct StandardScaler{T<:AbstractFloat} <: Transformer{AbstractVector{T}}
    stats::Vector{Variance}
    n::Int

    function StandardScaler{T}() where {T<:AbstractFloat}
        new{T}(Variance[], 0)
    end
end

StandardScaler() = StandardScaler{Float64}()

function OnlineStatsBase._fit!(scaler::StandardScaler{T}, x) where {T}
    x = collect(T, x)

    # Initialize stats on first observation
    if isempty(scaler.stats)
        scaler.stats = [Variance(T) for _ in 1:length(x)]
    end

    # Expand if needed
    while length(scaler.stats) < length(x)
        push!(scaler.stats, Variance(T))
    end

    for (i, xi) in enumerate(x)
        fit!(scaler.stats[i], xi)
    end

    scaler.n += 1
    return scaler
end

OnlineStatsBase.value(scaler::StandardScaler) = scaler.stats
OnlineStatsBase.nobs(scaler::StandardScaler) = scaler.n

function reset!(scaler::StandardScaler{T}) where {T}
    empty!(scaler.stats)
    scaler.n = 0
    return scaler
end

function transform(scaler::StandardScaler{T}, x::AbstractVector) where {T}
    if isempty(scaler.stats)
        return collect(T, x)
    end

    x = collect(T, x)
    result = similar(x)

    for i in eachindex(x)
        if i <= length(scaler.stats) && nobs(scaler.stats[i]) > 0
            μ = mean(scaler.stats[i])
            σ = sqrt(var(scaler.stats[i]))
            result[i] = σ > 0 ? (x[i] - μ) / σ : zero(T)
        else
            result[i] = x[i]
        end
    end

    return result
end

# Inverse transform
function inverse_transform(scaler::StandardScaler{T}, x::AbstractVector) where {T}
    if isempty(scaler.stats)
        return collect(T, x)
    end

    x = collect(T, x)
    result = similar(x)

    for i in eachindex(x)
        if i <= length(scaler.stats) && nobs(scaler.stats[i]) > 0
            μ = mean(scaler.stats[i])
            σ = sqrt(var(scaler.stats[i]))
            result[i] = x[i] * σ + μ
        else
            result[i] = x[i]
        end
    end

    return result
end

"""
    MinMaxScaler{T<:AbstractFloat} <: Transformer{AbstractVector{T}}

Online min-max scaling to [0, 1] range.

# Example
```jldoctest
julia> scaler = MinMaxScaler()
MinMaxScaler: n=0 | value=Extrema[]

julia> fit!(scaler, [1.0, 2.0]);

julia> transform(scaler, [1.0, 2.0])
2-element Vector{Float64}:
 0.0
 0.0

julia> nobs(scaler)
1
```
"""
mutable struct MinMaxScaler{T<:AbstractFloat} <: Transformer{AbstractVector{T}}
    extrema::Vector{Extrema}
    n::Int

    function MinMaxScaler{T}() where {T<:AbstractFloat}
        new{T}(Extrema[], 0)
    end
end

MinMaxScaler() = MinMaxScaler{Float64}()

function OnlineStatsBase._fit!(scaler::MinMaxScaler{T}, x) where {T}
    x = collect(T, x)

    if isempty(scaler.extrema)
        scaler.extrema = [Extrema(T) for _ in 1:length(x)]
    end

    while length(scaler.extrema) < length(x)
        push!(scaler.extrema, Extrema(T))
    end

    for (i, xi) in enumerate(x)
        fit!(scaler.extrema[i], xi)
    end

    scaler.n += 1
    return scaler
end

OnlineStatsBase.value(scaler::MinMaxScaler) = scaler.extrema
OnlineStatsBase.nobs(scaler::MinMaxScaler) = scaler.n

function reset!(scaler::MinMaxScaler{T}) where {T}
    empty!(scaler.extrema)
    scaler.n = 0
    return scaler
end

function transform(scaler::MinMaxScaler{T}, x::AbstractVector) where {T}
    if isempty(scaler.extrema)
        return collect(T, x)
    end

    x = collect(T, x)
    result = similar(x)

    for i in eachindex(x)
        if i <= length(scaler.extrema) && nobs(scaler.extrema[i]) > 0
            min_val, max_val = value(scaler.extrema[i])
            range = max_val - min_val
            result[i] = range > 0 ? (x[i] - min_val) / range : zero(T)
        else
            result[i] = x[i]
        end
    end

    return result
end
