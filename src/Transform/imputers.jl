# Online imputers for missing value handling

using OnlineStats: Mean, CountMap

"""
    MeanImputer{T<:AbstractFloat} <: OnlineStat{Union{T, Missing}}

Online imputer that replaces missing values with the running mean.

# Example
```jldoctest
julia> imputer = MeanImputer()
MeanImputer: n=0 | value=0.0

julia> fit!(imputer, 1.0);

julia> fit!(imputer, 2.0);

julia> fit!(imputer, 4.0);

julia> nobs(imputer)
3
```
"""
mutable struct MeanImputer{T<:AbstractFloat} <: OnlineStat{Union{T, Missing}}
    stat::Mean{T}
    n::Int

    function MeanImputer{T}() where {T<:AbstractFloat}
        new{T}(Mean(T), 0)
    end
end

MeanImputer() = MeanImputer{Float64}()

function OnlineStatsBase._fit!(imp::MeanImputer{T}, x) where {T}
    if !ismissing(x)
        fit!(imp.stat, T(x))
    end
    imp.n += 1
    return imp
end

OnlineStatsBase.value(imp::MeanImputer) = OnlineStats.mean(imp.stat)
OnlineStatsBase.nobs(imp::MeanImputer) = imp.n

function reset!(imp::MeanImputer{T}) where {T}
    imp.stat = Mean(T)
    imp.n = 0
    return imp
end

function transform(imp::MeanImputer{T}, x) where {T}
    if ismissing(x)
        return T(OnlineStats.mean(imp.stat))
    else
        return T(x)
    end
end

"""
    ModeImputer <: OnlineStat{Union{Any, Missing}}

Online imputer that replaces missing values with the running mode (most frequent value).

# Example
```jldoctest
julia> imputer = ModeImputer()
ModeImputer: n=0 | value=missing

julia> fit!(imputer, :a);

julia> fit!(imputer, :b);

julia> fit!(imputer, :a);

julia> fit!(imputer, :a);

julia> transform(imputer, missing)
:a

julia> nobs(imputer)
4
```
"""
mutable struct ModeImputer <: OnlineStat{Any}
    stat::CountMap{Any}
    n::Int

    function ModeImputer()
        new(CountMap{Any}(), 0)
    end
end

function OnlineStatsBase._fit!(imp::ModeImputer, x)
    if !ismissing(x)
        fit!(imp.stat, x)
    end
    imp.n += 1
    return imp
end

OnlineStatsBase.value(imp::ModeImputer) = mode(imp)
OnlineStatsBase.nobs(imp::ModeImputer) = imp.n

function reset!(imp::ModeImputer)
    imp.stat = CountMap{Any}()
    imp.n = 0
    return imp
end

"""
    mode(imp::ModeImputer)

Return the most frequent value seen by the imputer.
"""
function mode(imp::ModeImputer)
    counts = OnlineStatsBase.value(imp.stat)
    if isempty(counts)
        return missing
    end
    # Find key with maximum count
    max_count = 0
    max_key = missing
    for (k, v) in counts
        if v > max_count
            max_count = v
            max_key = k
        end
    end
    return max_key
end

function transform(imp::ModeImputer, x)
    if ismissing(x)
        return mode(imp)
    else
        return x
    end
end

export MeanImputer, ModeImputer, mode
