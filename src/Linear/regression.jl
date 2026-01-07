# Online Linear Regression wrapping OnlineStats.LinReg

"""
    Regression{T<:AbstractFloat} <: Learner{AbstractVector{T}, T}

Online linear regression using recursive least squares.
Wraps `OnlineStats.LinReg` for incremental fitting.

# Example
```jldoctest
julia> model = Regression()
Regression: n=0 | value=Float64[]

julia> fit!(model, ([1.0, 2.0], 3.0));

julia> nobs(model)
1
```
"""
mutable struct Regression{T<:AbstractFloat} <: Learner{AbstractVector{T}, T}
    stat::LinReg

    function Regression{T}() where {T<:AbstractFloat}
        new{T}(LinReg())
    end
end

Regression() = Regression{Float64}()

# OnlineStatsBase interface
function OnlineStatsBase._fit!(r::Regression, xy::Tuple)
    x, y = xy
    # Add 1.0 at the beginning for intercept term
    x_with_intercept = vcat(1.0, collect(x))
    fit!(r.stat, (x_with_intercept, y))
    return r
end

OnlineStatsBase.value(r::Regression) = OnlineStats.coef(r.stat)
OnlineStatsBase.nobs(r::Regression) = nobs(r.stat)

function OnlineStatsBase._merge!(r1::Regression, r2::Regression)
    merge!(r1.stat, r2.stat)
    return r1
end

function reset!(r::Regression{T}) where {T}
    r.stat = LinReg()
    return r
end

# Learner interface - extends OnlineML.predict
function predict(r::Regression, x::AbstractVector)
    coefs = OnlineStats.coef(r.stat)
    if length(coefs) == 0
        return zero(eltype(x))
    end
    # coef includes intercept as first element
    return coefs[1] + dot(coefs[2:end], x)
end

# Convenience methods
"""
    coef(r::Regression)

Return the regression coefficients (intercept, then feature weights).
"""
coef(r::Regression) = OnlineStats.coef(r.stat)

"""
    intercept(r::Regression)

Return the intercept term.
"""
intercept(r::Regression) = coef(r)[1]

"""
    weights(r::Regression)

Return the feature weights (excluding intercept).
"""
weights(r::Regression) = coef(r)[2:end]

# Helper for dot product
using LinearAlgebra: dot

export coef, intercept, weights
