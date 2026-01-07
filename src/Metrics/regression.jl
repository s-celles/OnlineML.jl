# Online regression metrics

"""
    MAE{T<:AbstractFloat} <: Metric{T}

Mean Absolute Error for regression.

MAE = (1/n) * Σ|y_pred - y_true|

# Example
```jldoctest
julia> metric = MAE()
MAE: n=0 | value=0.0

julia> fit!(metric, (1.0, 2.0));

julia> fit!(metric, (3.0, 2.0));

julia> value(metric)
1.0

julia> nobs(metric)
2
```
"""
mutable struct MAE{T<:AbstractFloat} <: Metric{T}
    sum_error::T
    n::Int

    MAE{T}() where {T<:AbstractFloat} = new{T}(zero(T), 0)
end

MAE() = MAE{Float64}()

function OnlineStatsBase._fit!(mae::MAE{T}, xy::Tuple) where {T}
    y_pred, y_true = xy
    mae.sum_error += abs(T(y_pred) - T(y_true))
    mae.n += 1
    return mae
end

OnlineStatsBase.value(mae::MAE) = mae.n > 0 ? mae.sum_error / mae.n : 0.0
OnlineStatsBase.nobs(mae::MAE) = mae.n

function reset!(mae::MAE{T}) where {T}
    mae.sum_error = zero(T)
    mae.n = 0
    return mae
end


"""
    MSE{T<:AbstractFloat} <: Metric{T}

Mean Squared Error for regression.

MSE = (1/n) * Σ(y_pred - y_true)²

# Example
```jldoctest
julia> metric = MSE()
MSE: n=0 | value=0.0

julia> fit!(metric, (1.0, 2.0));

julia> fit!(metric, (3.0, 2.0));

julia> value(metric)
1.0

julia> nobs(metric)
2
```
"""
mutable struct MSE{T<:AbstractFloat} <: Metric{T}
    sum_squared_error::T
    n::Int

    MSE{T}() where {T<:AbstractFloat} = new{T}(zero(T), 0)
end

MSE() = MSE{Float64}()

function OnlineStatsBase._fit!(mse::MSE{T}, xy::Tuple) where {T}
    y_pred, y_true = xy
    error = T(y_pred) - T(y_true)
    mse.sum_squared_error += error * error
    mse.n += 1
    return mse
end

OnlineStatsBase.value(mse::MSE) = mse.n > 0 ? mse.sum_squared_error / mse.n : 0.0
OnlineStatsBase.nobs(mse::MSE) = mse.n

function reset!(mse::MSE{T}) where {T}
    mse.sum_squared_error = zero(T)
    mse.n = 0
    return mse
end


"""
    RMSE{T<:AbstractFloat} <: Metric{T}

Root Mean Squared Error for regression.

RMSE = √MSE

# Example
```jldoctest
julia> metric = RMSE()
RMSE: n=0 | value=0.0

julia> fit!(metric, (1.0, 2.0));

julia> fit!(metric, (3.0, 2.0));

julia> value(metric)
1.0

julia> nobs(metric)
2
```
"""
mutable struct RMSE{T<:AbstractFloat} <: Metric{T}
    mse::MSE{T}

    RMSE{T}() where {T<:AbstractFloat} = new{T}(MSE{T}())
end

RMSE() = RMSE{Float64}()

function OnlineStatsBase._fit!(rmse::RMSE{T}, xy::Tuple) where {T}
    fit!(rmse.mse, xy)
    return rmse
end

OnlineStatsBase.value(rmse::RMSE) = sqrt(value(rmse.mse))
OnlineStatsBase.nobs(rmse::RMSE) = nobs(rmse.mse)

function reset!(rmse::RMSE)
    reset!(rmse.mse)
    return rmse
end


"""
    R2{T<:AbstractFloat} <: Metric{T}

Online R² (coefficient of determination) using Welford's algorithm.

R² = 1 - SS_res / SS_tot = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²

Uses running mean and variance to compute SS_tot incrementally.

# Example
```jldoctest
julia> metric = R2()
R2: n=0 | value=NaN

julia> fit!(metric, (1.0, 1.0));

julia> fit!(metric, (2.0, 2.0));

julia> fit!(metric, (3.0, 3.0));

julia> value(metric)
1.0

julia> nobs(metric)
3
```
"""
mutable struct R2{T<:AbstractFloat} <: Metric{T}
    ss_res::T      # Sum of squared residuals
    mean_y::T      # Running mean of y_true
    m2_y::T        # Running M2 for variance of y_true (Welford)
    n::Int

    R2{T}() where {T<:AbstractFloat} = new{T}(zero(T), zero(T), zero(T), 0)
end

R2() = R2{Float64}()

function OnlineStatsBase._fit!(r2::R2{T}, xy::Tuple) where {T}
    y_pred, y_true = T(xy[1]), T(xy[2])
    r2.n += 1

    # Update residual sum
    residual = y_true - y_pred
    r2.ss_res += residual * residual

    # Update running mean and M2 (Welford's algorithm)
    delta = y_true - r2.mean_y
    r2.mean_y += delta / r2.n
    delta2 = y_true - r2.mean_y
    r2.m2_y += delta * delta2

    return r2
end

function OnlineStatsBase.value(r2::R2)
    if r2.n < 2
        return NaN
    end
    ss_tot = r2.m2_y  # M2 is sum of squared deviations from mean
    if ss_tot <= 0
        return 1.0  # Perfect prediction when no variance in target
    end
    return 1.0 - r2.ss_res / ss_tot
end

OnlineStatsBase.nobs(r2::R2) = r2.n

function reset!(r2::R2{T}) where {T}
    r2.ss_res = zero(T)
    r2.mean_y = zero(T)
    r2.m2_y = zero(T)
    r2.n = 0
    return r2
end
