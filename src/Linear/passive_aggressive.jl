# Passive-Aggressive classifiers (PA-I, PA-II)

using LinearAlgebra: dot, norm

"""
    PassiveAggressive{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}

Online Passive-Aggressive classifier.

# Variants
- `:pa1` (PA-I): Clipped aggressiveness with max step size C
- `:pa2` (PA-II): Smooth aggressiveness with regularization parameter C

# Parameters
- `C`: Aggressiveness parameter (default: 1.0)
- `mode`: `:pa1` or `:pa2` (default: `:pa1`)

# Example
```jldoctest
julia> model = PassiveAggressive(C=0.5, mode=:pa2)
PassiveAggressive: n=0 | value=(weights = Float64[], bias = 0.0)

julia> fit!(model, ([1.0, 2.0], 1));

julia> predict(model, [1.0, 2.0])
1

julia> nobs(model)
1
```

# Reference
Crammer, K., et al. "Online passive-aggressive algorithms." JMLR 2006.
"""
mutable struct PassiveAggressive{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}
    weights::Vector{T}
    bias::T
    C::T
    mode::Symbol
    n::Int
    initialized::Bool

    function PassiveAggressive{T}(; C::Real=1.0, mode::Symbol=:pa1) where {T<:AbstractFloat}
        @assert mode in (:pa1, :pa2) "mode must be :pa1 or :pa2"
        new{T}(T[], zero(T), T(C), mode, 0, false)
    end
end

PassiveAggressive(; kwargs...) = PassiveAggressive{Float64}(; kwargs...)

# Initialize weights on first observation
function initialize!(pa::PassiveAggressive{T}, d::Int) where {T}
    pa.weights = zeros(T, d)
    pa.bias = zero(T)
    pa.initialized = true
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(pa::PassiveAggressive{T}, xy::Tuple) where {T}
    x, y_raw = xy
    x = collect(T, x)

    # Convert label to {-1, 1}
    y = y_raw <= 0 ? -one(T) : one(T)

    # Initialize on first observation
    if !pa.initialized
        initialize!(pa, length(x))
    end

    # Compute hinge loss
    z = dot(pa.weights, x) + pa.bias
    margin = y * z
    loss = max(zero(T), one(T) - margin)

    if loss > 0
        # Compute update step size
        x_norm_sq = dot(x, x) + one(T)  # +1 for bias term

        if pa.mode == :pa1
            # PA-I: τ = min(C, loss / ||x||²)
            τ = min(pa.C, loss / x_norm_sq)
        else
            # PA-II: τ = loss / (||x||² + 1/(2C))
            τ = loss / (x_norm_sq + one(T) / (2 * pa.C))
        end

        # Update
        pa.weights .+= τ .* y .* x
        pa.bias += τ * y
    end

    pa.n += 1
    return pa
end

OnlineStatsBase.value(pa::PassiveAggressive) = (weights=pa.weights, bias=pa.bias)
OnlineStatsBase.nobs(pa::PassiveAggressive) = pa.n

function reset!(pa::PassiveAggressive{T}) where {T}
    pa.weights = T[]
    pa.bias = zero(T)
    pa.n = 0
    pa.initialized = false
    return pa
end

# Learner interface
function predict(pa::PassiveAggressive, x::AbstractVector)
    if !pa.initialized
        return 0
    end
    z = dot(pa.weights, x) + pa.bias
    return z >= 0 ? 1 : 0
end

function predict_proba(pa::PassiveAggressive{T}, x::AbstractVector) where {T}
    # PA doesn't naturally provide probabilities
    pred = predict(pa, x)
    return Dict(0 => pred == 0 ? one(T) : zero(T),
                1 => pred == 1 ? one(T) : zero(T))
end

# Convenience methods
coef(pa::PassiveAggressive) = pa.weights
intercept(pa::PassiveAggressive) = pa.bias
