# Online Perceptron classifier

using LinearAlgebra: dot

"""
    Perceptron{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}

Online Perceptron classifier with configurable learning rate.

# Parameters
- `η`: Learning rate (default: 1.0)

# Example
```jldoctest
julia> model = Perceptron(η=0.5)
Perceptron: n=0 | value=(weights = Float64[], bias = 0.0)

julia> fit!(model, ([1.0, 2.0], 1));

julia> predict(model, [1.0, 2.0])
1

julia> nobs(model)
1
```
"""
mutable struct Perceptron{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}
    weights::Vector{T}
    bias::T
    η::T
    n::Int
    initialized::Bool

    function Perceptron{T}(; η::Real=1.0) where {T<:AbstractFloat}
        new{T}(T[], zero(T), T(η), 0, false)
    end
end

Perceptron(; kwargs...) = Perceptron{Float64}(; kwargs...)

# Initialize weights on first observation
function initialize!(p::Perceptron{T}, d::Int) where {T}
    p.weights = zeros(T, d)
    p.bias = zero(T)
    p.initialized = true
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(p::Perceptron{T}, xy::Tuple) where {T}
    x, y_raw = xy
    x = collect(T, x)

    # Convert label to {-1, 1}
    y = y_raw <= 0 ? -one(T) : one(T)

    # Initialize on first observation
    if !p.initialized
        initialize!(p, length(x))
    end

    # Perceptron update rule: only update on misclassification
    z = dot(p.weights, x) + p.bias
    y_pred = z >= 0 ? one(T) : -one(T)

    if y_pred != y
        p.weights .+= p.η .* y .* x
        p.bias += p.η * y
    end

    p.n += 1
    return p
end

OnlineStatsBase.value(p::Perceptron) = (weights=p.weights, bias=p.bias)
OnlineStatsBase.nobs(p::Perceptron) = p.n

function reset!(p::Perceptron{T}) where {T}
    p.weights = T[]
    p.bias = zero(T)
    p.n = 0
    p.initialized = false
    return p
end

# Learner interface
function predict(p::Perceptron, x::AbstractVector)
    if !p.initialized
        return 0
    end
    z = dot(p.weights, x) + p.bias
    return z >= 0 ? 1 : 0
end

function predict_proba(p::Perceptron{T}, x::AbstractVector) where {T}
    # Perceptron doesn't provide probabilities; return hard assignment
    pred = predict(p, x)
    return Dict(0 => pred == 0 ? one(T) : zero(T),
                1 => pred == 1 ? one(T) : zero(T))
end

# Convenience methods
coef(p::Perceptron) = p.weights
intercept(p::Perceptron) = p.bias
