# Online Logistic Regression with gradient-based optimization

using LinearAlgebra: dot, norm

"""
    LogisticRegression{T<:AbstractFloat, O} <: Learner{AbstractVector{T}, Int}

Online logistic regression using gradient descent with configurable optimizer.

# Parameters
- `optimizer`: Optimisers.jl optimizer (default: Adam(0.01))
- `l1`: L1 regularization strength (default: 0.0)
- `l2`: L2 regularization strength (default: 0.0)

# Example
```jldoctest
julia> model = LogisticRegression()
LogisticRegression: n=0 | value=(weights = Float64[], bias = 0.0)

julia> fit!(model, ([1.0, 2.0], 1));

julia> predict(model, [1.0, 2.0])
1

julia> nobs(model)
1
```
"""
mutable struct LogisticRegression{T<:AbstractFloat, O} <: Learner{AbstractVector{T}, Int}
    weights::Vector{T}
    bias::T
    optimizer::O
    opt_state::Any
    l1::T
    l2::T
    n::Int
    initialized::Bool

    function LogisticRegression{T}(;
        optimizer=Optimisers.Adam(0.01),
        l1::Real=0.0,
        l2::Real=0.0
    ) where {T<:AbstractFloat}
        new{T, typeof(optimizer)}(
            T[],
            zero(T),
            optimizer,
            nothing,
            T(l1),
            T(l2),
            0,
            false
        )
    end
end

LogisticRegression(; kwargs...) = LogisticRegression{Float64}(; kwargs...)

# Sigmoid activation
sigmoid(z) = 1 / (1 + exp(-z))

# Initialize weights on first observation
function initialize!(lr::LogisticRegression{T}, d::Int) where {T}
    lr.weights = zeros(T, d)
    lr.bias = zero(T)
    lr.opt_state = Optimisers.setup(lr.optimizer, (lr.weights, lr.bias))
    lr.initialized = true
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(lr::LogisticRegression{T}, xy::Tuple) where {T}
    x, y_raw = xy
    x = collect(T, x)

    # Convert label to {0, 1}
    y = y_raw <= 0 ? zero(T) : one(T)

    # Initialize on first observation
    if !lr.initialized
        initialize!(lr, length(x))
    end

    # Forward pass
    z = dot(lr.weights, x) + lr.bias
    p = sigmoid(z)

    # Gradient of binary cross-entropy loss
    error = p - y

    # Gradients
    grad_w = error .* x
    grad_b = error

    # Add regularization gradients
    if lr.l1 > 0
        grad_w .+= lr.l1 .* sign.(lr.weights)
    end
    if lr.l2 > 0
        grad_w .+= lr.l2 .* lr.weights
    end

    # Update using optimizer
    grads = (grad_w, grad_b)
    lr.opt_state, (lr.weights, lr.bias) = Optimisers.update!(
        lr.opt_state,
        (lr.weights, lr.bias),
        grads
    )

    lr.n += 1
    return lr
end

OnlineStatsBase.value(lr::LogisticRegression) = (weights=lr.weights, bias=lr.bias)
OnlineStatsBase.nobs(lr::LogisticRegression) = lr.n

function reset!(lr::LogisticRegression{T}) where {T}
    lr.weights = T[]
    lr.bias = zero(T)
    lr.opt_state = nothing
    lr.n = 0
    lr.initialized = false
    return lr
end

# Learner interface
function predict(lr::LogisticRegression, x::AbstractVector)
    if !lr.initialized
        return 0  # Default prediction when untrained
    end
    z = dot(lr.weights, x) + lr.bias
    return z >= 0 ? 1 : 0
end

function predict_proba(lr::LogisticRegression{T}, x::AbstractVector) where {T}
    if !lr.initialized
        return Dict(0 => T(0.5), 1 => T(0.5))
    end
    z = dot(lr.weights, x) + lr.bias
    p1 = sigmoid(z)
    return Dict(0 => one(T) - p1, 1 => p1)
end

# Convenience methods
"""
    coef(lr::LogisticRegression)

Return the weight vector.
"""
coef(lr::LogisticRegression) = lr.weights

"""
    intercept(lr::LogisticRegression)

Return the bias term.
"""
intercept(lr::LogisticRegression) = lr.bias
