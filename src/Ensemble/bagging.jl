# Online Bagging with Poisson resampling

"""
    Bagging{L} <: Learner{AbstractVector, Any}

Online bagging ensemble using Poisson(λ) resampling.

Each observation is used to update each base learner k times, where k ~ Poisson(λ).
This approximates bootstrap sampling in the online setting.

# Parameters
- `base_learner` - A callable that creates a new learner instance
- `n_estimators::Int = 10` - Number of base learners
- `lambda::Float64 = 1.0` - Poisson resampling rate (λ=1 approximates bootstrap)

# Example
```jldoctest
julia> bag = Bagging(() -> LogisticRegression(), n_estimators=3)
Bagging: n=0 | value=LogisticRegression{Float64, Adam}[LogisticRegression: n=0 | value=(weights = Float64[], bias = 0.0), LogisticRegression: n=0 | value=(weights = Float64[], bias = 0.0), LogisticRegression: n=0 | value=(weights = Float64[], bias = 0.0)]

julia> fit!(bag, ([1.0, 2.0], 1));

julia> nobs(bag)
1
```
"""
mutable struct Bagging{L} <: Learner{AbstractVector, Any}
    learners::Vector{L}
    n_estimators::Int
    lambda::Float64
    n::Int
    rng::Random.AbstractRNG

    function Bagging(base_learner;
                     n_estimators::Int = 10,
                     lambda::Float64 = 1.0,
                     rng::Random.AbstractRNG = Random.default_rng())
        n_estimators > 0 || throw(ArgumentError("n_estimators must be positive"))
        isfinite(lambda) && lambda > 0 ||
            throw(ArgumentError("lambda must be finite and positive"))
        learners = [base_learner() for _ in 1:n_estimators]
        L = eltype(learners)
        new{L}(learners, n_estimators, lambda, 0, rng)
    end
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(bag::Bagging, xy::Tuple)
    x, y = xy

    for learner in bag.learners
        # Poisson resampling: use this observation k times
        k = rand(bag.rng, Poisson(bag.lambda))
        for _ in 1:k
            fit!(learner, (x, y))  # Pass as tuple for OnlineStatsBase
        end
    end

    bag.n += 1
    return bag
end

OnlineStatsBase.value(bag::Bagging) = bag.learners
OnlineStatsBase.nobs(bag::Bagging) = bag.n

function reset!(bag::Bagging)
    for learner in bag.learners
        reset!(learner)
    end
    bag.n = 0
    return bag
end

# Learner interface - majority voting
function predict(bag::Bagging, x)
    if bag.n == 0
        return 0
    end

    votes = Dict{Any, Int}()
    for learner in bag.learners
        pred = predict(learner, x)
        votes[pred] = get(votes, pred, 0) + 1
    end

    return dict_argmax(votes)
end

# Averaged probability predictions
function predict_proba(bag::Bagging, x)
    if bag.n == 0
        return Dict{Any, Float64}()
    end

    probs = Dict{Any, Float64}()
    contributors = 0

    for learner in bag.learners
        learner_probs = predict_proba(learner, x)
        isempty(learner_probs) && continue
        contributors += 1
        for (class, prob) in learner_probs
            probs[class] = get(probs, class, 0.0) + prob
        end
    end

    contributors == 0 && return probs

    # Average over learners that can issue a probability distribution.
    for class in keys(probs)
        probs[class] /= contributors
    end

    return probs
end

# Simple Poisson distribution sampling
struct Poisson
    lambda::Float64
end

function Base.rand(rng::Random.AbstractRNG, p::Poisson)
    # Knuth algorithm for Poisson
    L = exp(-p.lambda)
    k = 0
    prob = 1.0
    while prob > L
        k += 1
        prob *= rand(rng)
    end
    return k - 1
end
