# Leveraging Bagging - Enhanced online bagging

"""
    LeveragingBagging{L} <: Learner{AbstractVector, Any}

Leveraging Bagging ensemble with configurable lambda.

An enhanced version of online bagging that uses higher Poisson
resampling rates (λ > 1) to increase diversity among base learners.
The default λ=6 has been shown to improve performance.

# Parameters
- `base_learner` - A callable that creates a new learner instance
- `n_estimators::Int = 10` - Number of base learners
- `lambda::Float64 = 6.0` - Poisson resampling rate (higher = more diversity)

# Example
```jldoctest
julia> ensemble = LeveragingBagging(() -> HoeffdingTree(), n_estimators=3)
LeveragingBagging: n=0 | value=HoeffdingTree{Float64}[HoeffdingTree: n=0 | value=LeafNode{Float64}(Dict{Int64, SufficientStats{Float64}}(), CountMap: n=0 | value=OrderedDict{Int64, Int64}(), 0, 0), HoeffdingTree: n=0 | value=LeafNode{Float64}(Dict{Int64, SufficientStats{Float64}}(), CountMap: n=0 | value=OrderedDict{Int64, Int64}(), 0, 0), HoeffdingTree: n=0 | value=LeafNode{Float64}(Dict{Int64, SufficientStats{Float64}}(), CountMap: n=0 | value=OrderedDict{Int64, Int64}(), 0, 0)]

julia> fit!(ensemble, ([1.0, 2.0], 1));

julia> nobs(ensemble)
1
```

# Reference
Bifet, A., Holmes, G., & Pfahringer, B. (2010). Leveraging bagging for
evolving data streams. In Joint European conference on machine learning
and knowledge discovery in databases (pp. 135-150).
"""
mutable struct LeveragingBagging{L} <: Learner{AbstractVector, Any}
    learners::Vector{L}
    n_estimators::Int
    lambda::Float64
    n::Int
    rng::Random.AbstractRNG

    function LeveragingBagging(
        base_learner;
        n_estimators::Int = 10,
        lambda::Float64 = 6.0,
        rng::Random.AbstractRNG = Random.default_rng()
    )
        learners = [base_learner() for _ in 1:n_estimators]
        L = eltype(learners)
        new{L}(learners, n_estimators, lambda, 0, rng)
    end
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(lb::LeveragingBagging, xy::Tuple)
    x, y = xy

    for learner in lb.learners
        # Leveraging uses Poisson with λ > 1 for more diversity
        k = rand(lb.rng, Poisson(lb.lambda))
        for _ in 1:k
            fit!(learner, (x, y))
        end
    end

    lb.n += 1
    return lb
end

OnlineStatsBase.value(lb::LeveragingBagging) = lb.learners
OnlineStatsBase.nobs(lb::LeveragingBagging) = lb.n

function reset!(lb::LeveragingBagging)
    for learner in lb.learners
        reset!(learner)
    end
    lb.n = 0
    return lb
end

# Learner interface - majority voting
function predict(lb::LeveragingBagging, x)
    if lb.n == 0
        return 0
    end

    votes = Dict{Any, Int}()
    for learner in lb.learners
        pred = predict(learner, x)
        votes[pred] = get(votes, pred, 0) + 1
    end

    return dict_argmax(votes)
end

# Averaged probability predictions
function predict_proba(lb::LeveragingBagging, x)
    if lb.n == 0
        return Dict{Any, Float64}()
    end

    probs = Dict{Any, Float64}()

    for learner in lb.learners
        learner_probs = predict_proba(learner, x)
        for (class, prob) in learner_probs
            probs[class] = get(probs, class, 0.0) + prob
        end
    end

    # Average
    for class in keys(probs)
        probs[class] /= lb.n_estimators
    end

    return probs
end
