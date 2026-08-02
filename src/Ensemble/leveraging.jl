# High-rate Poisson bagging

"""
    HighRatePoissonBagging{L} <: Learner{AbstractVector, Any}

Online bagging ensemble with a configurable Poisson resampling rate.

This implementation only provides the higher-rate Poisson resampling component
associated with leveraging bagging. It does not implement random output codes,
per-learner ADWIN detectors, or replacement of the worst learner after drift,
and therefore does not claim conformance with the full Leveraging Bagging
algorithm.

# Parameters
- `base_learner` - A callable that creates a new learner instance
- `n_estimators::Int = 10` - Number of base learners
- `lambda::Float64 = 6.0` - Poisson resampling rate (higher = more diversity)

# Example
```jldoctest
julia> ensemble = HighRatePoissonBagging(() -> HoeffdingTree(), n_estimators=3)
HighRatePoissonBagging: n=0 | value=HoeffdingTree{Float64}[HoeffdingTree: n=0 | value=LeafNode{Float64}(Dict{Int64, SufficientStats{Float64}}(), CountMap: n=0 | value=OrderedDict{Int64, Int64}(), 0, 0), HoeffdingTree: n=0 | value=LeafNode{Float64}(Dict{Int64, SufficientStats{Float64}}(), CountMap: n=0 | value=OrderedDict{Int64, Int64}(), 0, 0), HoeffdingTree: n=0 | value=LeafNode{Float64}(Dict{Int64, SufficientStats{Float64}}(), CountMap: n=0 | value=OrderedDict{Int64, Int64}(), 0, 0)]

julia> fit!(ensemble, ([1.0, 2.0], 1));

julia> nobs(ensemble)
1
```

# Reference
Bifet, A., Holmes, G., & Pfahringer, B. (2010). Leveraging bagging for
evolving data streams. In Joint European conference on machine learning
and knowledge discovery in databases (pp. 135-150).
"""
mutable struct HighRatePoissonBagging{L} <: Learner{AbstractVector, Any}
    learners::Vector{L}
    n_estimators::Int
    lambda::Float64
    n::Int
    rng::Random.AbstractRNG

    function HighRatePoissonBagging(
        base_learner;
        n_estimators::Int = 10,
        lambda::Float64 = 6.0,
        rng::Random.AbstractRNG = Random.default_rng()
    )
        n_estimators > 0 || throw(ArgumentError("n_estimators must be positive"))
        isfinite(lambda) && lambda > 0 ||
            throw(ArgumentError("lambda must be finite and positive"))
        learners = [base_learner() for _ in 1:n_estimators]
        L = eltype(learners)
        new{L}(learners, n_estimators, lambda, 0, rng)
    end
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(lb::HighRatePoissonBagging, xy::Tuple)
    x, y = xy

    for learner in lb.learners
        k = rand(lb.rng, Poisson(lb.lambda))
        for _ in 1:k
            fit!(learner, (x, y))
        end
    end

    lb.n += 1
    return lb
end

OnlineStatsBase.value(lb::HighRatePoissonBagging) = lb.learners
OnlineStatsBase.nobs(lb::HighRatePoissonBagging) = lb.n

function reset!(lb::HighRatePoissonBagging)
    for learner in lb.learners
        reset!(learner)
    end
    lb.n = 0
    return lb
end

# Learner interface - majority voting
function predict(lb::HighRatePoissonBagging, x)
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
function predict_proba(lb::HighRatePoissonBagging, x)
    if lb.n == 0
        return Dict{Any, Float64}()
    end

    probs = Dict{Any, Float64}()
    contributors = 0

    for learner in lb.learners
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

"""
    LeveragingBagging

Compatibility alias for [`HighRatePoissonBagging`](@ref). The alias does not
imply conformance with the full Leveraging Bagging algorithm.
"""
const LeveragingBagging = HighRatePoissonBagging
