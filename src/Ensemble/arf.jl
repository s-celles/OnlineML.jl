# Adaptive Random Forest (ARF) - Drift-aware ensemble

"""
    DriftAwareEstimator{L,D}

A single estimator within `DriftAwareBagging`.
Contains the model, its drift detector, warning detector,
and optionally a background model.
"""
mutable struct DriftAwareEstimator{L,D}
    model::L
    drift_detector::D
    warning_detector::D
    background_model::Union{L, Nothing}
    n_drifts::Int
    in_warning::Bool

    function DriftAwareEstimator(base_learner, detector_factory)
        model = base_learner()
        drift_detector = detector_factory()
        warning_detector = detector_factory()
        L = typeof(model)
        D = typeof(drift_detector)
        new{L,D}(model, drift_detector, warning_detector, nothing, 0, false)
    end
end

"""
    DriftAwareBagging{L,D} <: Learner{AbstractVector, Any}

Poisson online bagging with per-learner drift adaptation.

`DriftAwareBagging` combines:
- Online bagging with Poisson resampling
- Per-tree drift detection
- Background tree replacement on drift

When drift is detected on a learner, it is replaced with a background
learner that has been training since the warning was first detected.

This implementation does not perform random feature-subspace selection and is
therefore not an implementation of Adaptive Random Forest. The historical name
`AdaptiveRandomForest` remains as a compatibility alias.

# Parameters
- `base_learner` - A callable that creates a new learner instance
- `detector` - A callable that creates a new drift detector instance
- `n_estimators::Int = 10` - Number of base learners
- `lambda::Float64 = 6.0` - Poisson resampling rate

# Example
```jldoctest
julia> ensemble = DriftAwareBagging(() -> HoeffdingTree(), () -> DDM(), n_estimators=1);

julia> fit!(ensemble, ([1.0, 2.0], 1));

julia> nobs(ensemble)
1
```
"""
mutable struct DriftAwareBagging{L,D} <: Learner{AbstractVector, Any}
    estimators::Vector{DriftAwareEstimator{L,D}}
    base_learner::Function
    detector::Function
    n_estimators::Int
    lambda::Float64
    n::Int
    rng::Random.AbstractRNG

    function DriftAwareBagging(
        base_learner,
        detector;
        n_estimators::Int = 10,
        lambda::Float64 = 6.0,
        rng::Random.AbstractRNG = Random.default_rng()
    )
        n_estimators > 0 || throw(ArgumentError("n_estimators must be positive"))
        isfinite(lambda) && lambda > 0 ||
            throw(ArgumentError("lambda must be finite and positive"))
        estimators = [DriftAwareEstimator(base_learner, detector) for _ in 1:n_estimators]
        L = typeof(estimators[1].model)
        D = typeof(estimators[1].drift_detector)
        new{L,D}(estimators, base_learner, detector, n_estimators, lambda, 0, rng)
    end
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(arf::DriftAwareBagging, xy::Tuple)
    x, y = xy

    for est in arf.estimators
        # Poisson resampling
        k = rand(arf.rng, Poisson(arf.lambda))

        if k > 0
            # Measure the error before exposing the current observation to the
            # model. Drift detection must follow test-then-train semantics.
            y_pred = predict(est.model, x)
            error = y_pred != y ? 1.0 : 0.0

            # Train main model
            for _ in 1:k
                fit!(est.model, (x, y))
            end

            # Update drift detector
            fit!(est.drift_detector, error)

            # Update warning detector
            fit!(est.warning_detector, error)

            # Handle warning - start training background model
            if detected_warning(est.warning_detector)
                if !est.in_warning
                    est.in_warning = true
                    est.background_model = arf.base_learner()
                end
            end

            # Train background model if in warning state
            if est.background_model !== nothing
                for _ in 1:k
                    fit!(est.background_model, (x, y))
                end
            end

            # Handle drift - replace with background model
            if detected_drift(est.drift_detector)
                est.n_drifts += 1

                if est.background_model !== nothing
                    est.model = est.background_model
                else
                    est.model = arf.base_learner()
                end

                # Reset detectors and warning state
                reset!(est.drift_detector)
                reset!(est.warning_detector)
                est.background_model = nothing
                est.in_warning = false
            end
        end
    end

    arf.n += 1
    return arf
end

OnlineStatsBase.value(arf::DriftAwareBagging) = arf.estimators
OnlineStatsBase.nobs(arf::DriftAwareBagging) = arf.n

function reset!(arf::DriftAwareBagging)
    for est in arf.estimators
        reset!(est.model)
        reset!(est.drift_detector)
        reset!(est.warning_detector)
        est.background_model = nothing
        est.n_drifts = 0
        est.in_warning = false
    end
    arf.n = 0
    return arf
end

# Prediction - majority voting
function predict(arf::DriftAwareBagging, x)
    if arf.n == 0
        return 0
    end

    votes = Dict{Any, Int}()
    for est in arf.estimators
        pred = predict(est.model, x)
        votes[pred] = get(votes, pred, 0) + 1
    end

    return dict_argmax(votes)
end

# Averaged probability predictions
function predict_proba(arf::DriftAwareBagging, x)
    if arf.n == 0
        return Dict{Any, Float64}()
    end

    probs = Dict{Any, Float64}()
    contributors = 0

    for est in arf.estimators
        est_probs = predict_proba(est.model, x)
        isempty(est_probs) && continue
        contributors += 1
        for (class, prob) in est_probs
            probs[class] = get(probs, class, 0.0) + prob
        end
    end

    contributors == 0 && return probs

    # Average over estimators that can issue a probability distribution.
    for class in keys(probs)
        probs[class] /= contributors
    end

    return probs
end

# Inspection methods
total_drifts(arf::DriftAwareBagging) = sum(est.n_drifts for est in arf.estimators)
per_learner_drifts(arf::DriftAwareBagging) = [est.n_drifts for est in arf.estimators]

const ARFEstimator = DriftAwareEstimator
const AdaptiveRandomForest = DriftAwareBagging
const per_tree_drifts = per_learner_drifts
