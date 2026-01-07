"""
    OnlineMLMLJExt

MLJ.jl integration extension for OnlineML.

This extension provides MLJ-compatible wrappers for OnlineML learners,
enabling them to be used within the MLJ ecosystem.

# Usage
```julia
using MLJ
using OnlineML

# Create MLJ-compatible model
model = OnlineLogisticRegression()

# Use with MLJ's fit!/predict
mach = machine(model, X, y)
fit!(mach)
predict(mach, X_new)
```

# Supported Models
- `OnlineLogisticRegression` - Wraps OnlineML.Linear.LogisticRegression
- `OnlinePerceptron` - Wraps OnlineML.Linear.Perceptron
- `OnlineNaiveBayes` - Wraps OnlineML.Bayes.GaussianNB
- `OnlineHoeffdingTree` - Wraps OnlineML.Trees.HoeffdingTree
- `OnlineKNN` - Wraps OnlineML.Instance.KNNClassifier
"""
module OnlineMLMLJExt

using MLJBase
import MLJBase: fit, predict, fitted_params, package_name, package_uuid,
                is_wrapper, supports_weights, load_path, package_url,
                package_license
using OnlineML
using OnlineML.Linear: LogisticRegression, Perceptron
using OnlineML.Bayes: GaussianNB
using OnlineML.Trees: HoeffdingTree
using OnlineML.Instance: KNNClassifier

# =============================================================================
# Abstract Types
# =============================================================================

abstract type OnlineMLModel <: MLJBase.Probabilistic end
abstract type OnlineDeterministic <: MLJBase.Deterministic end

# =============================================================================
# Model Wrappers
# =============================================================================

"""
    OnlineLogisticRegression

MLJ-compatible wrapper for OnlineML LogisticRegression.
"""
Base.@kwdef mutable struct OnlineLogisticRegression <: OnlineMLModel
    learning_rate::Float64 = 0.01
    regularization::Float64 = 0.0
end

"""
    OnlinePerceptronClassifier

MLJ-compatible wrapper for OnlineML Perceptron.
"""
Base.@kwdef mutable struct OnlinePerceptronClassifier <: OnlineDeterministic
    learning_rate::Float64 = 1.0
end

"""
    OnlineNaiveBayes

MLJ-compatible wrapper for OnlineML GaussianNB.
"""
Base.@kwdef mutable struct OnlineNaiveBayes <: OnlineMLModel
    # No hyperparameters for basic Gaussian NB
end

"""
    OnlineHoeffdingTreeClassifier

MLJ-compatible wrapper for OnlineML HoeffdingTree.
"""
Base.@kwdef mutable struct OnlineHoeffdingTreeClassifier <: OnlineDeterministic
    grace_period::Int = 200
    split_criterion::Symbol = :info_gain
    delta::Float64 = 1e-7
    max_depth::Int = 20
end

"""
    OnlineKNNClassifier

MLJ-compatible wrapper for OnlineML KNNClassifier.
"""
Base.@kwdef mutable struct OnlineKNNClassifier <: OnlineDeterministic
    k::Int = 5
    window_size::Int = 1000
end

# =============================================================================
# Fitted State
# =============================================================================

mutable struct OnlineMLFitResult{M}
    model::M
    classes::Vector
end

# =============================================================================
# MLJBase Interface - Package Metadata
# =============================================================================

for T in (OnlineLogisticRegression, OnlinePerceptronClassifier, OnlineNaiveBayes,
          OnlineHoeffdingTreeClassifier, OnlineKNNClassifier)
    @eval begin
        MLJBase.package_name(::Type{<:$T}) = "OnlineML"
        MLJBase.package_uuid(::Type{<:$T}) = "e8b0a8b0-0001-4000-8000-000000000001"
        MLJBase.package_url(::Type{<:$T}) = "https://github.com/OnlineML/OnlineML.jl"
        MLJBase.package_license(::Type{<:$T}) = "MIT"
        MLJBase.is_wrapper(::Type{<:$T}) = false
        MLJBase.supports_weights(::Type{<:$T}) = false
    end
end

MLJBase.load_path(::Type{<:OnlineLogisticRegression}) = "OnlineML.Linear.LogisticRegression"
MLJBase.load_path(::Type{<:OnlinePerceptronClassifier}) = "OnlineML.Linear.Perceptron"
MLJBase.load_path(::Type{<:OnlineNaiveBayes}) = "OnlineML.Bayes.GaussianNB"
MLJBase.load_path(::Type{<:OnlineHoeffdingTreeClassifier}) = "OnlineML.Trees.HoeffdingTree"
MLJBase.load_path(::Type{<:OnlineKNNClassifier}) = "OnlineML.Instance.KNNClassifier"

# =============================================================================
# MLJBase Interface - fit
# =============================================================================

function MLJBase.fit(model::OnlineLogisticRegression, verbosity::Int, X, y)
    learner = LogisticRegression(lr=model.learning_rate, lambda=model.regularization)
    classes = sort(unique(y))

    # Online training - fit each sample
    Xmat = MLJBase.matrix(X)
    for i in 1:size(Xmat, 1)
        OnlineML.fit!(learner, (Xmat[i, :], y[i]))
    end

    return OnlineMLFitResult(learner, classes), nothing, nothing
end

function MLJBase.fit(model::OnlinePerceptronClassifier, verbosity::Int, X, y)
    learner = Perceptron(lr=model.learning_rate)
    classes = sort(unique(y))

    Xmat = MLJBase.matrix(X)
    for i in 1:size(Xmat, 1)
        OnlineML.fit!(learner, (Xmat[i, :], y[i]))
    end

    return OnlineMLFitResult(learner, classes), nothing, nothing
end

function MLJBase.fit(model::OnlineNaiveBayes, verbosity::Int, X, y)
    learner = GaussianNB()
    classes = sort(unique(y))

    Xmat = MLJBase.matrix(X)
    for i in 1:size(Xmat, 1)
        OnlineML.fit!(learner, (Xmat[i, :], y[i]))
    end

    return OnlineMLFitResult(learner, classes), nothing, nothing
end

function MLJBase.fit(model::OnlineHoeffdingTreeClassifier, verbosity::Int, X, y)
    learner = HoeffdingTree(
        grace_period=model.grace_period,
        split_criterion=model.split_criterion,
        delta=model.delta,
        max_depth=model.max_depth
    )
    classes = sort(unique(y))

    Xmat = MLJBase.matrix(X)
    for i in 1:size(Xmat, 1)
        OnlineML.fit!(learner, (Xmat[i, :], y[i]))
    end

    return OnlineMLFitResult(learner, classes), nothing, nothing
end

function MLJBase.fit(model::OnlineKNNClassifier, verbosity::Int, X, y)
    learner = KNNClassifier(k=model.k, window_size=model.window_size)
    classes = sort(unique(y))

    Xmat = MLJBase.matrix(X)
    for i in 1:size(Xmat, 1)
        OnlineML.fit!(learner, (Xmat[i, :], y[i]))
    end

    return OnlineMLFitResult(learner, classes), nothing, nothing
end

# =============================================================================
# MLJBase Interface - predict
# =============================================================================

function MLJBase.predict(model::OnlineLogisticRegression, fitresult::OnlineMLFitResult, Xnew)
    learner = fitresult.model
    classes = fitresult.classes
    Xmat = MLJBase.matrix(Xnew)

    preds = Vector{MLJBase.UnivariateFinite}(undef, size(Xmat, 1))
    for i in 1:size(Xmat, 1)
        probs = OnlineML.predict_proba(learner, Xmat[i, :])
        # Convert to probability vector matching classes order
        prob_vec = [get(probs, c, 0.0) for c in classes]
        preds[i] = MLJBase.UnivariateFinite(classes, prob_vec)
    end

    return preds
end

function MLJBase.predict(model::OnlineNaiveBayes, fitresult::OnlineMLFitResult, Xnew)
    learner = fitresult.model
    classes = fitresult.classes
    Xmat = MLJBase.matrix(Xnew)

    preds = Vector{MLJBase.UnivariateFinite}(undef, size(Xmat, 1))
    for i in 1:size(Xmat, 1)
        probs = OnlineML.predict_proba(learner, Xmat[i, :])
        prob_vec = [get(probs, c, 0.0) for c in classes]
        preds[i] = MLJBase.UnivariateFinite(classes, prob_vec)
    end

    return preds
end

# Deterministic predictions
function MLJBase.predict(model::OnlinePerceptronClassifier, fitresult::OnlineMLFitResult, Xnew)
    learner = fitresult.model
    Xmat = MLJBase.matrix(Xnew)

    preds = Vector{eltype(fitresult.classes)}(undef, size(Xmat, 1))
    for i in 1:size(Xmat, 1)
        preds[i] = OnlineML.predict(learner, Xmat[i, :])
    end

    return preds
end

function MLJBase.predict(model::OnlineHoeffdingTreeClassifier, fitresult::OnlineMLFitResult, Xnew)
    learner = fitresult.model
    Xmat = MLJBase.matrix(Xnew)

    preds = Vector{Any}(undef, size(Xmat, 1))
    for i in 1:size(Xmat, 1)
        preds[i] = OnlineML.predict(learner, Xmat[i, :])
    end

    return preds
end

function MLJBase.predict(model::OnlineKNNClassifier, fitresult::OnlineMLFitResult, Xnew)
    learner = fitresult.model
    Xmat = MLJBase.matrix(Xnew)

    preds = Vector{Any}(undef, size(Xmat, 1))
    for i in 1:size(Xmat, 1)
        preds[i] = OnlineML.predict(learner, Xmat[i, :])
    end

    return preds
end

# =============================================================================
# MLJBase Interface - fitted_params
# =============================================================================

function MLJBase.fitted_params(model::OnlineLogisticRegression, fitresult::OnlineMLFitResult)
    learner = fitresult.model
    return (weights=learner.weights, classes=fitresult.classes, n_obs=OnlineML.nobs(learner))
end

function MLJBase.fitted_params(model::OnlinePerceptronClassifier, fitresult::OnlineMLFitResult)
    learner = fitresult.model
    return (weights=learner.weights, classes=fitresult.classes, n_obs=OnlineML.nobs(learner))
end

function MLJBase.fitted_params(model::OnlineNaiveBayes, fitresult::OnlineMLFitResult)
    return (classes=fitresult.classes, n_obs=OnlineML.nobs(fitresult.model))
end

function MLJBase.fitted_params(model::OnlineHoeffdingTreeClassifier, fitresult::OnlineMLFitResult)
    return (classes=fitresult.classes, n_obs=OnlineML.nobs(fitresult.model))
end

function MLJBase.fitted_params(model::OnlineKNNClassifier, fitresult::OnlineMLFitResult)
    return (k=model.k, classes=fitresult.classes, n_obs=OnlineML.nobs(fitresult.model))
end

# =============================================================================
# Exports
# =============================================================================

export OnlineLogisticRegression, OnlinePerceptronClassifier, OnlineNaiveBayes
export OnlineHoeffdingTreeClassifier, OnlineKNNClassifier

end # module OnlineMLMLJExt
