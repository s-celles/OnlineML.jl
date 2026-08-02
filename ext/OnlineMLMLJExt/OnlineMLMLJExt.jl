"""
    OnlineMLMLJExt

MLJ.jl integration extension for OnlineML.

This extension provides MLJ-compatible model interfaces for OnlineML learners,
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
- `OnlineKNNClassifier` - Interfaces OnlineML.Instance.KNN
"""
module OnlineMLMLJExt

import MLJBase
import OnlineML
using OnlineML.Linear: LogisticRegression, Perceptron
using OnlineML.Bayes: GaussianNB
using OnlineML.Trees: HoeffdingTree
using OnlineML.Instance: KNN

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

struct OnlineMLFitResult{M,C<:AbstractVector}
    model::M
    classes::C
end

function _target_metadata(y; binary::Bool=false)
    isempty(y) && throw(ArgumentError("the target must contain at least one observation"))
    classes = collect(MLJBase.classes(first(y)))
    binary && length(classes) != 2 &&
        throw(ArgumentError("this OnlineML learner supports exactly two classes"))
    encoded = Int.(MLJBase.int(y))
    binary && (encoded .-= 1)
    return classes, encoded
end

function _fit_batch!(learner, X, y)
    Xmat = MLJBase.matrix(X)
    size(Xmat, 1) == length(y) ||
        throw(DimensionMismatch("X and y have different numbers of observations"))
    for i in axes(Xmat, 1)
        OnlineML.fit!(learner, (view(Xmat, i, :), y[i]))
    end
    return learner
end

_uses_binary_encoding(::Union{OnlineLogisticRegression,OnlinePerceptronClassifier}) = true
_uses_binary_encoding(::OnlineMLModel) = false
_uses_binary_encoding(::OnlineDeterministic) = false

function _encode_delta(model, fitresult::OnlineMLFitResult, ynew)
    offset = _uses_binary_encoding(model) ? 1 : 0
    encoded = Vector{Int}(undef, length(ynew))
    for i in eachindex(ynew)
        class_index = findfirst(==(ynew[i]), fitresult.classes)
        class_index === nothing && throw(ArgumentError(
            "new class $(repr(ynew[i])) is not in the fitted target pool",
        ))
        encoded[i] = class_index - offset
    end
    return encoded
end

# =============================================================================
# MLJBase Interface - Package Metadata
# =============================================================================

for T in (OnlineLogisticRegression, OnlinePerceptronClassifier, OnlineNaiveBayes,
          OnlineHoeffdingTreeClassifier, OnlineKNNClassifier)
    @eval begin
        MLJBase.package_name(::Type{<:$T}) = "OnlineML"
        MLJBase.package_uuid(::Type{<:$T}) = "e06e4760-77c8-4553-90da-06185c08ed2e"
        MLJBase.package_url(::Type{<:$T}) = "https://github.com/s-celles/OnlineML.jl"
        MLJBase.package_license(::Type{<:$T}) = "MIT"
        MLJBase.is_wrapper(::Type{<:$T}) = false
        MLJBase.supports_weights(::Type{<:$T}) = false
    end
end

MLJBase.load_path(::Type{<:OnlineLogisticRegression}) = "OnlineMLMLJExt.OnlineLogisticRegression"
MLJBase.load_path(::Type{<:OnlinePerceptronClassifier}) = "OnlineMLMLJExt.OnlinePerceptronClassifier"
MLJBase.load_path(::Type{<:OnlineNaiveBayes}) = "OnlineMLMLJExt.OnlineNaiveBayes"
MLJBase.load_path(::Type{<:OnlineHoeffdingTreeClassifier}) = "OnlineMLMLJExt.OnlineHoeffdingTreeClassifier"
MLJBase.load_path(::Type{<:OnlineKNNClassifier}) = "OnlineMLMLJExt.OnlineKNNClassifier"

for T in (OnlineLogisticRegression, OnlinePerceptronClassifier)
    @eval begin
        MLJBase.input_scitype(::Type{<:$T}) = MLJBase.Table(MLJBase.Continuous)
        MLJBase.target_scitype(::Type{<:$T}) = AbstractVector{<:MLJBase.Binary}
    end
end

for T in (OnlineNaiveBayes, OnlineHoeffdingTreeClassifier, OnlineKNNClassifier)
    @eval begin
        MLJBase.input_scitype(::Type{<:$T}) = MLJBase.Table(MLJBase.Continuous)
        MLJBase.target_scitype(::Type{<:$T}) = AbstractVector{<:MLJBase.Finite}
    end
end

# =============================================================================
# MLJBase Interface - fit
# =============================================================================

function MLJBase.fit(model::OnlineLogisticRegression, verbosity::Int, X, y)
    learner = LogisticRegression(
        optimizer=OnlineML.Optim.Adam(model.learning_rate),
        l2=model.regularization,
    )
    classes, encoded = _target_metadata(y; binary=true)
    _fit_batch!(learner, X, encoded)
    return OnlineMLFitResult(learner, classes), nothing, nothing
end

function MLJBase.fit(model::OnlinePerceptronClassifier, verbosity::Int, X, y)
    learner = Perceptron(η=model.learning_rate)
    classes, encoded = _target_metadata(y; binary=true)
    _fit_batch!(learner, X, encoded)
    return OnlineMLFitResult(learner, classes), nothing, nothing
end

function MLJBase.fit(model::OnlineNaiveBayes, verbosity::Int, X, y)
    learner = GaussianNB()
    classes, encoded = _target_metadata(y)
    _fit_batch!(learner, X, encoded)
    return OnlineMLFitResult(learner, classes), nothing, nothing
end

function MLJBase.fit(model::OnlineHoeffdingTreeClassifier, verbosity::Int, X, y)
    learner = HoeffdingTree(
        grace_period=model.grace_period,
        split_criterion=model.split_criterion,
        delta=model.delta,
        max_depth=model.max_depth
    )
    classes, encoded = _target_metadata(y)
    _fit_batch!(learner, X, encoded)
    return OnlineMLFitResult(learner, classes), nothing, nothing
end

function MLJBase.fit(model::OnlineKNNClassifier, verbosity::Int, X, y)
    learner = KNN(k=model.k, window_size=model.window_size)
    classes, encoded = _target_metadata(y)
    _fit_batch!(learner, X, encoded)
    return OnlineMLFitResult(learner, classes), nothing, nothing
end

"""
    update_observations!(model, fitresult, Xnew, ynew)

Consume only the new observations in `Xnew` and `ynew`, preserving the fitted
OnlineML learner state. All target classes are validated before training starts.

This operation is deliberately distinct from `MLJBase.update`, whose data
arguments represent the machine's complete training data rather than a delta.
"""
function update_observations!(
    model::Union{OnlineMLModel,OnlineDeterministic},
    fitresult::OnlineMLFitResult,
    Xnew,
    ynew,
)
    encoded = _encode_delta(model, fitresult, ynew)
    _fit_batch!(fitresult.model, Xnew, encoded)
    return fitresult
end

# =============================================================================
# MLJBase Interface - predict
# =============================================================================

function MLJBase.predict(model::OnlineLogisticRegression, fitresult::OnlineMLFitResult, Xnew)
    learner = fitresult.model
    classes = fitresult.classes
    Xmat = MLJBase.matrix(Xnew)

    probs = Matrix{Float64}(undef, size(Xmat, 1), length(classes))
    for i in 1:size(Xmat, 1)
        prediction = OnlineML.predict_proba(learner, view(Xmat, i, :))
        for j in eachindex(classes)
            probs[i, j] = get(prediction, j - 1, 0.0)
        end
    end
    return MLJBase.UnivariateFinite(classes, probs)
end

function MLJBase.predict(model::OnlineNaiveBayes, fitresult::OnlineMLFitResult, Xnew)
    learner = fitresult.model
    classes = fitresult.classes
    Xmat = MLJBase.matrix(Xnew)

    probs = Matrix{Float64}(undef, size(Xmat, 1), length(classes))
    for i in 1:size(Xmat, 1)
        prediction = OnlineML.predict_proba(learner, view(Xmat, i, :))
        for j in eachindex(classes)
            probs[i, j] = get(prediction, j, 0.0)
        end
    end
    return MLJBase.UnivariateFinite(classes, probs)
end

# Deterministic predictions
function MLJBase.predict(model::OnlinePerceptronClassifier, fitresult::OnlineMLFitResult, Xnew)
    learner = fitresult.model
    Xmat = MLJBase.matrix(Xnew)

    preds = Vector{eltype(fitresult.classes)}(undef, size(Xmat, 1))
    for i in 1:size(Xmat, 1)
        preds[i] = fitresult.classes[OnlineML.predict(learner, view(Xmat, i, :)) + 1]
    end

    return preds
end

function MLJBase.predict(model::OnlineHoeffdingTreeClassifier, fitresult::OnlineMLFitResult, Xnew)
    learner = fitresult.model
    Xmat = MLJBase.matrix(Xnew)

    preds = Vector{eltype(fitresult.classes)}(undef, size(Xmat, 1))
    for i in 1:size(Xmat, 1)
        preds[i] = fitresult.classes[OnlineML.predict(learner, view(Xmat, i, :))]
    end

    return preds
end

function MLJBase.predict(model::OnlineKNNClassifier, fitresult::OnlineMLFitResult, Xnew)
    learner = fitresult.model
    Xmat = MLJBase.matrix(Xnew)

    preds = Vector{eltype(fitresult.classes)}(undef, size(Xmat, 1))
    for i in 1:size(Xmat, 1)
        preds[i] = fitresult.classes[OnlineML.predict(learner, view(Xmat, i, :))]
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
export update_observations!

end # module OnlineMLMLJExt
