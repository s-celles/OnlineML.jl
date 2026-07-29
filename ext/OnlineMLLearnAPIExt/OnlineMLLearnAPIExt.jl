module OnlineMLLearnAPIExt

using LearnAPI
using OnlineML
using OnlineML.Bayes: GaussianNB
using OnlineML.Linear: LogisticRegression, Regression

"""LearnAPI configuration for OnlineML's incremental Gaussian naive Bayes."""
struct GaussianNBLearner{T<:AbstractFloat} end

"""LearnAPI fitted model containing the mutable OnlineML state."""
struct OnlineFitted{L,M}
    learner::L
    state::M
end

"""LearnAPI configuration for OnlineML's incremental linear regression."""
struct RegressionLearner{T<:AbstractFloat} end

"""LearnAPI configuration for OnlineML's incremental logistic regression."""
struct LogisticRegressionLearner{T<:AbstractFloat,O}
    optimizer::O
    l1::T
    l2::T
end

function LogisticRegressionLearner{T}(;
    optimizer=LogisticRegression{T}().optimizer,
    l1=zero(T),
    l2=zero(T),
) where {T<:AbstractFloat}
    return LogisticRegressionLearner{T,typeof(optimizer)}(
        optimizer,
        T(l1),
        T(l2),
    )
end

function LogisticRegressionLearner{T}(
    optimizer,
    l1,
    l2,
) where {T<:AbstractFloat}
    return LogisticRegressionLearner{T}(; optimizer, l1, l2)
end

function OnlineML.learnapi(model::GaussianNB{T}) where {T}
    OnlineML.nobs(model) == 0 ||
        throw(ArgumentError("learnapi expects an unfitted GaussianNB configuration"))
    return GaussianNBLearner{T}()
end

function OnlineML.learnapi(model::Regression{T}) where {T}
    OnlineML.nobs(model) == 0 ||
        throw(ArgumentError("learnapi expects an unfitted Regression configuration"))
    return RegressionLearner{T}()
end

function OnlineML.learnapi(model::LogisticRegression{T}) where {T}
    OnlineML.nobs(model) == 0 ||
        throw(ArgumentError(
            "learnapi expects an unfitted LogisticRegression configuration",
        ))
    return LogisticRegressionLearner{T}(;
        optimizer=model.optimizer,
        l1=model.l1,
        l2=model.l2,
    )
end

LearnAPI.constructor(::GaussianNBLearner{T}) where {T} = GaussianNBLearner{T}
LearnAPI.constructor(::RegressionLearner{T}) where {T} = RegressionLearner{T}
LearnAPI.constructor(::LogisticRegressionLearner{T}) where {T} =
    LogisticRegressionLearner{T}
LearnAPI.learner(model::OnlineFitted) = model.learner

const SUPPORTED_LEARNER =
    Union{GaussianNBLearner,LogisticRegressionLearner,RegressionLearner}

LearnAPI.functions(::SUPPORTED_LEARNER) = (
    :(LearnAPI.fit),
    :(LearnAPI.learner),
    :(LearnAPI.clone),
    :(LearnAPI.strip),
    :(LearnAPI.obs),
    :(LearnAPI.predict),
    :(LearnAPI.update_observations),
)

LearnAPI.kinds_of_proxy(::SUPPORTED_LEARNER) = (LearnAPI.Point(),)
LearnAPI.tags(::GaussianNBLearner) = ("classification", "incremental algorithms")
LearnAPI.tags(::LogisticRegressionLearner) =
    ("classification", "incremental algorithms")
LearnAPI.tags(::RegressionLearner) = ("regression", "incremental algorithms")
LearnAPI.is_pure_julia(::GaussianNBLearner) = true
LearnAPI.is_pure_julia(::LogisticRegressionLearner) = true
LearnAPI.is_pure_julia(::RegressionLearner) = true
LearnAPI.human_name(::GaussianNBLearner) = "online Gaussian naive Bayes classifier"
LearnAPI.human_name(::LogisticRegressionLearner) =
    "online logistic regression classifier"
LearnAPI.human_name(::RegressionLearner) = "online linear regressor"

function LearnAPI.fit(
    learner::GaussianNBLearner{T},
    observations;
    verbosity=LearnAPI.default_verbosity(),
) where {T}
    state = GaussianNB{T}()
    OnlineML.fit_batch!(state, observations)
    return OnlineFitted(learner, state)
end

function LearnAPI.fit(
    learner::RegressionLearner{T},
    observations;
    verbosity=LearnAPI.default_verbosity(),
) where {T}
    state = Regression{T}()
    OnlineML.fit_batch!(state, observations)
    return OnlineFitted(learner, state)
end

function LearnAPI.fit(
    learner::LogisticRegressionLearner{T},
    observations;
    verbosity=LearnAPI.default_verbosity(),
) where {T}
    state = LogisticRegression{T}(;
        optimizer=deepcopy(learner.optimizer),
        l1=learner.l1,
        l2=learner.l2,
    )
    OnlineML.fit_batch!(state, observations)
    return OnlineFitted(learner, state)
end

function LearnAPI.update_observations(
    model::OnlineFitted,
    new_observations;
    verbosity=LearnAPI.default_verbosity(),
)
    OnlineML.fit_batch!(model.state, new_observations)
    return model
end

function LearnAPI.predict(model::OnlineFitted, ::LearnAPI.Point, features)
    return [OnlineML.predict(model.state, x) for x in features]
end

export GaussianNBLearner, LogisticRegressionLearner, RegressionLearner, OnlineFitted

end
