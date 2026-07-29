module OnlineMLLearnAPIExt

using LearnAPI
using OnlineML
using OnlineML.Bayes: GaussianNB
using OnlineML.Linear: Regression

"""LearnAPI configuration for OnlineML's incremental Gaussian naive Bayes."""
struct GaussianNBLearner{T<:AbstractFloat} end

"""LearnAPI fitted model containing the mutable OnlineML state."""
struct OnlineFitted{L,M}
    learner::L
    state::M
end

"""LearnAPI configuration for OnlineML's incremental linear regression."""
struct RegressionLearner{T<:AbstractFloat} end

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

LearnAPI.constructor(::GaussianNBLearner{T}) where {T} = GaussianNBLearner{T}
LearnAPI.constructor(::RegressionLearner{T}) where {T} = RegressionLearner{T}
LearnAPI.learner(model::OnlineFitted) = model.learner

const SUPPORTED_LEARNER = Union{GaussianNBLearner,RegressionLearner}

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
LearnAPI.tags(::RegressionLearner) = ("regression", "incremental algorithms")
LearnAPI.is_pure_julia(::GaussianNBLearner) = true
LearnAPI.is_pure_julia(::RegressionLearner) = true
LearnAPI.human_name(::GaussianNBLearner) = "online Gaussian naive Bayes classifier"
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

export GaussianNBLearner, RegressionLearner, OnlineFitted

end
