module OnlineMLLearnAPIExt

using LearnAPI
using OnlineML
using OnlineML.Bayes: GaussianNB

"""LearnAPI configuration for OnlineML's incremental Gaussian naive Bayes."""
struct GaussianNBLearner end

"""LearnAPI fitted model containing the mutable OnlineML state."""
struct GaussianNBFitted{M}
    learner::GaussianNBLearner
    state::M
end

function OnlineML.learnapi(model::GaussianNB)
    OnlineML.nobs(model) == 0 ||
        throw(ArgumentError("learnapi expects an unfitted GaussianNB configuration"))
    return GaussianNBLearner()
end

LearnAPI.constructor(::GaussianNBLearner) = GaussianNBLearner
LearnAPI.learner(model::GaussianNBFitted) = model.learner

LearnAPI.functions(::GaussianNBLearner) = (
    :(LearnAPI.fit),
    :(LearnAPI.learner),
    :(LearnAPI.clone),
    :(LearnAPI.strip),
    :(LearnAPI.obs),
    :(LearnAPI.predict),
    :(LearnAPI.update_observations),
)

LearnAPI.kinds_of_proxy(::GaussianNBLearner) = (LearnAPI.Point(),)
LearnAPI.tags(::GaussianNBLearner) = ("classification", "incremental algorithms")
LearnAPI.is_pure_julia(::GaussianNBLearner) = true
LearnAPI.human_name(::GaussianNBLearner) = "online Gaussian naive Bayes classifier"

function LearnAPI.fit(
    learner::GaussianNBLearner,
    observations;
    verbosity=LearnAPI.default_verbosity(),
)
    state = GaussianNB()
    OnlineML.fit_batch!(state, observations)
    return GaussianNBFitted(learner, state)
end

function LearnAPI.update_observations(
    model::GaussianNBFitted,
    new_observations;
    verbosity=LearnAPI.default_verbosity(),
)
    OnlineML.fit_batch!(model.state, new_observations)
    return model
end

function LearnAPI.predict(model::GaussianNBFitted, ::LearnAPI.Point, features)
    return [OnlineML.predict(model.state, x) for x in features]
end

export GaussianNBLearner, GaussianNBFitted

end
