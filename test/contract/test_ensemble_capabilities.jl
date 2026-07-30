using OnlineML
using OnlineML.Drift: DDM
using OnlineML.Ensemble: AdaptiveRandomForest, Bagging, LeveragingBagging
using OnlineML.Linear: LogisticRegression
using OnlineStatsBase
using Random: MersenneTwister

mutable struct EnsembleOneShot{T}
    data::T
    started::Bool
end

function Base.iterate(stream::EnsembleOneShot)
    stream.started && error("stream was traversed more than once")
    stream.started = true
    return iterate(stream.data)
end

Base.iterate(stream::EnsembleOneShot, state) = iterate(stream.data, state)

mutable struct RecordingDetector <: Detector
    observations::Vector{Float64}
end

OnlineStatsBase._fit!(detector::RecordingDetector, x::Real) =
    (push!(detector.observations, Float64(x)); detector)
OnlineStatsBase.value(detector::RecordingDetector) = detector.observations
OnlineStatsBase.nobs(detector::RecordingDetector) = length(detector.observations)
OnlineML.status(::RecordingDetector) = NoDrift
OnlineML.detected_drift(::RecordingDetector) = false
OnlineML.detected_warning(::RecordingDetector) = false
OnlineML.reset!(detector::RecordingDetector) = (empty!(detector.observations); detector)

mutable struct LastLabelLearner <: Learner{AbstractVector{Float64}, Int}
    label::Int
    n::Int
end

OnlineStatsBase._fit!(learner::LastLabelLearner, xy::Tuple) =
    (learner.label = Int(last(xy)); learner.n += 1; learner)
OnlineStatsBase.value(learner::LastLabelLearner) = learner.label
OnlineStatsBase.nobs(learner::LastLabelLearner) = learner.n
OnlineML.predict(learner::LastLabelLearner, _) = learner.label
OnlineML.predict_proba(learner::LastLabelLearner, _) =
    learner.n == 0 ? Dict{Int,Float64}() : Dict(learner.label => 1.0)
OnlineML.reset!(learner::LastLabelLearner) =
    (learner.label = 0; learner.n = 0; learner)

learned_observations(ensemble::Union{Bagging,LeveragingBagging}) =
    sum(nobs, ensemble.learners)
learned_observations(ensemble::AdaptiveRandomForest) =
    sum(estimator -> nobs(estimator.model), ensemble.estimators)

@testset "constructor validation" begin
    base = () -> LogisticRegression()
    detector = () -> DDM()

    for factory in (
        () -> Bagging(base; n_estimators=0),
        () -> LeveragingBagging(base; n_estimators=0),
        () -> AdaptiveRandomForest(base, detector; n_estimators=0),
        () -> Bagging(base; lambda=0.0),
        () -> LeveragingBagging(base; lambda=-1.0),
        () -> AdaptiveRandomForest(base, detector; lambda=Inf),
    )
        @test_throws ArgumentError factory()
    end
end

@testset "stochastic ensembles consume a delta once" begin
    observations = [
        ([-1.0], 0),
        ([1.0], 1),
        ([-2.0], 0),
        ([2.0], 1),
    ]
    factories = (
        () -> Bagging(
            () -> LogisticRegression();
            n_estimators=3,
            rng=MersenneTwister(11),
        ),
        () -> LeveragingBagging(
            () -> LogisticRegression();
            n_estimators=3,
            rng=MersenneTwister(11),
        ),
        () -> AdaptiveRandomForest(
            () -> LogisticRegression(),
            () -> DDM(min_instances=10);
            n_estimators=3,
            rng=MersenneTwister(11),
        ),
    )

    for factory in factories
        ensemble = factory()
        @test fit_batch!(ensemble, EnsembleOneShot(observations, false)) === ensemble
        @test nobs(ensemble) == length(observations)

        state_before = learned_observations(ensemble)
        OnlineML.predict(ensemble, [0.5])
        @test learned_observations(ensemble) == state_before

        probabilities = OnlineML.predict_proba(ensemble, [0.5])
        @test sum(values(probabilities)) ≈ 1.0

        @test reset!(ensemble) === ensemble
        @test nobs(ensemble) == 0
    end
end

@testset "probabilities ignore untrained contributors" begin
    for factory in (
        () -> Bagging(() -> LastLabelLearner(0, 0); n_estimators=3),
        () -> LeveragingBagging(() -> LastLabelLearner(0, 0); n_estimators=3),
    )
        ensemble = factory()
        fit!(first(ensemble.learners), ([1.0], 1))
        ensemble.n = 1
        @test OnlineML.predict_proba(ensemble, [1.0]) == Dict(1 => 1.0)
    end
end

@testset "ARF drift error is test-then-train" begin
    forest = AdaptiveRandomForest(
        () -> LastLabelLearner(0, 0),
        () -> RecordingDetector(Float64[]);
        n_estimators=1,
        lambda=1.0,
        rng=MersenneTwister(1),
    )

    estimator = only(forest.estimators)

    # Advance to the first positive Poisson draw. Before that first training
    # update, the learner predicts 0, so label 1 must be recorded as an error.
    for _ in 1:20
        fit!(forest, ([1.0], 1))
        isempty(estimator.drift_detector.observations) || break
    end

    @test first(estimator.drift_detector.observations) == 1.0
    @test first(estimator.warning_detector.observations) == 1.0
    @test value(estimator.model) == 1
end
