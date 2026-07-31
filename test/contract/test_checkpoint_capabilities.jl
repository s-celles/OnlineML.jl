using Random: MersenneTwister
using Serialization: deserialize, serialize
using OnlineML.Anomaly: GaussianProjectionDetector, RandomCutForestApproximation,
    fit_score!
using OnlineML.Ensemble: Bagging
using OnlineML.Linear: LogisticRegression

function serialization_roundtrip(model)
    buffer = IOBuffer()
    serialize(buffer, model)
    seekstart(buffer)
    return deserialize(buffer)
end

@testset "bagging resumes stochastic updates exactly" begin
    model = Bagging(
        () -> LogisticRegression();
        n_estimators=3,
        rng=MersenneTwister(21),
    )
    first_delta = [([-1.0], 0), ([1.0], 1), ([-2.0], 0)]
    second_delta = [([2.0], 1), ([-3.0], 0), ([3.0], 1)]
    fit_batch!(model, first_delta)

    restored = serialization_roundtrip(model)
    fit_batch!(model, second_delta)
    fit_batch!(restored, second_delta)

    @test nobs(restored) == nobs(model)
    @test OnlineML.predict(restored, [-1.0]) == OnlineML.predict(model, [-1.0])
    @test OnlineML.predict(restored, [1.0]) == OnlineML.predict(model, [1.0])
    @test map(nobs, restored.learners) == map(nobs, model.learners)
    @test map(learner -> copy(learner.weights), restored.learners) ==
          map(learner -> copy(learner.weights), model.learners)
end

@testset "anomaly detectors preserve RNG and retained state" begin
    loda = GaussianProjectionDetector(n_projections=4, rng=MersenneTwister(22))
    fit_batch!(loda, [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    restored_loda = serialization_roundtrip(loda)

    fit!(loda, [2.0, 2.0])
    fit!(restored_loda, [2.0, 2.0])
    @test nobs(restored_loda) == nobs(loda)
    @test score_one(restored_loda, [1.0, 1.0]) == score_one(loda, [1.0, 1.0])
    @test restored_loda.projections == loda.projections

    forest = RandomCutForestApproximation(n_trees=3, tree_size=4, seed=23)
    fit_batch!(forest, [[0.0], [1.0], [2.0]])
    restored_forest = serialization_roundtrip(forest)

    score = fit_score!(forest, [3.0])
    restored_score = fit_score!(restored_forest, [3.0])
    @test restored_score == score
    @test collect(restored_forest.point_buffer) == collect(forest.point_buffer)
    @test collect(restored_forest.point_indices) == collect(forest.point_indices)
end
