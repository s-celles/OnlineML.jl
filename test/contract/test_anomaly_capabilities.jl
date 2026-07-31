using Random
using OnlineML.Anomaly: GaussianProjectionDetector, HalfSpaceTrees, LODA,
    RandomCutForestApproximation, RobustRandomCutForest,
    current_size, fit_score!

@testset "constructor and schema constraints" begin
    @test_throws ArgumentError HalfSpaceTrees(n_trees=0)
    @test_throws ArgumentError HalfSpaceTrees(height=-1)
    @test_throws ArgumentError HalfSpaceTrees(window_size=0)
    @test_throws ArgumentError LODA(n_projections=0)
    @test LODA === GaussianProjectionDetector
    @test_throws ArgumentError RobustRandomCutForest(n_trees=0)
    @test_throws ArgumentError RobustRandomCutForest(tree_size=0)
    @test RobustRandomCutForest === RandomCutForestApproximation

    for model in (
        HalfSpaceTrees(n_trees=2, height=2, rng=MersenneTwister(1)),
        GaussianProjectionDetector(n_projections=2, rng=MersenneTwister(1)),
        RandomCutForestApproximation(n_trees=2, tree_size=3, seed=1),
    )
        fit!(model, [1.0, 2.0])
        @test_throws DimensionMismatch fit!(model, [1.0])
        @test_throws DimensionMismatch score_one(model, [1.0])
    end
end

@testset "single-pass delta batches and reset" begin
    models = (
        HalfSpaceTrees(n_trees=2, height=2, window_size=3, rng=MersenneTwister(2)),
        GaussianProjectionDetector(n_projections=2, rng=MersenneTwister(2)),
        RandomCutForestApproximation(n_trees=2, tree_size=3, seed=2),
    )

    for model in models
        xs = (x for x in ([0.0, 0.0], [0.1, 0.1], [2.0, 2.0]))
        fit_batch!(model, xs)
        @test nobs(model) == 3
        fit_batch!(model, (x for x in ([3.0, 3.0],)))
        @test nobs(model) == 4

        probe = model isa RandomCutForestApproximation ? [3.0, 3.0] : [1.0, 1.0]
        before = nobs(model)
        result = score_one(model, probe)
        @test isfinite(result)
        @test nobs(model) == before

        reset!(model)
        @test nobs(model) == 0
    end
end

@testset "bounded retained state" begin
    hst = HalfSpaceTrees(n_trees=3, height=2, window_size=2, rng=MersenneTwister(3))
    fit_batch!(hst, ([Float64(i), 0.0] for i in 1:10))
    @test length(hst.trees) == 3

    loda = GaussianProjectionDetector(n_projections=4, rng=MersenneTwister(3))
    fit_batch!(loda, ([Float64(i), 0.0] for i in 1:10))
    @test length(loda.projections) == 4
    @test length(loda.stats) == 4

    rrcf = RandomCutForestApproximation(n_trees=3, tree_size=4, seed=3)
    scores = [fit_score!(rrcf, [Float64(i), 0.0]) for i in 1:10]
    @test all(isfinite, scores)
    @test current_size(rrcf) == 4
    @test all(tree.n_points == 4 for tree in rrcf.trees)
    @test_throws ArgumentError score_one(rrcf, [-1.0, 0.0])
end

@testset "deterministic construction with explicit RNG state" begin
    hst1 = HalfSpaceTrees(n_trees=3, height=2, rng=MersenneTwister(4))
    hst2 = HalfSpaceTrees(n_trees=3, height=2, rng=MersenneTwister(4))
    loda1 = GaussianProjectionDetector(n_projections=3, rng=MersenneTwister(4))
    loda2 = GaussianProjectionDetector(n_projections=3, rng=MersenneTwister(4))

    data = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    fit_batch!(hst1, data)
    fit_batch!(hst2, data)
    fit_batch!(loda1, data)
    fit_batch!(loda2, data)

    @test score_one(hst1, [0.25, 0.25]) == score_one(hst2, [0.25, 0.25])
    @test score_one(loda1, [0.25, 0.25]) == score_one(loda2, [0.25, 0.25])
    @test 0.0 <= score_one(hst1, [0.25, 0.25]) <= 1.0
    @test 0.0 <= score_one(loda1, [0.25, 0.25]) <= 1.0
    @test score_one(HalfSpaceTrees(), [0.0]) == 0.5
end
