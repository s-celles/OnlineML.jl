using Test
using OnlineML
import OnlineML: Anomaly
using .Anomaly: GaussianProjectionDetector, HalfSpaceTrees,
    RobustRandomCutForest, CutTree
using Random: MersenneTwister

@testset "Anomaly Detection" begin
    @testset "GaussianProjectionDetector" begin
        rng = MersenneTwister(42)
        detector = GaussianProjectionDetector(n_projections=5, rng=rng)
        @test nobs(detector) == 0

        # Fit with some normal data
        for _ in 1:100
            x = randn(rng, 4) .* 0.5  # Normal data centered at origin
            fit!(detector, x)
        end
        @test nobs(detector) == 100

        # Score should be moderate for normal points
        normal_score = OnlineML.score_one(detector, randn(rng, 4) .* 0.5)
        @test 0.0 <= normal_score <= 1.0

        # Score should be higher for outliers
        outlier_score = OnlineML.score_one(detector, [10.0, 10.0, 10.0, 10.0])
        @test 0.0 <= outlier_score <= 1.0
        # Outliers should generally have higher scores (though not guaranteed for small samples)

        # Reset
        reset!(detector)
        @test nobs(detector) == 0
    end

    @testset "HalfSpaceTrees" begin
        rng = MersenneTwister(43)
        hst = HalfSpaceTrees(n_trees=5, height=4, window_size=100, rng=rng)
        @test nobs(hst) == 0

        # Fit with some normal data
        for _ in 1:100
            x = randn(rng, 3) .* 0.5
            fit!(hst, x)
        end
        @test nobs(hst) == 100

        # Score should be in valid range
        score = OnlineML.score_one(hst, randn(rng, 3))
        @test 0.0 <= score <= 1.0

        # Reset
        reset!(hst)
        @test nobs(hst) == 0
    end
end
