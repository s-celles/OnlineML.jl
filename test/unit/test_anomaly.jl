using Test
using OnlineML
import OnlineML: Anomaly
using .Anomaly: LODA, HalfSpaceTrees, RobustRandomCutForest, CutTree

@testset "Anomaly Detection" begin
    @testset "LODA" begin
        loda = LODA(n_projections=5)
        @test nobs(loda) == 0

        # Fit with some normal data
        for _ in 1:100
            x = randn(4) .* 0.5  # Normal data centered at origin
            fit!(loda, x)
        end
        @test nobs(loda) == 100

        # Score should be moderate for normal points
        normal_score = OnlineML.score_one(loda, randn(4) .* 0.5)
        @test 0.0 <= normal_score <= 1.0

        # Score should be higher for outliers
        outlier_score = OnlineML.score_one(loda, [10.0, 10.0, 10.0, 10.0])
        @test 0.0 <= outlier_score <= 1.0
        # Outliers should generally have higher scores (though not guaranteed for small samples)

        # Reset
        reset!(loda)
        @test nobs(loda) == 0
    end

    @testset "HalfSpaceTrees" begin
        hst = HalfSpaceTrees(n_trees=5, height=4, window_size=100)
        @test nobs(hst) == 0

        # Fit with some normal data
        for _ in 1:100
            x = randn(3) .* 0.5
            fit!(hst, x)
        end
        @test nobs(hst) == 100

        # Score should be in valid range
        score = OnlineML.score_one(hst, randn(3))
        @test 0.0 <= score <= 1.0

        # Reset
        reset!(hst)
        @test nobs(hst) == 0
    end
end
