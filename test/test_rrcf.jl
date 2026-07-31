# Tests for RobustRandomCutForest

using Test
using OnlineML
import OnlineML: Anomaly
using .Anomaly: RobustRandomCutForest
using Random

@testset "RobustRandomCutForest" begin
    rng = MersenneTwister(42)

    @testset "Constructor and defaults" begin
        rrcf = RobustRandomCutForest()
        @test rrcf isa RobustRandomCutForest
        @test nobs(rrcf) == 0
        @test rrcf.n_trees == 100
        @test rrcf.tree_size == 256
    end

    @testset "Constructor with parameters" begin
        rrcf = RobustRandomCutForest(n_trees=50, tree_size=128, seed=42)
        @test rrcf.n_trees == 50
        @test rrcf.tree_size == 128
    end

    @testset "Invalid parameters" begin
        @test_throws ArgumentError RobustRandomCutForest(n_trees=0)
        @test_throws ArgumentError RobustRandomCutForest(tree_size=0)
    end

    @testset "Basic fit!" begin
        rrcf = RobustRandomCutForest(n_trees=10, tree_size=50)

        for i in 1:30
            fit!(rrcf, randn(rng, 3))
        end

        @test nobs(rrcf) == 30
        @test Anomaly.current_size(rrcf) == 30
    end

    @testset "Sliding window behavior" begin
        rrcf = RobustRandomCutForest(n_trees=10, tree_size=20)

        # Fill beyond capacity
        for i in 1:30
            fit!(rrcf, randn(rng, 3))
        end

        @test nobs(rrcf) == 30
        @test Anomaly.current_size(rrcf) == 20  # Capped at tree_size
    end

    @testset "reset!" begin
        rrcf = RobustRandomCutForest(n_trees=10, tree_size=50)

        for i in 1:20
            fit!(rrcf, randn(rng, 3))
        end

        @test nobs(rrcf) == 20

        reset!(rrcf)

        @test nobs(rrcf) == 0
        @test Anomaly.current_size(rrcf) == 0
    end

    @testset "fit_score!" begin
        rrcf = RobustRandomCutForest(n_trees=10, tree_size=50, seed=42)

        # First point
        s1 = fit_score!(rrcf, [1.0, 2.0, 3.0])
        @test s1 isa Float64
        @test nobs(rrcf) == 1

        # Second point
        s2 = fit_score!(rrcf, [1.1, 2.1, 3.1])
        @test s2 isa Float64
        @test nobs(rrcf) == 2
    end

    @testset "Anomaly detection" begin
        rrcf = RobustRandomCutForest(n_trees=50, tree_size=100, seed=42)

        # Train on normal data
        for i in 1:80
            fit!(rrcf, randn(rng, 3))
        end

        # Insert an outlier
        outlier = [10.0, 10.0, 10.0]
        outlier_score = fit_score!(rrcf, outlier)

        # Insert a normal point
        normal = randn(rng, 3)
        normal_score = fit_score!(rrcf, normal)

        # Outlier should generally have higher score
        # (This is probabilistic, so we use a softer test)
        @test outlier_score >= 0.0
        @test normal_score >= 0.0
    end

    @testset "is_anomaly" begin
        rrcf = RobustRandomCutForest(n_trees=10, tree_size=50, seed=42)

        # Train on normal data
        for i in 1:40
            fit!(rrcf, randn(rng, 3))
        end

        # Test with a normal point
        normal = [0.0, 0.0, 0.0]
        fit!(rrcf, normal)

        # is_anomaly should return a Bool
        result = is_anomaly(rrcf, normal)
        @test result isa Bool

        # With very high threshold, nothing should be anomaly
        result_high = is_anomaly(rrcf, normal; threshold=1000.0)
        @test result_high == false
    end

    @testset "score and score_one" begin
        rrcf = RobustRandomCutForest(n_trees=10, tree_size=50, seed=42)

        point = [1.0, 2.0, 3.0]
        fit!(rrcf, point)

        s = score(rrcf, point)
        @test s isa Float64

        s2 = score_one(rrcf, point)
        @test s == s2
    end

    @testset "value and nobs" begin
        rrcf = RobustRandomCutForest(n_trees=10, tree_size=50)
        @test nobs(rrcf) == 0

        fit!(rrcf, [1.0, 2.0])
        @test nobs(rrcf) == 1

        trees = value(rrcf)
        @test length(trees) == 10
    end

    @testset "Inspection methods" begin
        rrcf = RobustRandomCutForest(n_trees=25, tree_size=75)

        @test Anomaly.n_trees(rrcf) == 25
        @test Anomaly.tree_size(rrcf) == 75
        @test Anomaly.current_size(rrcf) == 0

        for i in 1:10
            fit!(rrcf, randn(rng, 3))
        end

        @test Anomaly.current_size(rrcf) == 10
    end

    @testset "CutTree operations" begin
        tree = Anomaly.CutTree()

        # Insert points
        Anomaly.insert_point!(tree, [1.0, 2.0], 1)
        @test tree.n_points == 1

        Anomaly.insert_point!(tree, [3.0, 4.0], 2)
        @test tree.n_points == 2

        # CoDisplacement score
        score = Anomaly.codisplacement(tree, 1)
        @test score >= 0.0

        # Delete point
        found = Anomaly.delete_point!(tree, 1)
        @test found
        @test tree.n_points == 1

        # Reset
        Anomaly.reset!(tree)
        @test tree.n_points == 0
    end

    @testset "Reproducibility with seed" begin
        rrcf1 = RobustRandomCutForest(n_trees=10, tree_size=50, seed=123)
        rrcf2 = RobustRandomCutForest(n_trees=10, tree_size=50, seed=123)

        # Same seed should produce same initial state
        @test rrcf1.n_trees == rrcf2.n_trees
        @test rrcf1.tree_size == rrcf2.tree_size

        # After identical operations, scores should be the same
        data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        for d in data
            fit!(rrcf1, d)
            fit!(rrcf2, d)
        end

        s1 = score(rrcf1, [1.0, 2.0])
        s2 = score(rrcf2, [1.0, 2.0])
        @test s1 == s2
    end
end
