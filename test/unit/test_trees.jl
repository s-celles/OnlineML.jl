using Test
using OnlineML
using Random: MersenneTwister
import OnlineML: Trees
using .Trees: HoeffdingTree, ExtremelyFastTree, HoeffdingAdaptiveTree, n_nodes, height, n_leaves

@testset "Decision Trees" begin
    @testset "HoeffdingTree" begin
        rng = MersenneTwister(401)
        tree = HoeffdingTree()
        @test nobs(tree) == 0

        # Fit with some data
        for i in 1:100
            x = randn(rng, 4)
            y = rand(rng, [0, 1])
            fit!(tree, (x, y))
        end
        @test nobs(tree) == 100

        # Predict
        y_pred = predict(tree, randn(rng, 4))
        @test y_pred in [0, 1]

        # Predict proba
        probs = predict_proba(tree, randn(rng, 4))
        @test isa(probs, Dict)
        if !isempty(probs)
            @test all(0 <= v <= 1 for v in values(probs))
        end

        # Tree statistics
        @test n_nodes(tree) >= 1
        @test n_leaves(tree) >= 1
        @test height(tree) >= 0

        # Reset
        reset!(tree)
        @test nobs(tree) == 0
    end

    @testset "HoeffdingTree with parameters" begin
        rng = MersenneTwister(402)
        # Test with custom parameters
        tree = HoeffdingTree(
            grace_period=50,
            split_criterion=:gini,
            delta=1e-5,
            max_depth=10
        )

        for i in 1:200
            x = randn(rng, 3)
            y = x[1] > 0 ? 1 : 0  # Simple split on first feature
            fit!(tree, (x, y))
        end

        # Verify predictions are valid class labels
        y1 = predict(tree, [1.0, 0.0, 0.0])
        y2 = predict(tree, [-1.0, 0.0, 0.0])
        @test y1 in [0, 1]
        @test y2 in [0, 1]
    end
end
