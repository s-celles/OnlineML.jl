using Test
using OnlineML
using OnlineML.Trees

@testset "Decision Trees" begin
    @testset "HoeffdingTree" begin
        tree = HoeffdingTree()
        @test nobs(tree) == 0

        # Fit with some data
        for i in 1:100
            x = randn(4)
            y = rand([0, 1])
            fit!(tree, (x, y))
        end
        @test nobs(tree) == 100

        # Predict
        y_pred = predict(tree, randn(4))
        @test y_pred in [0, 1]

        # Predict proba
        probs = predict_proba(tree, randn(4))
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
        # Test with custom parameters
        tree = HoeffdingTree(
            grace_period=50,
            split_criterion=:gini,
            delta=1e-5,
            max_depth=10
        )

        for i in 1:200
            x = randn(3)
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
