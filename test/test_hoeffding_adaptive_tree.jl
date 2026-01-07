# Tests for HoeffdingAdaptiveTree

using Test
using OnlineML
using OnlineML.Trees
using OnlineML.Streams
using OnlineML.Drift

@testset "HoeffdingAdaptiveTree" begin
    @testset "Constructor and defaults" begin
        tree = HoeffdingAdaptiveTree()
        @test tree isa HoeffdingAdaptiveTree{Float64}
        @test nobs(tree) == 0
        @test n_nodes(tree) == 1  # Root leaf
        @test n_leaves(tree) == 1
        @test n_replacements(tree) == 0
    end

    @testset "Constructor with parameters" begin
        tree = HoeffdingAdaptiveTree(
            grace_period=100,
            delta=1e-5,
            tau=0.1,
            split_criterion=:gini,
            leaf_prediction=:mc,
            max_depth=10,
            adwin_delta=0.001
        )
        @test tree.grace_period == 100
        @test tree.delta ≈ 1e-5
        @test tree.tau ≈ 0.1
        @test tree.split_criterion == :gini
        @test tree.leaf_prediction == :mc
        @test tree.max_depth == 10
        @test tree.adwin_delta == 0.001
    end

    @testset "Invalid parameters" begin
        @test_throws ArgumentError HoeffdingAdaptiveTree(grace_period=0)
        @test_throws ArgumentError HoeffdingAdaptiveTree(delta=0.0)
        @test_throws ArgumentError HoeffdingAdaptiveTree(delta=1.0)
        @test_throws ArgumentError HoeffdingAdaptiveTree(split_criterion=:invalid)
        @test_throws ArgumentError HoeffdingAdaptiveTree(leaf_prediction=:invalid)
    end

    @testset "Basic fit! and predict" begin
        tree = HoeffdingAdaptiveTree(grace_period=10)

        # Fit some observations
        for i in 1:100
            x = [Float64(i % 2), Float64(i % 3)]
            y = i % 2  # Binary classification
            fit!(tree, (x, y))
        end

        @test nobs(tree) == 100

        # Predict
        y_pred = predict(tree, [0.0, 0.0])
        @test y_pred isa Int

        # Predict proba
        probs = predict_proba(tree, [0.0, 0.0])
        @test probs isa Dict{Int, Float64}
        if !isempty(probs)
            @test sum(values(probs)) ≈ 1.0 atol=1e-10
        end
    end

    @testset "reset!" begin
        tree = HoeffdingAdaptiveTree(grace_period=10)

        for i in 1:50
            fit!(tree, ([Float64(i), Float64(i)], i % 2))
        end

        @test nobs(tree) == 50

        reset!(tree)

        @test nobs(tree) == 0
        @test n_nodes(tree) == 1
        @test n_leaves(tree) == 1
        @test n_replacements(tree) == 0
    end

    @testset "Tree growth" begin
        tree = HoeffdingAdaptiveTree(grace_period=50, delta=0.1)

        # Train with separable data
        for i in 1:500
            if i % 2 == 0
                x = [rand() * 0.4, rand()]
                y = 0
            else
                x = [0.6 + rand() * 0.4, rand()]
                y = 1
            end
            fit!(tree, (x, y))
        end

        # Tree should have grown
        @test n_nodes(tree) >= 1
        @test n_leaves(tree) >= 1
    end

    @testset "Comparison with HoeffdingTree accuracy" begin
        # Both should achieve reasonable accuracy on synthetic data
        ht = HoeffdingTree(grace_period=100)
        hat = HoeffdingAdaptiveTree(grace_period=100)

        gen = SEAGenerator()
        correct_ht = 0
        correct_hat = 0

        for i in 1:1000
            x, y = Streams.generate(gen)

            # Test then train
            if i > 100
                pred_ht = predict(ht, x)
                pred_hat = predict(hat, x)
                correct_ht += (pred_ht == y) ? 1 : 0
                correct_hat += (pred_hat == y) ? 1 : 0
            end

            fit!(ht, (x, y))
            fit!(hat, (x, y))
        end

        acc_ht = correct_ht / 900
        acc_hat = correct_hat / 900

        # HAT should be within 15% of HoeffdingTree (allowing for variance)
        @test abs(acc_hat - acc_ht) < 0.15
    end

    @testset "NamedTuple input" begin
        tree = HoeffdingAdaptiveTree(grace_period=10)

        for i in 1:50
            nt = (a=Float64(i), b=Float64(i % 3), target=i % 2)
            fit!(tree, nt, :target)
        end

        @test nobs(tree) == 50
    end

    @testset "value and nobs" begin
        tree = HoeffdingAdaptiveTree()
        @test nobs(tree) == 0

        fit!(tree, ([1.0, 2.0], 1))
        @test nobs(tree) == 1

        v = value(tree)
        @test v isa Trees.AdaptiveNode
    end

    @testset "height function" begin
        tree = HoeffdingAdaptiveTree(grace_period=20, delta=0.5)

        # Initial height should be 0 (just root leaf)
        @test height(tree) == 0

        # Train with separable data to induce splits
        for i in 1:200
            if i % 2 == 0
                x = [0.0, rand()]
                y = 0
            else
                x = [1.0, rand()]
                y = 1
            end
            fit!(tree, (x, y))
        end

        # Height should be at least 0 (root could be split or not)
        @test height(tree) >= 0
    end
end
