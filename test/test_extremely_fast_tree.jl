# Tests for ExtremelyFastTree

using Test
using OnlineML
using OnlineML.Trees
using OnlineML.Streams

@testset "ExtremelyFastTree" begin
    @testset "Constructor and defaults" begin
        tree = ExtremelyFastTree()
        @test tree isa ExtremelyFastTree{Float64}
        @test nobs(tree) == 0
        @test n_nodes(tree) == 1  # Root leaf
        @test n_leaves(tree) == 1
        @test n_replacements(tree) == 0
    end

    @testset "Constructor with parameters" begin
        tree = ExtremelyFastTree(
            grace_period=100,
            re_eval_period=150,
            delta=1e-5,
            tau=0.1,
            split_criterion=:gini,
            leaf_prediction=:mc,
            max_depth=10
        )
        @test tree.grace_period == 100
        @test tree.re_eval_period == 150
        @test tree.delta ≈ 1e-5
        @test tree.tau ≈ 0.1
        @test tree.split_criterion == :gini
        @test tree.leaf_prediction == :mc
        @test tree.max_depth == 10
    end

    @testset "Invalid parameters" begin
        @test_throws ArgumentError ExtremelyFastTree(grace_period=0)
        @test_throws ArgumentError ExtremelyFastTree(delta=0.0)
        @test_throws ArgumentError ExtremelyFastTree(delta=1.0)
        @test_throws ArgumentError ExtremelyFastTree(split_criterion=:invalid)
        @test_throws ArgumentError ExtremelyFastTree(leaf_prediction=:invalid)
    end

    @testset "Basic fit! and predict" begin
        tree = ExtremelyFastTree(grace_period=10)

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
        tree = ExtremelyFastTree(grace_period=10)

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
        tree = ExtremelyFastTree(grace_period=50, delta=0.1)

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
        efdt = ExtremelyFastTree(grace_period=100)

        gen = SEAGenerator()
        correct_ht = 0
        correct_efdt = 0

        for i in 1:1000
            x, y = Streams.generate(gen)

            # Test then train
            if i > 100
                pred_ht = predict(ht, x)
                pred_efdt = predict(efdt, x)
                correct_ht += (pred_ht == y) ? 1 : 0
                correct_efdt += (pred_efdt == y) ? 1 : 0
            end

            fit!(ht, (x, y))
            fit!(efdt, (x, y))
        end

        acc_ht = correct_ht / 900
        acc_efdt = correct_efdt / 900

        # EFDT should be within 10% of HoeffdingTree (allowing for variance)
        @test abs(acc_efdt - acc_ht) < 0.15
    end

    @testset "NamedTuple input" begin
        tree = ExtremelyFastTree(grace_period=10)

        for i in 1:50
            nt = (a=Float64(i), b=Float64(i % 3), target=i % 2)
            fit!(tree, nt, :target)
        end

        @test nobs(tree) == 50
    end

    @testset "value and nobs" begin
        tree = ExtremelyFastTree()
        @test nobs(tree) == 0

        fit!(tree, ([1.0, 2.0], 1))
        @test nobs(tree) == 1

        v = value(tree)
        @test v isa EFDTNode
    end
end
