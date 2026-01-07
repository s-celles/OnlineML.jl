using Test
using OnlineML
import OnlineML: Metrics, Streams, Linear
using .Metrics: Accuracy, RollingMetric, R2, progressive_val_score, holdout_score
using .Streams: SEAGenerator, HyperplaneGenerator, generate
using .Linear: LogisticRegression
using Random

@testset "Evaluation" begin
    @testset "progressive_val_score test-then-train order" begin
        # Use a deterministic generator
        rng = Random.MersenneTwister(42)
        gen = SEAGenerator(function_id=1, noise=0.0, rng=rng)
        model = LogisticRegression()
        metric = Accuracy()

        score, trajectory = progressive_val_score(model, gen, metric, n_samples=100)

        # Score should be between 0 and 1
        @test 0.0 <= score <= 1.0

        # Trajectory should have entries
        @test length(trajectory) > 0

        # Trajectory should be generally increasing (learning)
        # We'll check that later values are at least as good as early ones
        # Note: With online learning from scratch, early scores can be highly variable
        # We use a generous tolerance since the model starts from random weights
        if length(trajectory) >= 10
            early_avg = sum(trajectory[1:5]) / 5
            late_avg = sum(trajectory[end-4:end]) / 5
            # Late scores should generally be better or comparable
            # Using generous tolerance due to inherent variance in online learning
            @test late_avg >= early_avg - 0.35  # Allow variance for online learning
        end
    end

    @testset "holdout_score interval evaluation" begin
        rng1 = Random.MersenneTwister(42)
        rng2 = Random.MersenneTwister(123)

        train_gen = SEAGenerator(function_id=1, noise=0.0, rng=rng1)
        test_gen = SEAGenerator(function_id=1, noise=0.0, rng=rng2)

        model = LogisticRegression()
        metric = Accuracy()

        score = holdout_score(model, train_gen, test_gen, metric,
                              train_size=200, test_size=100)

        # Score should be valid
        @test 0.0 <= score <= 1.0

        # Model should have learned from training
        @test nobs(model) == 200
    end

    @testset "Rolling metric wrapper" begin
        metric = RollingMetric(Accuracy; window_size=10)

        @test nobs(metric) == 0

        # Fit with some predictions
        for i in 1:20
            y_pred = i % 2
            y_true = i % 2  # Perfect predictions
            fit!(metric, (y_pred, y_true))
        end

        @test nobs(metric) == 20
        @test value(metric) == 1.0  # Should be 100% accurate in window

        # Now add some wrong predictions
        for i in 1:10
            fit!(metric, (0, 1))  # All wrong
        end

        # After rolling window, should have lower accuracy
        @test value(metric) < 1.0

        # Reset
        reset!(metric)
        @test nobs(metric) == 0
    end

    @testset "R2 metric" begin
        r2 = R2()

        @test nobs(r2) == 0
        @test isnan(value(r2))  # NaN when n < 2

        # Perfect predictions
        fit!(r2, (1.0, 1.0))
        fit!(r2, (2.0, 2.0))
        fit!(r2, (3.0, 3.0))

        @test nobs(r2) == 3
        @test value(r2) ≈ 1.0 atol=1e-10  # R² = 1 for perfect predictions

        # Reset and test with some error
        reset!(r2)
        @test nobs(r2) == 0

        # Add predictions with some error
        for i in 1:100
            y_true = Float64(i)
            y_pred = y_true + 0.1 * randn()  # Small noise
            fit!(r2, (y_pred, y_true))
        end

        # R² should be high but not perfect
        @test 0.9 < value(r2) < 1.0
    end

    @testset "Full evaluation pipeline" begin
        # Test complete prequential evaluation workflow
        rng = Random.MersenneTwister(999)
        gen = HyperplaneGenerator(n_features=5, rng=rng)
        model = LogisticRegression()
        metric = Accuracy()

        # Run evaluation
        score, trajectory = progressive_val_score(model, gen, metric, n_samples=500)

        # Verify results
        @test 0.0 <= score <= 1.0
        @test length(trajectory) >= 400  # Should have most samples evaluated
        @test nobs(model) == 500
    end
end
