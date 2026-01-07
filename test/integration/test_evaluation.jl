using Test
using OnlineML
import OnlineML: Metrics, Streams, Linear, Trees, Ensemble
using .Metrics: Accuracy, RollingMetric, R2, MSE, progressive_val_score, holdout_score
using .Streams: SEAGenerator, HyperplaneGenerator, RandomRBFGenerator
using .Streams: SineGenerator, ConceptDriftStream, generate
using .Linear: LogisticRegression
using .Trees: HoeffdingTree
using .Ensemble: Bagging
using Random

@testset "Integration: Evaluation" begin
    @testset "Full prequential evaluation with LogisticRegression" begin
        rng = Random.MersenneTwister(42)
        gen = SEAGenerator(function_id=1, noise=0.05, rng=rng)
        model = LogisticRegression()
        metric = Accuracy()

        score, trajectory = progressive_val_score(model, gen, metric, n_samples=500)

        @test 0.0 <= score <= 1.0
        @test length(trajectory) >= 400
        @test score > 0.5  # Should be better than random
    end

    @testset "Holdout evaluation with HoeffdingTree" begin
        rng1 = Random.MersenneTwister(1)
        rng2 = Random.MersenneTwister(2)

        train_gen = HyperplaneGenerator(n_features=5, rng=rng1)
        test_gen = HyperplaneGenerator(n_features=5, rng=rng2)

        model = HoeffdingTree()
        metric = Accuracy()

        score = holdout_score(model, train_gen, test_gen, metric,
                              train_size=500, test_size=200)

        @test 0.0 <= score <= 1.0
    end

    @testset "Rolling metric with ensemble" begin
        rng = Random.MersenneTwister(123)
        gen = RandomRBFGenerator(n_classes=2, n_features=5, rng=rng)

        model = Bagging(() -> LogisticRegression(), n_estimators=3)
        metric = RollingMetric(Accuracy; window_size=50)

        # Train and track rolling accuracy
        scores = Float64[]
        for i in 1:200
            x, y = generate(gen)

            if nobs(model) > 0
                y_pred = OnlineML.predict(model, x)
                fit!(metric, (y_pred, y))
                push!(scores, value(metric))
            end

            fit!(model, (x, y))
        end

        @test length(scores) >= 150
        @test all(0.0 .<= scores .<= 1.0)
    end

    @testset "Evaluation with concept drift" begin
        # Create a stream with concept drift
        base_gen = RandomRBFGenerator(n_classes=2, n_features=3, n_centroids=5)
        stream = ConceptDriftStream(base_gen, drift_interval=100, drift_type=:sudden)

        model = LogisticRegression()
        metric = Accuracy()

        # Run prequential evaluation
        score, trajectory = progressive_val_score(model, stream, metric, n_samples=300)

        @test 0.0 <= score <= 1.0
        @test length(trajectory) >= 200
    end

    @testset "R2 with regression task" begin
        rng = Random.MersenneTwister(42)
        gen = SineGenerator(amplitude=2.0, noise=0.1, rng=rng)

        # Use a simple model (we'll use metrics directly since we don't have online regression)
        r2 = R2()
        mse = MSE()

        # Simulate a simple predictor that learns the mean
        running_mean = 0.0
        for i in 1:200
            x, y_true = generate(gen)

            # Simple predictor: running mean
            y_pred = running_mean

            # Update metrics
            fit!(r2, (y_pred, y_true))
            fit!(mse, (y_pred, y_true))

            # Update running mean
            running_mean = running_mean + (y_true - running_mean) / i
        end

        @test nobs(r2) == 200
        @test nobs(mse) == 200
        @test value(mse) >= 0.0
        # R² might be low or negative for simple mean predictor on sine data
        # Just check it's a valid number
        @test !isnan(value(r2))
    end
end
