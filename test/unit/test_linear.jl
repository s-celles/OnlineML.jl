using Test
using OnlineML
using OnlineML.Linear

@testset "Linear Models" begin
    @testset "LogisticRegression" begin
        lr = LogisticRegression()
        @test nobs(lr) == 0

        # Fit with some data
        for i in 1:20
            x = randn(5)
            y = rand([0, 1])
            fit!(lr, (x, y))
        end
        @test nobs(lr) == 20

        # Predict
        y_pred = predict(lr, randn(5))
        @test y_pred in [0, 1]

        # Predict proba
        probs = predict_proba(lr, randn(5))
        @test isa(probs, Dict)
        if !isempty(probs)
            @test all(0 <= v <= 1 for v in values(probs))
        end

        # Reset
        reset!(lr)
        @test nobs(lr) == 0
    end

    @testset "Perceptron" begin
        perc = Perceptron()
        @test nobs(perc) == 0

        for i in 1:20
            x = randn(5)
            y = rand([0, 1])
            fit!(perc, (x, y))
        end
        @test nobs(perc) == 20

        y_pred = predict(perc, randn(5))
        @test y_pred in [0, 1]
    end

    @testset "fit_predict! for test-then-train" begin
        lr = LogisticRegression()

        # First prediction should be 0 (no training data)
        # fit_predict! takes a Learner and tuple
        fit!(lr, (randn(5), 1))  # First train
        y_pred = fit_predict!(lr, randn(5), 1)  # Then test-then-train
        @test y_pred in [0, 1]  # Should make a valid prediction
        @test nobs(lr) == 2  # Should have trained twice
    end
end
