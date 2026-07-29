using Test
using OnlineML
import OnlineML: Linear
using .Linear: Regression, LogisticRegression, Perceptron, PassiveAggressive, intercept, weights
using OnlineStats

@testset "Contract: Linear Models" begin
    @testset "Regression wrapper equivalence with OnlineStats.LinReg" begin
        # Contract: Our Regression wrapper adds intercept handling internally
        # When we fit the same data, our model should learn the same relationship
        # as OnlineStats.LinReg when we give it the intercept column

        # Test data - true relationship: y = 3 + 2*x1 - 1.5*x2 + 0.5*x3
        n_samples = 100
        n_features = 3
        X = randn(n_samples, n_features)
        y = X[:, 1] * 2.0 .+ X[:, 2] * (-1.5) .+ X[:, 3] * 0.5 .+ 3.0 .+ randn(n_samples) * 0.1

        # Our wrapper - handles intercept automatically
        our_model = Regression()

        # Direct OnlineStats.LinReg - needs explicit intercept column
        os_model = LinReg()

        # Fit both with same data
        for i in 1:n_samples
            x_i = X[i, :]
            y_i = y[i]
            fit!(our_model, (x_i, y_i))
            # For OnlineStats, add intercept column manually
            fit!(os_model, (vcat(1.0, x_i), y_i))
        end

        # Contract check: coefficients should match (both include intercept)
        our_coef = OnlineML.Linear.coef(our_model)
        os_coef = OnlineStats.coef(os_model)

        @test length(our_coef) == length(os_coef)
        @test isapprox(our_coef, os_coef, rtol=1e-10)

        # Contract check: observation counts should match
        @test nobs(our_model) == nobs(os_model) == n_samples

        # Contract check: predictions should match
        x_test = randn(n_features)
        our_pred = OnlineML.predict(our_model, x_test)
        # Manual prediction from OnlineStats coefficients
        os_pred = os_coef[1] + sum(os_coef[2:end] .* x_test)
        @test isapprox(our_pred, os_pred, rtol=1e-10)

        # Contract check: intercept and weights accessors
        @test isapprox(intercept(our_model), os_coef[1], rtol=1e-10)
        @test isapprox(weights(our_model), os_coef[2:end], rtol=1e-10)

        # Verify we learned something close to true coefficients
        @test isapprox(intercept(our_model), 3.0, atol=0.5)
        @test isapprox(weights(our_model)[1], 2.0, atol=0.3)
        @test isapprox(weights(our_model)[2], -1.5, atol=0.3)
        @test isapprox(weights(our_model)[3], 0.5, atol=0.3)
    end

    @testset "Regression with reset!" begin
        model = Regression()

        # Fit some data
        for _ in 1:20
            fit!(model, (randn(3), randn()))
        end
        @test nobs(model) == 20

        # Reset and verify state
        reset!(model)
        @test nobs(model) == 0
    end

    @testset "Regression with empty model" begin
        model = Regression()

        # Prediction on unfitted model should return zero
        @test OnlineML.predict(model, [1.0, 2.0, 3.0]) == 0.0
    end
end
