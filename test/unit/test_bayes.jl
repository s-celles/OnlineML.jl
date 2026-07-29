using Test
using OnlineML
import OnlineML: Bayes
using .Bayes: GaussianNB, MultinomialNB, BernoulliNB, classes, class_prior

@testset "Naive Bayes" begin
    @testset "GaussianNB" begin
        gnb = GaussianNB()
        @test nobs(gnb) == 0

        # Fit with some data
        for i in 1:50
            x = randn(4)
            y = rand([0, 1, 2])
            fit!(gnb, (x, y))
        end
        @test nobs(gnb) == 50

        # Predict
        y_pred = predict(gnb, randn(4))
        @test y_pred in [0, 1, 2]

        # Predict proba should return probabilities
        probs = predict_proba(gnb, randn(4))
        @test isa(probs, Dict)
        @test all(0 <= v <= 1 for v in values(probs))
        @test isapprox(sum(values(probs)), 1.0, atol=1e-6)

        # Classes
        @test !isempty(classes(gnb))

        # Class prior
        priors = class_prior(gnb)
        @test isa(priors, Dict)
        @test all(0 <= v <= 1 for v in values(priors))

        # Reset
        reset!(gnb)
        @test nobs(gnb) == 0
    end

    @testset "MultinomialNB" begin
        mnb = MultinomialNB()
        @test nobs(mnb) == 0

        # Fit with count data (non-negative floats)
        for i in 1:30
            x = Float64.(rand(1:10, 5))  # Count data as Float64
            y = rand([0, 1])
            fit!(mnb, (x, y))
        end
        @test nobs(mnb) == 30

        # Predict
        y_pred = predict(mnb, Float64.(rand(1:10, 5)))
        @test y_pred in [0, 1]
    end

    @testset "BernoulliNB" begin
        bnb = BernoulliNB()
        @test nobs(bnb) == 0

        # Fit with binary data (as Float64)
        for i in 1:30
            x = Float64.(rand([0, 1], 5))
            y = rand([0, 1])
            fit!(bnb, (x, y))
        end
        @test nobs(bnb) == 30

        # Predict
        y_pred = predict(bnb, Float64.(rand([0, 1], 5)))
        @test y_pred in [0, 1]
    end
end
