using Test
using OnlineML
import OnlineML: Instance
using .Instance: KNN

@testset "Instance-Based Models" begin
    @testset "KNN" begin
        knn = KNN(k=5)
        @test nobs(knn) == 0

        # Fit with some data
        for i in 1:50
            x = randn(4)
            y = rand([0, 1])
            fit!(knn, (x, y))
        end
        @test nobs(knn) == 50

        # Predict
        y_pred = predict(knn, randn(4))
        @test y_pred in [0, 1]

        # Predict proba
        probs = predict_proba(knn, randn(4))
        @test isa(probs, Dict)
        if !isempty(probs)
            @test all(0 <= v <= 1 for v in values(probs))
        end

        # Reset
        reset!(knn)
        @test nobs(knn) == 0
    end

    @testset "KNN with window_size" begin
        knn = KNN(k=3, window_size=20)

        # Add more than window size
        for i in 1:50
            x = randn(3)
            y = rand([0, 1])
            fit!(knn, (x, y))
        end

        # Should only keep window_size observations
        val = value(knn)
        @test length(val.X) <= 20
        @test length(val.y) <= 20
    end

    @testset "KNN correctly classifies simple data" begin
        knn = KNN(k=3, weighted=false)

        # Add clearly separable data
        # Class 0: points around origin
        for i in 1:20
            fit!(knn, ([0.0, 0.0] .+ 0.1 * randn(2), 0))
        end

        # Class 1: points around (10, 10)
        for i in 1:20
            fit!(knn, ([10.0, 10.0] .+ 0.1 * randn(2), 1))
        end

        # Test classification
        @test predict(knn, [0.0, 0.0]) == 0
        @test predict(knn, [10.0, 10.0]) == 1
    end
end
