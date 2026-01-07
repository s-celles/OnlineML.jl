using Test
using OnlineML
using OnlineML.Pipeline
using OnlineML.Transform
using OnlineML.Linear

@testset "Integration: Pipeline" begin
    @testset "StandardScaler |> LogisticRegression pipeline" begin
        # Create pipeline
        scaler = StandardScaler{Float64}()
        lr = LogisticRegression()
        pipe = OnlinePipeline(scaler, lr)

        @test nobs(pipe) == 0

        # Fit with some training data
        # Binary classification with scaled features
        for i in 1:50
            x = randn(3)
            y = x[1] > 0 ? 1 : 0  # Simple linear boundary
            fit!(pipe, (x, y))
        end

        @test nobs(pipe) == 50

        # Predict
        y_pred = OnlineML.predict(pipe, randn(3))
        @test y_pred in [0, 1]

        # Reset
        reset!(pipe)
        @test nobs(pipe) == 0
    end

    @testset "Pipeline with multiple transformers" begin
        # MinMaxScaler followed by StandardScaler (unusual but valid)
        pipe = OnlinePipeline(
            MinMaxScaler{Float64}(),
            StandardScaler{Float64}(),
            LogisticRegression()
        )

        # Fit
        for i in 1:30
            x = randn(2) .* 10  # Large scale data
            y = rand([0, 1])
            fit!(pipe, (x, y))
        end

        @test nobs(pipe) == 30

        # Predict should work
        y_pred = OnlineML.predict(pipe, randn(2) .* 10)
        @test y_pred in [0, 1]
    end

    @testset "Pipeline transform-only mode" begin
        # Pipeline of just transformers
        scaler = StandardScaler{Float64}()

        # Fit scaler
        for x in [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            fit!(scaler, x)
        end

        # Transform
        result = OnlineML.transform(scaler, [3.0, 4.0])
        @test length(result) == 2
        @test all(isfinite.(result))
    end

    @testset "FeatureUnion in pipeline" begin
        # Create feature union
        union = FeatureUnion(
            StandardScaler{Float64}(),
            MinMaxScaler{Float64}()
        )

        # Fit
        for x in [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
            fit!(union, x)
        end

        @test nobs(union) == 4

        # Transform should concatenate
        result = OnlineML.transform(union, [4.0, 5.0])
        @test length(result) == 4  # 2 from each scaler
    end
end
