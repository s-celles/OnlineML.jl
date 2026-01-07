using Test
using OnlineML
using OnlineML.Pipeline
using OnlineML.Transform

@testset "Pipeline" begin
    @testset "FeatureUnion" begin
        scaler1 = StandardScaler{Float64}()
        scaler2 = MinMaxScaler{Float64}()
        union = FeatureUnion(scaler1, scaler2)

        @test nobs(union) == 0

        # Fit with some data
        for x in [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            fit!(union, x)
        end
        @test nobs(union) == 3

        # Transform should concatenate outputs
        result = OnlineML.transform(union, [3.0, 4.0])
        @test length(result) == 4  # 2 from each scaler

        # Reset
        reset!(union)
        @test nobs(union) == 0
    end

    @testset "FeatureUnion + operator" begin
        scaler1 = StandardScaler{Float64}()
        scaler2 = MinMaxScaler{Float64}()

        # Using + operator
        union = scaler1 + scaler2
        @test union isa FeatureUnion
        @test length(union.transformers) == 2
    end

    @testset "ColumnTransformer" begin
        ct = ColumnTransformer([
            ("first", StandardScaler{Float64}(), [1]),
            ("second", MinMaxScaler{Float64}(), [2])
        ])

        @test nobs(ct) == 0

        # Fit with some data
        for x in [[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]]
            fit!(ct, x)
        end
        @test nobs(ct) == 3

        # Transform
        result = OnlineML.transform(ct, [3.0, 30.0])
        @test length(result) == 2

        # Reset
        reset!(ct)
        @test nobs(ct) == 0
    end

    @testset "ColumnTransformer with ranges" begin
        ct = ColumnTransformer([
            ("all", StandardScaler{Float64}(), 1:2)
        ])

        # Fit and transform
        fit!(ct, [1.0, 2.0])
        fit!(ct, [3.0, 4.0])
        result = OnlineML.transform(ct, [2.0, 3.0])
        @test length(result) == 2
    end
end
