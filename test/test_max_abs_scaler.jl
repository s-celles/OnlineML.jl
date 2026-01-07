# Tests for MaxAbsScaler

using Test
using OnlineML
using OnlineML.Transform

@testset "MaxAbsScaler" begin
    @testset "Constructor and defaults" begin
        scaler = MaxAbsScaler()
        @test scaler isa MaxAbsScaler{Float64}
        @test nobs(scaler) == 0
        @test Transform.n_features(scaler) == 0
    end

    @testset "Type parameter" begin
        scaler32 = MaxAbsScaler{Float32}()
        @test scaler32 isa MaxAbsScaler{Float32}

        scaler64 = MaxAbsScaler{Float64}()
        @test scaler64 isa MaxAbsScaler{Float64}
    end

    @testset "Basic fit!" begin
        scaler = MaxAbsScaler()

        fit!(scaler, [2.0, -4.0, 1.0])
        @test nobs(scaler) == 1
        @test Transform.n_features(scaler) == 3
        @test Transform.max_abs_values(scaler) == [2.0, 4.0, 1.0]

        # Second observation with larger values
        fit!(scaler, [5.0, -2.0, 3.0])
        @test nobs(scaler) == 2
        @test Transform.max_abs_values(scaler) == [5.0, 4.0, 3.0]
    end

    @testset "transform" begin
        scaler = MaxAbsScaler()
        fit!(scaler, [2.0, -4.0, 1.0])

        result = transform(scaler, [2.0, -4.0, 1.0])
        @test result ≈ [1.0, -1.0, 1.0]

        result2 = transform(scaler, [1.0, 2.0, 0.5])
        @test result2 ≈ [0.5, 0.5, 0.5]
    end

    @testset "transform scales to [-1, 1]" begin
        scaler = MaxAbsScaler()

        # Train on various values
        for _ in 1:100
            fit!(scaler, randn(5) .* 10)
        end

        # Transform should produce values in [-1, 1]
        for _ in 1:20
            x = randn(5) .* 5
            result = transform(scaler, x)
            @test all(-1 .<= result .<= 1) || any(abs.(x) .> Transform.max_abs_values(scaler))
        end
    end

    @testset "reset!" begin
        scaler = MaxAbsScaler()

        for i in 1:10
            fit!(scaler, randn(3))
        end

        @test nobs(scaler) == 10

        reset!(scaler)

        @test nobs(scaler) == 0
        @test Transform.n_features(scaler) == 0
    end

    @testset "fit_transform!" begin
        scaler = MaxAbsScaler()

        result1 = fit_transform!(scaler, [2.0, -4.0, 1.0])
        @test nobs(scaler) == 1
        @test result1 ≈ [1.0, -1.0, 1.0]

        result2 = fit_transform!(scaler, [4.0, -2.0, 0.5])
        @test nobs(scaler) == 2
        @test result2 ≈ [1.0, -0.5, 0.5]
    end

    @testset "inverse_transform" begin
        scaler = MaxAbsScaler()
        fit!(scaler, [2.0, -4.0, 1.0])

        # Transform then inverse should recover original
        original = [1.0, -2.0, 0.5]
        transformed = transform(scaler, original)
        recovered = inverse_transform(scaler, transformed)

        @test recovered ≈ original
    end

    @testset "inverse_transform roundtrip" begin
        scaler = MaxAbsScaler()

        # Train on data
        for _ in 1:50
            fit!(scaler, randn(4))
        end

        # Test roundtrip
        for _ in 1:10
            x = randn(4)
            transformed = transform(scaler, x)
            recovered = inverse_transform(scaler, transformed)
            @test recovered ≈ x atol=1e-10
        end
    end

    @testset "Zero handling" begin
        scaler = MaxAbsScaler()
        fit!(scaler, [0.0, 2.0, 0.0])

        # Zero max_abs should leave value unchanged
        result = transform(scaler, [1.0, 1.0, 1.0])
        @test result[1] == 1.0  # No scaling for feature with max_abs=0
        @test result[2] == 0.5
        @test result[3] == 1.0
    end

    @testset "value and nobs" begin
        scaler = MaxAbsScaler()
        @test nobs(scaler) == 0
        @test isempty(value(scaler))

        fit!(scaler, [1.0, 2.0])
        @test nobs(scaler) == 1
        @test value(scaler) == [1.0, 2.0]
    end

    @testset "Growing feature dimension" begin
        scaler = MaxAbsScaler()

        # First observation with 2 features
        fit!(scaler, [1.0, 2.0])
        @test Transform.n_features(scaler) == 2

        # Second observation with 4 features - should grow
        fit!(scaler, [3.0, 4.0, 5.0, 6.0])
        @test Transform.n_features(scaler) == 4
        @test Transform.max_abs_values(scaler) == [3.0, 4.0, 5.0, 6.0]
    end

    @testset "Negative values" begin
        scaler = MaxAbsScaler()

        fit!(scaler, [-5.0, 3.0])
        @test Transform.max_abs_values(scaler) == [5.0, 3.0]

        result = transform(scaler, [-5.0, 3.0])
        @test result ≈ [-1.0, 1.0]

        result2 = transform(scaler, [5.0, -3.0])
        @test result2 ≈ [1.0, -1.0]
    end

    @testset "Empty transform" begin
        scaler = MaxAbsScaler()

        # Transform without fitting should return input unchanged
        result = transform(scaler, [1.0, 2.0, 3.0])
        @test result == [1.0, 2.0, 3.0]
    end
end
