using Test
using OnlineML
using OnlineML.Transform

@testset "Transform" begin
    @testset "StandardScaler" begin
        scaler = StandardScaler{Float64}()
        @test nobs(scaler) == 0

        # Fit with some data
        for x in [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            fit!(scaler, x)
        end
        @test nobs(scaler) == 3

        # Transform should produce roughly standardized values
        result = OnlineML.transform(scaler, [3.0, 4.0])
        @test length(result) == 2

        # Reset
        reset!(scaler)
        @test nobs(scaler) == 0
    end

    @testset "MinMaxScaler" begin
        scaler = MinMaxScaler{Float64}()
        @test nobs(scaler) == 0

        # Fit with range data
        for x in [[0.0, 0.0], [10.0, 100.0], [5.0, 50.0]]
            fit!(scaler, x)
        end
        @test nobs(scaler) == 3

        # Transform should produce values in [0, 1]
        result = OnlineML.transform(scaler, [5.0, 50.0])
        @test all(0 .<= result .<= 1)

        # Reset
        reset!(scaler)
        @test nobs(scaler) == 0
    end

    @testset "OneHotEncoder" begin
        enc = OneHotEncoder()
        @test nobs(enc) == 0

        # Fit with categories
        for x in [:a, :b, :a, :c]
            fit!(enc, x)
        end
        @test nobs(enc) == 4
        @test length(value(enc)) == 3  # 3 unique categories

        # Transform
        result_a = OnlineML.transform(enc, :a)
        result_b = OnlineML.transform(enc, :b)
        @test sum(result_a) == 1.0
        @test sum(result_b) == 1.0
        @test result_a != result_b

        # Unknown category
        result_unknown = OnlineML.transform(enc, :unknown)
        @test sum(result_unknown) == 0.0

        # Reset
        reset!(enc)
        @test nobs(enc) == 0
    end

    @testset "OrdinalEncoder" begin
        enc = OrdinalEncoder()
        @test nobs(enc) == 0

        # Fit with categories
        for x in [:low, :medium, :high]
            fit!(enc, x)
        end
        @test nobs(enc) == 3

        # Transform returns ordinal values
        @test OnlineML.transform(enc, :low) == 1.0
        @test OnlineML.transform(enc, :medium) == 2.0
        @test OnlineML.transform(enc, :high) == 3.0

        # Unknown category
        @test OnlineML.transform(enc, :unknown) == 0.0
    end

    @testset "TargetEncoder" begin
        enc = TargetEncoder(smoothing=1.0)
        @test nobs(enc) == 0

        # Fit with category-target pairs
        for (cat, target) in zip([:a, :a, :b, :b], [1.0, 1.0, 0.0, 0.0])
            fit!(enc, (cat, target))
        end
        @test nobs(enc) == 4

        # Category :a has mean target 1.0, :b has mean target 0.0
        # With smoothing, values should be between category mean and global mean (0.5)
        result_a = OnlineML.transform(enc, :a)
        result_b = OnlineML.transform(enc, :b)
        @test result_a > 0.5  # Pulled toward 1.0
        @test result_b < 0.5  # Pulled toward 0.0

        # Unknown category gets global mean
        result_unknown = OnlineML.transform(enc, :unknown)
        @test isapprox(result_unknown, 0.5, atol=0.01)
    end

    @testset "MeanImputer" begin
        imp = MeanImputer()
        @test nobs(imp) == 0

        # Fit with some values (including missing)
        for x in [1.0, 2.0, missing, 4.0]
            fit!(imp, x)
        end
        @test nobs(imp) == 4

        # Mean of non-missing values is (1+2+4)/3 ≈ 2.33
        expected_mean = (1.0 + 2.0 + 4.0) / 3
        @test isapprox(value(imp), expected_mean, atol=0.01)

        # Transform missing values
        @test isapprox(OnlineML.transform(imp, missing), expected_mean, atol=0.01)
        @test OnlineML.transform(imp, 5.0) == 5.0

        # Reset
        reset!(imp)
        @test nobs(imp) == 0
    end

    @testset "ModeImputer" begin
        imp = ModeImputer()
        @test nobs(imp) == 0

        # Fit with some values (including missing)
        for x in [:a, :b, :a, missing, :a]
            fit!(imp, x)
        end
        @test nobs(imp) == 5

        # Mode should be :a (most frequent)
        @test mode(imp) == :a

        # Transform missing values
        @test OnlineML.transform(imp, missing) == :a
        @test OnlineML.transform(imp, :b) == :b

        # Reset
        reset!(imp)
        @test nobs(imp) == 0
    end
end
