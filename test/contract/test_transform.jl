using Test
using OnlineML
import OnlineML: Transform
using .Transform: StandardScaler, MinMaxScaler, MaxAbsScaler
using OnlineStats

@testset "Contract: Transform" begin
    @testset "StandardScaler wrapper equivalence with OnlineStats.Variance" begin
        # Contract: StandardScaler should produce same results as OnlineStats.Variance
        # when computing mean and std

        scaler = StandardScaler{Float64}()
        var_stats = [OnlineStats.Variance(Float64) for _ in 1:3]

        # Same training data
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]

        for x in data
            fit!(scaler, x)
            for (i, v) in enumerate(x)
                fit!(var_stats[i], v)
            end
        end

        @test nobs(scaler) == length(data)

        # Check that means and stds match
        for (i, vs) in enumerate(var_stats)
            scaler_mean = scaler.stats[i] |> OnlineStats.mean
            scaler_std = sqrt(scaler.stats[i] |> OnlineStats.var)
            @test isapprox(scaler_mean, OnlineStats.mean(vs), rtol=1e-10)
            @test isapprox(scaler_std, sqrt(OnlineStats.var(vs)), rtol=1e-10)
        end

        # Transform should produce standardized output
        x_test = [5.5, 6.5, 7.5]
        result = OnlineML.transform(scaler, x_test)
        @test length(result) == 3
        # Result should be centered around 0 for middle values
    end

    @testset "MinMaxScaler wrapper equivalence with OnlineStats.Extrema" begin
        scaler = MinMaxScaler{Float64}()
        extrema_stats = [OnlineStats.Extrema(Float64) for _ in 1:2]

        # Same training data
        data = [[0.0, 10.0], [5.0, 50.0], [10.0, 100.0]]

        for x in data
            fit!(scaler, x)
            for (i, v) in enumerate(x)
                fit!(extrema_stats[i], v)
            end
        end

        @test nobs(scaler) == length(data)

        # Check that min/max match
        for (i, es) in enumerate(extrema_stats)
            scaler_min, scaler_max = OnlineStatsBase.value(scaler.extrema[i])
            es_vals = OnlineStatsBase.value(es)
            @test isapprox(scaler_min, es_vals[1], rtol=1e-10)
            @test isapprox(scaler_max, es_vals[2], rtol=1e-10)
        end

        # Transform should produce values in [0, 1]
        x_test = [5.0, 55.0]  # Mid-points of [0,10] and [10,100]
        result = OnlineML.transform(scaler, x_test)
        @test all(0.0 .<= result .<= 1.0)
        @test isapprox(result, [0.5, 0.5], rtol=0.01)  # Middle values should be 0.5
    end
end
