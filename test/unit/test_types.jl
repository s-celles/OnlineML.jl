using Test
using OnlineML
using OnlineStatsBase

@testset "Core Types" begin
    @testset "DriftStatus enum" begin
        # Get enum values directly (Drift submodule shadows the enum value)
        drift_status_warning = OnlineML.Warning
        drift_status_drift = DriftStatus(2)  # Drift is the 3rd value (0-indexed becomes 2)

        @test NoDrift isa DriftStatus
        @test drift_status_warning isa DriftStatus
        @test drift_status_drift isa DriftStatus
        @test NoDrift != drift_status_warning
        @test drift_status_warning != drift_status_drift
        @test NoDrift != drift_status_drift
    end

    @testset "Abstract type hierarchy" begin
        # Verify abstract types are subtypes of OnlineStat
        @test Learner{Vector{Float64}, Float64} <: OnlineStat{Tuple{Vector{Float64}, Float64}}
        @test UnsupervisedLearner{Vector{Float64}} <: OnlineStat{Vector{Float64}}
        @test Transformer{Vector{Float64}} <: OnlineStat{Vector{Float64}}
        @test Detector <: OnlineStat{Float64}
        @test Metric{Int} <: OnlineStat{Tuple{Int, Int}}
    end

    @testset "Interface functions exist" begin
        # Verify interface functions are defined
        @test isdefined(OnlineML, :predict)
        @test isdefined(OnlineML, :predict_proba)
        @test isdefined(OnlineML, :transform)
        @test isdefined(OnlineML, :fit_transform!)
        @test isdefined(OnlineML, :inverse_transform)
        @test isdefined(OnlineML, :update!)
        @test isdefined(OnlineML, :detected_drift)
        @test isdefined(OnlineML, :detected_warning)
        @test isdefined(OnlineML, :score_one)
        @test isdefined(OnlineML, :is_anomaly)
        @test isdefined(OnlineML, :fit_score!)
        @test isdefined(OnlineML, :fit_predict!)
    end

    @testset "Optim submodule" begin
        using OnlineML.Optim
        using Optimisers

        # Verify optimizers are exported
        @test isdefined(Optim, :Adam)
        @test isdefined(Optim, :SGD)
        @test isdefined(Optim, :RMSProp)
        @test isdefined(Optim, :AdaGrad)
        @test isdefined(Optim, :Momentum)
        @test isdefined(Optim, :AdamW)

        # Test default optimizer
        opt = Optim.default_optimizer()
        @test opt isa Optimisers.Adam
    end
end
