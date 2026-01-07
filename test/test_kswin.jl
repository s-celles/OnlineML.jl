# Tests for KSWIN Drift Detector

using Test
using OnlineML
import OnlineML: Drift
using .Drift: KSWIN, status

@testset "KSWIN" begin
    @testset "Constructor and defaults" begin
        detector = KSWIN()
        @test detector isa KSWIN
        @test nobs(detector) == 0
        @test detector.alpha == 0.005
        @test detector.window_size == 100
        @test detector.stat_size == 30
    end

    @testset "Constructor with parameters" begin
        detector = KSWIN(alpha=0.01, window_size=200, stat_size=50)
        @test detector.alpha == 0.01
        @test detector.window_size == 200
        @test detector.stat_size == 50
    end

    @testset "Invalid parameters" begin
        @test_throws ArgumentError KSWIN(alpha=0.0)
        @test_throws ArgumentError KSWIN(alpha=1.0)
        @test_throws ArgumentError KSWIN(window_size=0)
        @test_throws ArgumentError KSWIN(stat_size=0)
        @test_throws ArgumentError KSWIN(window_size=100, stat_size=60)  # stat_size > window_size/2
    end

    @testset "Basic fit! and status" begin
        detector = KSWIN(window_size=50, stat_size=20)

        # Feed stable data
        for i in 1:60
            fit!(detector, randn())
        end

        @test nobs(detector) == 60
        # Status should be available
        s = status(detector)
        @test s isa DriftStatus
    end

    @testset "reset!" begin
        detector = KSWIN(window_size=50, stat_size=20)

        for i in 1:30
            fit!(detector, randn())
        end

        @test nobs(detector) == 30

        reset!(detector)

        @test nobs(detector) == 0
        @test status(detector) == NoDrift
    end

    @testset "Drift detection on distribution change" begin
        detector = KSWIN(alpha=0.01, window_size=100, stat_size=40)

        # Feed data from N(0, 1) first
        for i in 1:100
            fit!(detector, randn())
        end

        initial_status = status(detector)

        # Now feed data from N(5, 1) - significant mean shift
        drift_detected = false
        for i in 1:100
            fit!(detector, 5.0 + randn())
            if detected_drift(detector)
                drift_detected = true
                break
            end
        end

        # Should eventually detect drift with such a large shift
        @test drift_detected
    end

    @testset "No drift on stable distribution" begin
        detector = KSWIN(alpha=0.001, window_size=100, stat_size=40)

        # Feed data from same distribution
        drift_count = 0
        for i in 1:500
            fit!(detector, randn())
            if detected_drift(detector)
                drift_count += 1
            end
        end

        # With alpha=0.001, we expect very few false positives
        # Should be less than 5% of checks (after window fills)
        @test drift_count < 25
    end

    @testset "update! returns status" begin
        detector = KSWIN()
        s = update!(detector, 1.0)
        @test s isa DriftStatus
    end

    @testset "value and nobs" begin
        detector = KSWIN()
        @test nobs(detector) == 0
        @test value(detector) == 0.0

        fit!(detector, 5.0)
        @test nobs(detector) == 1
        @test value(detector) == 5.0

        fit!(detector, 3.0)
        @test nobs(detector) == 2
        @test value(detector) == 3.0  # Last value
    end

    @testset "detected_drift and detected_warning" begin
        detector = KSWIN(window_size=50, stat_size=20)

        # Initially no drift
        @test !detected_drift(detector)
        @test !detected_warning(detector)
    end

    @testset "window_size function" begin
        detector = KSWIN(window_size=100, stat_size=30)

        for i in 1:50
            fit!(detector, randn())
        end

        @test Drift.window_size(detector) == 50

        for i in 1:100
            fit!(detector, randn())
        end

        @test Drift.window_size(detector) == 100  # Capped at window_size
    end

    @testset "KS statistic computation" begin
        # Test the internal KS statistic function
        sample1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample2 = [1.0, 2.0, 3.0, 4.0, 5.0]

        # Identical samples should have KS stat of 0
        @test Drift.ks_statistic(sample1, sample2) == 0.0

        # Completely different samples should have high KS stat
        sample3 = [10.0, 11.0, 12.0, 13.0, 14.0]
        ks = Drift.ks_statistic(sample1, sample3)
        @test ks == 1.0  # Maximum possible difference
    end
end
