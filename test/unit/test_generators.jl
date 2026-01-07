using Test
using OnlineML
using OnlineML.Streams: SineGenerator, RandomRBFGenerator, ConceptDriftStream
using OnlineML.Streams: LEDGenerator, AgrawalGenerator, SEAGenerator, HyperplaneGenerator
using OnlineML.Streams: generate, initialize!
using Random

@testset "Generators" begin
    @testset "AgrawalGenerator output format" begin
        gen = AgrawalGenerator(function_id=1)

        for _ in 1:10
            x, y = generate(gen)
            @test length(x) == 9  # 9 demographic attributes
            @test y in [0, 1]  # Binary classification
        end

        # Test all 10 functions
        for func_id in 1:10
            gen = AgrawalGenerator(function_id=func_id)
            x, y = generate(gen)
            @test length(x) == 9
            @test y in [0, 1]
        end
    end

    @testset "SEAGenerator with noise" begin
        gen = SEAGenerator(function_id=1, noise=0.0)

        for _ in 1:10
            x, y = generate(gen)
            @test length(x) == 3  # 3 features
            @test all(0 .<= x .<= 10)  # Range [0, 10]
            @test y in [0, 1]
        end

        # Test all 4 functions
        for func_id in 1:4
            gen = SEAGenerator(function_id=func_id)
            x, y = generate(gen)
            @test length(x) == 3
            @test y in [0, 1]
        end
    end

    @testset "HyperplaneGenerator with drift" begin
        gen = HyperplaneGenerator(n_features=5, mag_change=0.0)

        for _ in 1:10
            x, y = generate(gen)
            @test length(x) == 5
            @test all(0 .<= x .<= 1)  # Range [0, 1]
            @test y in [0, 1]
        end

        # Test with drift
        gen_drift = HyperplaneGenerator(n_features=5, n_drift_features=2, mag_change=0.1)
        initial_weights = copy(gen_drift.weights)

        for _ in 1:100
            generate(gen_drift)
        end

        # Weights should have changed
        @test gen_drift.weights != initial_weights
    end

    @testset "LEDGenerator" begin
        gen = LEDGenerator(noise=0.0, irrelevant_features=false)

        for _ in 1:20
            x, y = generate(gen)
            @test length(x) == 7  # 7 LED segments
            @test y in 0:9  # Digit 0-9
            @test all(v in [0.0, 1.0] for v in x)  # Binary features
        end

        # With irrelevant features
        gen_full = LEDGenerator(noise=0.0, irrelevant_features=true)
        x, y = generate(gen_full)
        @test length(x) == 24  # 7 + 17 irrelevant
    end

    @testset "SineGenerator" begin
        gen = SineGenerator(amplitude=1.0, frequency=0.1, noise=0.0)

        for _ in 1:10
            x, y = generate(gen)
            @test length(x) == 1  # Time step
            @test typeof(y) == Float64
        end

        # Check that values are bounded
        gen_bounded = SineGenerator(amplitude=2.0, noise=0.0)
        for i in 1:100
            x, y = generate(gen_bounded)
            @test -2.5 <= y <= 2.5  # amplitude + small tolerance
        end
    end

    @testset "RandomRBFGenerator" begin
        gen = RandomRBFGenerator(n_classes=3, n_features=5, n_centroids=10)

        for _ in 1:20
            x, y = generate(gen)
            @test length(x) == 5
            @test y in 1:3  # Classes
        end

        # Check centroid count
        @test length(gen.centroids) == 10
    end

    @testset "ConceptDriftStream transition" begin
        base = RandomRBFGenerator(n_classes=2, n_features=3, n_centroids=5)
        stream = ConceptDriftStream(base, drift_interval=50, drift_type=:sudden)

        # Generate samples before drift
        initial_centroids = copy(base.centroids)

        for i in 1:45
            x, y = generate(stream)
            @test length(x) == 3
            @test y in 1:2
        end

        # Centroids should be same before drift point
        @test base.centroids == initial_centroids

        # Cross drift point
        for i in 46:55
            generate(stream)
        end

        # After drift, centroids should be different
        @test base.centroids != initial_centroids
    end

    @testset "Seed reproducibility across generators" begin
        # Test that same seed produces same sequence
        rng1 = Random.MersenneTwister(42)
        rng2 = Random.MersenneTwister(42)

        gen1 = SEAGenerator(rng=rng1)
        gen2 = SEAGenerator(rng=rng2)

        for _ in 1:10
            x1, y1 = generate(gen1)
            x2, y2 = generate(gen2)
            @test x1 == x2
            @test y1 == y2
        end

        # Test for HyperplaneGenerator
        rng1 = Random.MersenneTwister(123)
        rng2 = Random.MersenneTwister(123)

        gen1 = HyperplaneGenerator(n_features=5, rng=rng1)
        gen2 = HyperplaneGenerator(n_features=5, rng=rng2)

        for _ in 1:10
            x1, y1 = generate(gen1)
            x2, y2 = generate(gen2)
            @test x1 == x2
            @test y1 == y2
        end
    end
end
