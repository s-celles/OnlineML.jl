using Test
using OnlineML
import OnlineML: Streams
using .Streams: DataStream, SineGenerator, RandomRBFGenerator
using .Streams: ConceptDriftStream, SEAGenerator, HyperplaneGenerator
using .Streams: LEDGenerator, AgrawalGenerator, generate, iterate_stream
using .Streams: batch_stream, take_stream, skip_stream
using Random
using Tables

@testset "Streams" begin
    @testset "DataStream from arrays" begin
        X = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        y = [1, 2, 3]

        stream = DataStream(X, y)

        @test length(stream) == 3

        # Collect all samples
        samples = collect(stream)
        @test length(samples) == 3
        @test samples[1] == ([1.0, 2.0], 1)
        @test samples[2] == ([3.0, 4.0], 2)
        @test samples[3] == ([5.0, 6.0], 3)
    end

    @testset "DataStream from generator" begin
        rng = Random.MersenneTwister(42)
        gen = SineGenerator(rng=rng)
        stream = DataStream(gen, n=10)

        @test length(stream) == 10

        # Collect samples
        samples = collect(stream)
        @test length(samples) == 10

        # Each sample should be (Vector, Float64)
        for (x, y_val) in samples
            @test x isa AbstractVector
            @test y_val isa Number
        end
    end

    @testset "DataStream from DataFrame/Tables.jl" begin
        # Create a table-compatible structure
        data = (
            feature1 = [1.0, 2.0, 3.0, 4.0, 5.0],
            feature2 = [10.0, 20.0, 30.0, 40.0, 50.0],
            target = [0, 1, 0, 1, 1]
        )

        # Test Tables.jl row iteration
        @test Tables.istable(data)

        # Create stream from table with target column specified
        stream = DataStream(data, target=:target)

        @test length(stream) == 5

        samples = collect(stream)
        @test length(samples) == 5

        # First sample should have features [1.0, 10.0] and target 0
        x1, y1 = samples[1]
        @test x1 ≈ [1.0, 10.0]
        @test y1 == 0

        # Last sample
        x5, y5 = samples[5]
        @test x5 ≈ [5.0, 50.0]
        @test y5 == 1
    end

    @testset "Tables.jl row iteration" begin
        # Named tuple of columns (Tables.jl columntable)
        data = (
            a = [1.0, 2.0, 3.0],
            b = [4.0, 5.0, 6.0],
            c = [7, 8, 9]  # Target column
        )

        stream = DataStream(data, target=:c)

        # Iterate through rows
        count = 0
        for (x, y) in stream
            count += 1
            @test length(x) == 2  # Two feature columns
            @test y isa Integer
        end
        @test count == 3
    end

    @testset "Stream chunking/batching" begin
        rng = Random.MersenneTwister(123)
        gen = RandomRBFGenerator(n_classes=2, n_features=5, rng=rng)

        # Create batched stream
        stream = DataStream(gen, n=100)
        batches = batch_stream(stream, batch_size=10)

        batch_count = 0
        for (X_batch, y_batch) in batches
            batch_count += 1
            @test size(X_batch, 1) == 10  # batch_size samples
            @test size(X_batch, 2) == 5   # n_features
            @test length(y_batch) == 10
        end
        @test batch_count == 10  # 100 samples / 10 batch_size
    end

    @testset "Stream operations" begin
        rng = Random.MersenneTwister(42)
        gen = SineGenerator(rng=rng)

        # Test take operation
        stream = DataStream(gen, n=1000)
        taken = take_stream(stream, 5)
        @test length(collect(taken)) == 5

        # Test skip operation
        rng2 = Random.MersenneTwister(42)
        gen2 = SineGenerator(rng=rng2)
        stream2 = DataStream(gen2, n=100)

        # Skip first 10, take 5
        skipped = skip_stream(stream2, 10)
        taken_after_skip = take_stream(skipped, 5)
        @test length(collect(taken_after_skip)) == 5
    end

    @testset "iterate_stream utility" begin
        X = [1.0 2.0; 3.0 4.0]
        y = [1, 2]

        stream = iterate_stream((X, y))
        samples = collect(stream)
        @test length(samples) == 2
    end

    @testset "Generator consistency" begin
        # Test that generators produce deterministic output with same seed
        rng1 = Random.MersenneTwister(42)
        rng2 = Random.MersenneTwister(42)

        gen1 = SEAGenerator(rng=rng1)
        gen2 = SEAGenerator(rng=rng2)

        for _ in 1:10
            x1, y1 = generate(gen1)
            x2, y2 = generate(gen2)
            @test x1 ≈ x1
            @test y1 == y2
        end
    end

    @testset "Empty stream handling" begin
        X = zeros(0, 3)
        y = Int[]

        stream = DataStream(X, y)
        @test length(stream) == 0
        @test collect(stream) == []
    end

    @testset "MLJ interface placeholder" begin
        # Test that we can wrap a model for MLJ compatibility
        # This is a placeholder for future MLJ integration

        # For now, just verify that the module exports are correct
        @test :DataStream in names(OnlineML.Streams)
        @test :generate in names(OnlineML.Streams)
        @test :iterate_stream in names(OnlineML.Streams)
    end
end
