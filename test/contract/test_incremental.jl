using OnlineML
using OnlineML.Bayes: GaussianNB
using OnlineML.Cluster: StreamingKMeans
using OnlineML.Instance: KNN
using OnlineML.Linear: LogisticRegression
using OnlineML.Trees: HoeffdingTree

mutable struct OneShot{T}
    data::T
    started::Bool
end

function Base.iterate(stream::OneShot)
    stream.started && error("stream was traversed more than once")
    stream.started = true
    result = iterate(stream.data)
    result === nothing && return nothing
    value, state = result
    return value, state
end

Base.iterate(stream::OneShot, state) = iterate(stream.data, state)

@testset "supervised learners consume two batches" begin
    factories = (
        LogisticRegression,
        GaussianNB,
        () -> HoeffdingTree(grace_period=100),
        () -> KNN(k=1, window_size=10),
    )
    first_batch = [([0.0, 0.0], 0), ([1.0, 1.0], 1)]
    second_batch = [([0.1, 0.2], 0), ([0.9, 0.8], 1)]

    for factory in factories
        model = factory()
        @test fit_batch!(model, first_batch) === model
        @test nobs(model) == 2
        fit_batch!(model, second_batch)
        @test nobs(model) == 4
        @test OnlineML.predict(model, [0.95, 0.9]) isa Int
    end
end

@testset "a non-reiterable stream is traversed once" begin
    model = GaussianNB()
    stream = OneShot([([0.0], 0), ([1.0], 1)], false)

    fit_batch!(model, stream)

    @test stream.started
    @test nobs(model) == 2
end

@testset "unsupervised batches preserve order and identity" begin
    model = StreamingKMeans(k=2)
    stream = OneShot([[0.0, 0.0], [10.0, 10.0], [0.1, 0.1]], false)

    @test fit_batch!(model, stream) === model
    @test nobs(model) == 3
    @test OnlineML.predict(model, [0.0, 0.0]) == 1
end

@testset "errors retain the committed prefix" begin
    model = GaussianNB()
    malformed = Any[([0.0], 0), [1.0], ([2.0], 1)]

    @test_throws ArgumentError fit_batch!(model, malformed)
    @test nobs(model) == 1
end

@testset "reset starts a new incremental history" begin
    model = KNN(k=1, window_size=2)
    fit_batch!(model, [([0.0], 0), ([1.0], 1), ([2.0], 1)])
    @test nobs(model) == 3
    @test length(value(model).X) == 2

    reset!(model)
    @test nobs(model) == 0
    fit_batch!(model, [([3.0], 1)])
    @test nobs(model) == 1
end
