using OnlineML
using OnlineML.Bayes: GaussianNB, classes
using OnlineML.Linear: LogisticRegression, Regression, coef

mutable struct ReferenceOneShot{T}
    data::T
    started::Bool
end

function Base.iterate(stream::ReferenceOneShot)
    stream.started && error("stream was traversed more than once")
    stream.started = true
    result = iterate(stream.data)
    result === nothing && return nothing
    observation, state = result
    return observation, state
end

Base.iterate(stream::ReferenceOneShot, state) = iterate(stream.data, state)

function test_common_incremental_contract(factory, first_batch, second_batch, probe)
    model = factory()

    @test fit_batch!(model, ReferenceOneShot(first_batch, false)) === model
    @test nobs(model) == length(first_batch)

    n_before_predict = nobs(model)
    OnlineML.predict(model, probe)
    @test nobs(model) == n_before_predict

    @test fit_batch!(model, second_batch) === model
    @test nobs(model) == length(first_batch) + length(second_batch)

    @test reset!(model) === model
    @test nobs(model) == 0
end

@testset "common incremental lifecycle" begin
    test_common_incremental_contract(
        Regression,
        [([0.0], 1.0), ([1.0], 3.0)],
        [([2.0], 5.0), ([3.0], 7.0)],
        [1.5],
    )
    test_common_incremental_contract(
        LogisticRegression,
        [([-1.0], 0), ([1.0], 1)],
        [([-2.0], 0), ([2.0], 1)],
        [0.5],
    )
    test_common_incremental_contract(
        GaussianNB,
        [([0.0], 0), ([1.0], 1)],
        [([0.1], 0), ([0.9], 1)],
        [0.5],
    )
end

@testset "Regression has a mergeable OnlineStats state" begin
    observations = [
        ([0.0], 1.0),
        ([1.0], 3.0),
        ([2.0], 5.0),
        ([3.0], 7.0),
        ([4.0], 9.0),
        ([5.0], 11.0),
    ]
    left = Regression()
    right = Regression()
    whole = Regression()

    fit_batch!(left, observations[1:3])
    fit_batch!(right, observations[4:6])
    fit_batch!(whole, observations)
    merge!(left, right)

    @test nobs(left) == nobs(whole) == length(observations)
    @test coef(left) ≈ coef(whole)
end

@testset "GaussianNB discovers classes and trailing features" begin
    model = GaussianNB()
    fit_batch!(model, [([0.0], 0), ([1.0], 1), ([2.0], 2)])
    @test Set(classes(model)) == Set((0, 1, 2))

    fit!(model, ([0.0, 1.0], 0))
    @test length(value(model)[0]) == 2

    probabilities = OnlineML.predict_proba(model, [0.5, 1.0])
    @test Set(keys(probabilities)) == Set((0, 1, 2))
    @test sum(values(probabilities)) ≈ 1.0
end

@testset "LogisticRegression is order-sensitive" begin
    observations = [
        ([0.0], 1),
        ([1.0], 1),
        ([3.0], 0),
    ]
    forward = LogisticRegression()
    reverse_order = LogisticRegression()

    fit_batch!(forward, observations)
    fit_batch!(reverse_order, reverse(observations))

    @test value(forward) != value(reverse_order)
end
