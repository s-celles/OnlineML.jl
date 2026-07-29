using LearnAPI
using OnlineML.Bayes: GaussianNB
using OnlineML.Linear: LogisticRegression, Regression
using OnlineML.Optim: Descent

mutable struct LearnAPIOneShot{T}
    data::T
    started::Bool
end

function Base.iterate(stream::LearnAPIOneShot)
    stream.started && error("stream was traversed more than once")
    stream.started = true
    result = iterate(stream.data)
    result === nothing && return nothing
    value, state = result
    return value, state
end

Base.iterate(stream::LearnAPIOneShot, state) = iterate(stream.data, state)

@testset "LearnAPI GaussianNB incremental lifecycle" begin
    learner = learnapi(GaussianNB{Float32}())
    @test LearnAPI.clone(learner) == learner
    @test LearnAPI.kinds_of_proxy(learner) == (LearnAPI.Point(),)
    @test :(LearnAPI.update_observations) in LearnAPI.functions(learner)

    first_batch = LearnAPIOneShot([
        (Float32[0, 0], 0),
        (Float32[1, 1], 1),
    ], false)
    model = LearnAPI.fit(learner, first_batch)
    state = model.state
    @test first_batch.started
    @test OnlineML.nobs(state) == 2
    @test state isa GaussianNB{Float32}
    @test LearnAPI.learner(model) == learner
    @test LearnAPI.predict(
        model,
        LearnAPI.Point(),
        [Float32[0, 0], Float32[1, 1]],
    ) == [0, 1]

    second_batch = LearnAPIOneShot([
        (Float32[0.1, 0.2], 0),
        (Float32[0.9, 0.8], 1),
    ], false)
    returned = LearnAPI.update_observations(model, second_batch)
    @test returned === model
    @test returned.state === state
    @test second_batch.started
    @test OnlineML.nobs(returned.state) == 4
end

@testset "LearnAPI Regression incremental lifecycle" begin
    learner = learnapi(Regression{Float32}())
    first_batch = LearnAPIOneShot([
        (Float32[0], 1.0f0),
        (Float32[1], 3.0f0),
    ], false)
    model = LearnAPI.fit(learner, first_batch)
    state = model.state

    @test first_batch.started
    @test state isa Regression{Float32}
    @test OnlineML.nobs(state) == 2
    @test LearnAPI.predict(model, LearnAPI.Point(), [Float32[2]]) ≈ [5.0]

    second_batch = LearnAPIOneShot([
        (Float32[2], 5.0f0),
        (Float32[3], 7.0f0),
    ], false)
    returned = LearnAPI.update_observations(model, second_batch)

    @test returned === model
    @test returned.state === state
    @test second_batch.started
    @test OnlineML.nobs(state) == 4
    @test LearnAPI.predict(model, LearnAPI.Point(), [Float32[4]]) ≈ [9.0]
end

@testset "LearnAPI Regression rejects fitted configurations" begin
    core = Regression()
    OnlineML.fit!(core, ([0.0], 1.0))
    @test_throws ArgumentError learnapi(core)
end

@testset "LearnAPI LogisticRegression preserves configuration and state" begin
    core = LogisticRegression{Float32}(;
        optimizer=Descent(0.05),
        l1=0.01,
        l2=0.02,
    )
    learner = learnapi(core)
    cloned = LearnAPI.clone(learner; l2=0.03)

    @test learner.optimizer == core.optimizer
    @test learner.l1 === 0.01f0
    @test learner.l2 === 0.02f0
    @test cloned.l1 === learner.l1
    @test cloned.l2 === 0.03f0

    first_observations = [
        (Float32[-1, -1], 0),
        (Float32[1, 1], 1),
    ]
    first_batch = LearnAPIOneShot(first_observations, false)
    model = LearnAPI.fit(learner, first_batch)
    state = model.state

    @test first_batch.started
    @test state isa LogisticRegression{Float32}
    @test state.optimizer == learner.optimizer
    @test state.l1 === learner.l1
    @test state.l2 === learner.l2
    @test OnlineML.nobs(state) == 2

    second_observations = [
        (Float32[-2, -1], 0),
        (Float32[2, 1], 1),
    ]
    second_batch = LearnAPIOneShot(second_observations, false)
    returned = LearnAPI.update_observations(model, second_batch)

    expected = LogisticRegression{Float32}(;
        optimizer=Descent(0.05),
        l1=0.01,
        l2=0.02,
    )
    OnlineML.fit_batch!(expected, Iterators.flatten((
        first_observations,
        second_observations,
    )))

    @test returned === model
    @test returned.state === state
    @test second_batch.started
    @test OnlineML.nobs(state) == 4
    @test state.weights ≈ expected.weights
    @test state.bias ≈ expected.bias
    @test LearnAPI.predict(
        model,
        LearnAPI.Point(),
        [Float32[-1, -1], Float32[1, 1]],
    ) == [0, 1]
end

@testset "LearnAPI LogisticRegression rejects fitted configurations" begin
    core = LogisticRegression()
    OnlineML.fit!(core, ([0.0], 0))
    @test_throws ArgumentError learnapi(core)
end

@testset "LearnAPI adapter rejects fitted configurations" begin
    core = GaussianNB()
    OnlineML.fit!(core, ([0.0], 0))
    @test_throws ArgumentError learnapi(core)
end
