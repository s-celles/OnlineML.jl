using LearnAPI
using OnlineML.Bayes: GaussianNB

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
    learner = learnapi(GaussianNB())
    @test LearnAPI.clone(learner) == learner
    @test LearnAPI.kinds_of_proxy(learner) == (LearnAPI.Point(),)
    @test :(LearnAPI.update_observations) in LearnAPI.functions(learner)

    first_batch = LearnAPIOneShot([([0.0, 0.0], 0), ([1.0, 1.0], 1)], false)
    model = LearnAPI.fit(learner, first_batch)
    state = model.state
    @test first_batch.started
    @test OnlineML.nobs(state) == 2
    @test LearnAPI.learner(model) == learner
    @test LearnAPI.predict(model, LearnAPI.Point(), [[0.0, 0.0], [1.0, 1.0]]) == [0, 1]

    second_batch = LearnAPIOneShot([([0.1, 0.2], 0), ([0.9, 0.8], 1)], false)
    returned = LearnAPI.update_observations(model, second_batch)
    @test returned === model
    @test returned.state === state
    @test second_batch.started
    @test OnlineML.nobs(returned.state) == 4
end

@testset "LearnAPI adapter rejects fitted configurations" begin
    core = GaussianNB()
    OnlineML.fit!(core, ([0.0], 0))
    @test_throws ArgumentError learnapi(core)
end
