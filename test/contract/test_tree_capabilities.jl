using OnlineML
using OnlineML.Trees:
    ExtremelyFastTree,
    HoeffdingAdaptiveTree,
    HoeffdingTree,
    n_leaves,
    n_nodes

mutable struct TreeOneShot{T}
    data::T
    started::Bool
end

function Base.iterate(stream::TreeOneShot)
    stream.started && error("stream was traversed more than once")
    stream.started = true
    return iterate(stream.data)
end

Base.iterate(stream::TreeOneShot, state) = iterate(stream.data, state)

const TREE_FACTORIES = (
    () -> HoeffdingTree(grace_period=4, max_depth=2),
    () -> ExtremelyFastTree(grace_period=4, max_depth=2),
    () -> HoeffdingAdaptiveTree(grace_period=4, max_depth=2),
)

@testset "experimental tree lifecycle" begin
    first_batch = [([0.0], 0), ([1.0], 1)]
    second_batch = [([0.2, 1.0], 2), ([0.8, 1.0], 1)]

    for factory in TREE_FACTORIES
        tree = factory()
        @test fit_batch!(tree, TreeOneShot(first_batch, false)) === tree
        @test nobs(tree) == 2

        structure_before = (nobs(tree), n_nodes(tree), n_leaves(tree))
        OnlineML.predict(tree, [0.5])
        @test (nobs(tree), n_nodes(tree), n_leaves(tree)) == structure_before

        @test fit_batch!(tree, second_batch) === tree
        @test nobs(tree) == 4

        probabilities = OnlineML.predict_proba(tree, [0.5, 1.0])
        @test Set(keys(probabilities)) == Set((0, 1, 2))
        @test sum(values(probabilities)) ≈ 1.0

        @test reset!(tree) === tree
        @test (nobs(tree), n_nodes(tree), n_leaves(tree)) == (0, 1, 1)
    end
end
