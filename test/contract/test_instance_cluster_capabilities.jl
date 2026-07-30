using OnlineML
using OnlineML.Cluster: StreamingKMeans, centroids, cluster_sizes
using OnlineML.Instance: KNN

mutable struct CapabilityOneShot{T}
    data::T
    started::Bool
end

function Base.iterate(stream::CapabilityOneShot)
    stream.started && error("stream was traversed more than once")
    stream.started = true
    return iterate(stream.data)
end

Base.iterate(stream::CapabilityOneShot, state) = iterate(stream.data, state)

@testset "StreamingKMeans incremental lifecycle" begin
    model = StreamingKMeans(k=2, learning_rate=0.5)
    first_batch = [[0.0, 0.0], [10.0, 10.0]]
    second_batch = [[2.0, 2.0], [8.0, 8.0]]

    @test fit_batch!(model, CapabilityOneShot(first_batch, false)) === model
    @test nobs(model) == 2

    state_before = deepcopy(value(model))
    OnlineML.predict(model, [1.0, 1.0])
    @test value(model) == state_before

    @test fit_batch!(model, second_batch) === model
    @test nobs(model) == 4
    @test length(centroids(model)) == 2
    @test sum(cluster_sizes(model)) == 4

    @test reset!(model) === model
    @test nobs(model) == 0
    @test isempty(centroids(model))
end

@testset "KNN incremental lifecycle and bounded window" begin
    model = KNN(k=2, window_size=3, weighted=false)
    first_batch = [([0.0], 0), ([1.0], 1)]
    second_batch = [([2.0], 2), ([3.0], 3)]

    @test fit_batch!(model, CapabilityOneShot(first_batch, false)) === model
    @test nobs(model) == 2

    state_before = deepcopy(value(model))
    OnlineML.predict(model, [0.5])
    @test value(model) == state_before

    @test fit_batch!(model, second_batch) === model
    @test nobs(model) == 4
    @test value(model).X == [[1.0], [2.0], [3.0]]
    @test value(model).y == [1, 2, 3]

    probabilities = OnlineML.predict_proba(model, [2.5])
    @test Set(keys(probabilities)) == Set((2, 3))
    @test sum(values(probabilities)) ≈ 1.0

    @test reset!(model) === model
    @test nobs(model) == 0
    @test isempty(value(model).X)
    @test isempty(value(model).y)
end

@testset "constructor and schema validation" begin
    @test_throws ArgumentError StreamingKMeans(k=0)
    @test_throws ArgumentError StreamingKMeans(learning_rate=0)
    @test_throws ArgumentError StreamingKMeans(learning_rate=1.1)
    @test_throws ArgumentError KNN(k=0)
    @test_throws ArgumentError KNN(window_size=0)

    kmeans = StreamingKMeans(k=2)
    fit!(kmeans, [0.0, 1.0])
    @test_throws DimensionMismatch fit!(kmeans, [0.0])
    @test_throws DimensionMismatch OnlineML.predict(kmeans, [0.0, 1.0, 2.0])

    knn = KNN(k=1, window_size=2)
    fit!(knn, ([0.0, 1.0], 0))
    @test_throws DimensionMismatch fit!(knn, ([0.0], 1))
    @test_throws DimensionMismatch OnlineML.predict(knn, [0.0, 1.0, 2.0])
    @test_throws DimensionMismatch OnlineML.predict_proba(knn, [0.0])
end

@testset "order-sensitive state" begin
    observations = [[0.0], [10.0], [2.0], [8.0]]
    forward = StreamingKMeans(k=2, learning_rate=0.5)
    reverse_order = StreamingKMeans(k=2, learning_rate=0.5)
    fit_batch!(forward, observations)
    fit_batch!(reverse_order, reverse(observations))
    @test value(forward) != value(reverse_order)

    labeled = [([0.0], 0), ([1.0], 1), ([2.0], 2), ([3.0], 3)]
    forward_knn = KNN(k=1, window_size=2)
    reverse_knn = KNN(k=1, window_size=2)
    fit_batch!(forward_knn, labeled)
    fit_batch!(reverse_knn, reverse(labeled))
    @test value(forward_knn) != value(reverse_knn)
end
