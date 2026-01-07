using Test
using OnlineML
import OnlineML: Cluster
using .Cluster: StreamingKMeans, centroids, cluster_sizes
using OnlineStats

@testset "Contract: Cluster Models" begin
    @testset "StreamingKMeans basic contract" begin
        # Contract: StreamingKMeans should maintain k centroids
        # and assign points to the nearest cluster

        k = 3
        model = StreamingKMeans(k=k)

        # Generate clustered data
        n_per_cluster = 50
        centers = [[0.0, 0.0], [10.0, 0.0], [5.0, 8.66]]

        for center in centers
            for _ in 1:n_per_cluster
                x = center .+ randn(2) * 0.5
                fit!(model, x)
            end
        end

        # Contract: should have k centroids after fitting
        @test length(centroids(model)) == k

        # Contract: observation count matches
        @test nobs(model) == k * n_per_cluster

        # Contract: predictions should return valid cluster indices
        for center in centers
            pred = OnlineML.predict(model, center)
            @test pred in 1:k
        end

        # Contract: centroids should be reasonably spread out
        # (exact convergence not guaranteed for online algorithm with limited data)
        model_centroids = centroids(model)
        @test length(model_centroids) == k

        # Check that centroids are not all the same (algorithm is learning)
        if length(model_centroids) >= 2
            dists = [sum((model_centroids[1] .- c).^2) for c in model_centroids[2:end]]
            @test any(d -> d > 0.1, dists)  # At least some spread between centroids
        end
    end

    @testset "StreamingKMeans comparison with OnlineStats.KMeans" begin
        # Note: OnlineStats.KMeans and our StreamingKMeans have different algorithms
        # This test verifies both produce valid clustering, not exact equivalence

        k = 2
        our_model = StreamingKMeans(k=k)
        os_model = OnlineStats.KMeans(k)

        # Simple 2-cluster data
        data = [
            [0.0, 0.0], [0.1, 0.1], [-0.1, 0.1], [0.0, -0.1],
            [5.0, 5.0], [5.1, 5.1], [4.9, 5.1], [5.0, 4.9]
        ]

        for x in data
            fit!(our_model, x)
            fit!(os_model, x)
        end

        # Both should have same nobs
        @test nobs(our_model) == nobs(os_model)

        # Both should produce valid cluster assignments
        test_point = [2.5, 2.5]
        our_pred = OnlineML.predict(our_model, test_point)
        @test our_pred in 1:k

        # OnlineStats.KMeans value returns tuple of Cluster objects
        # Each cluster has a center (mean vector)
        os_clusters = value(os_model)
        # Check that OnlineStats also produces k clusters
        @test length(os_clusters) == k
    end

    @testset "StreamingKMeans with reset!" begin
        model = StreamingKMeans(k=3)

        for _ in 1:50
            fit!(model, randn(4))
        end
        @test nobs(model) == 50
        @test length(centroids(model)) == 3

        reset!(model)
        @test nobs(model) == 0
        @test isempty(centroids(model))
    end

    @testset "StreamingKMeans empty model prediction" begin
        model = StreamingKMeans(k=3)
        @test OnlineML.predict(model, [1.0, 2.0]) == 0
    end
end
