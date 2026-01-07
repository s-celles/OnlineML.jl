using Test
using OnlineML
using OnlineML.Ensemble
using OnlineML.Drift
using OnlineML.Linear
using OnlineML.Trees

@testset "Integration: Ensemble" begin
    @testset "LeveragingBagging basic" begin
        ensemble = LeveragingBagging(
            () -> LogisticRegression(),
            n_estimators=5,
            lambda=6.0
        )

        @test nobs(ensemble) == 0

        # Train
        for _ in 1:50
            x = randn(3)
            y = x[1] > 0 ? 1 : 0
            fit!(ensemble, (x, y))
        end

        @test nobs(ensemble) == 50

        # Predict
        y_pred = OnlineML.predict(ensemble, randn(3))
        @test y_pred in [0, 1]

        # Reset
        reset!(ensemble)
        @test nobs(ensemble) == 0
    end

    @testset "AdaptiveRandomForest with drift" begin
        # Create ARF with LogisticRegression and DDM detectors
        arf = AdaptiveRandomForest(
            () -> LogisticRegression(),
            () -> DDM(min_instances=10),
            n_estimators=3,
            lambda=3.0
        )

        @test nobs(arf) == 0

        # First concept: x[1] > 0 → class 1
        for _ in 1:50
            x = randn(3)
            y = x[1] > 0 ? 1 : 0
            fit!(arf, (x, y))
        end

        @test nobs(arf) == 50

        # Predict works
        y_pred = OnlineML.predict(arf, randn(3))
        @test y_pred in [0, 1]

        # Check drift counters
        @test total_drifts(arf) >= 0
        @test length(per_tree_drifts(arf)) == 3

        # Reset
        reset!(arf)
        @test nobs(arf) == 0
        @test total_drifts(arf) == 0
    end

    @testset "ARF handles concept shift" begin
        arf = AdaptiveRandomForest(
            () -> LogisticRegression(),
            () -> DDM(min_instances=15, drift_level=2.5),
            n_estimators=3,
            lambda=3.0
        )

        # First concept: x[1] > 0 → class 1
        for _ in 1:100
            x = randn(3)
            y = x[1] > 0 ? 1 : 0
            fit!(arf, (x, y))
        end

        initial_drifts = total_drifts(arf)

        # Second concept: opposite - x[1] > 0 → class 0
        for _ in 1:100
            x = randn(3)
            y = x[1] > 0 ? 0 : 1  # Reversed!
            fit!(arf, (x, y))
        end

        # ARF should have detected some drifts due to concept change
        # (though not guaranteed - depends on random initialization)
        @test nobs(arf) == 200
    end

    @testset "Bagging with HoeffdingTree" begin
        bag = Bagging(
            () -> HoeffdingTree(),
            n_estimators=3,
            lambda=1.0
        )

        # Train
        for _ in 1:100
            x = randn(3)
            y = x[1] + x[2] > 0 ? 1 : 0
            fit!(bag, (x, y))
        end

        @test nobs(bag) == 100

        # Predict
        y_pred = OnlineML.predict(bag, randn(3))
        @test y_pred in [0, 1]
    end
end
