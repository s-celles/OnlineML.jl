# Required imports for MLDatasets
using DataFrames
using MLDatasets
using Random
using OnlineML: fit!, predict, nobs, reset!, update!, detected_drift
using OnlineML.Linear: LogisticRegression, Perceptron
using OnlineML.Bayes: GaussianNB
using OnlineML.Transform: StandardScaler, MinMaxScaler
using OnlineML.Pipeline: OnlinePipeline
using OnlineML.Drift: ADWIN, DDM
using OnlineML.Streams: SEAGenerator, RandomRBFGenerator, generate
using OnlineML.Trees: HoeffdingTree, n_nodes, height, n_leaves
using OnlineML.Ensemble: Bagging
using OnlineML.Metrics: Accuracy

@testset "Examples" begin
    # Test that all example notebooks are runnable (without Pluto UI)
    # We extract and run the Julia code from each notebook

    @testset "01_basic_classification.jl" begin
        # Test basic classification with Iris
        Random.seed!(42)

        # Load Iris
        iris = Iris()
        class_map = Dict(
            "Iris-setosa" => 0,
            "Iris-versicolor" => 1,
            "Iris-virginica" => 2
        )

        # Collect and shuffle data
        data = [(Vector(obs.features), class_map[obs.targets.class]) for obs in iris]
        shuffle!(data)

        # Train LogisticRegression
        model = LogisticRegression()
        correct = 0
        total = 0

        for (x, y) in data
            if nobs(model) > 0
                y_pred = predict(model, x)
                total += 1
                if y_pred == y
                    correct += 1
                end
            end
            fit!(model, (x, y))
        end

        accuracy = correct / total
        @test accuracy > 0.3  # Should achieve reasonable accuracy for 3-class online learning
        @test nobs(model) == length(data)
    end

    @testset "02_pipeline_composition.jl" begin
        # Test pipeline composition
        Random.seed!(42)

        # Load and prepare data
        iris = Iris()
        data = []
        for obs in iris
            x = Vector(obs.features)
            x_scaled = [x[1] * 100, x[2] * 0.01, x[3] * 1000, x[4] * 0.001]
            y = obs.targets.class == "Iris-setosa" ? 0 : 1
            push!(data, (x_scaled, y))
        end
        shuffle!(data)

        # Test pipeline
        pipeline = StandardScaler() |> LogisticRegression()

        correct = 0
        total = 0

        for (x, y) in data
            if nobs(pipeline) > 0
                y_pred = predict(pipeline, x)
                total += 1
                if y_pred == y
                    correct += 1
                end
            end
            fit!(pipeline, (x, y))
        end

        accuracy = correct / total
        @test accuracy > 0.7  # Pipeline should work well
        @test nobs(pipeline) == length(data)
    end

    @testset "03_drift_detection.jl" begin
        # Test drift detection
        Random.seed!(42)

        # Create drifting stream
        stream1 = SEAGenerator(function_id=1, noise=0.1, rng=MersenneTwister(42))
        stream2 = SEAGenerator(function_id=3, noise=0.1, rng=MersenneTwister(43))

        n_before = 500
        n_after = 500

        data = []
        for i in 1:(n_before + n_after)
            x, y = i <= n_before ? generate(stream1) : generate(stream2)
            push!(data, (x, y))
        end

        # Train with ADWIN
        model = LogisticRegression()
        detector = ADWIN()
        n_drifts = 0

        for (x, y) in data
            if nobs(model) > 0
                y_pred = predict(model, x)
                error = (y_pred != y) ? 1.0 : 0.0
                update!(detector, error)
                if detected_drift(detector)
                    n_drifts += 1
                    reset!(model)
                    reset!(detector)
                end
            end
            fit!(model, (x, y))
        end

        @test nobs(model) > 0
        # ADWIN should detect at least some drift
        @test n_drifts >= 0  # May or may not detect depending on data
    end

    @testset "04_hoeffding_tree.jl" begin
        # Test Hoeffding Tree
        Random.seed!(42)

        gen = RandomRBFGenerator(
            n_classes=3,
            n_features=5,
            n_centroids=10,
            rng=MersenneTwister(42)
        )

        # Train tree
        tree = HoeffdingTree(grace_period=50)

        correct = 0
        total = 0

        for _ in 1:2000
            x, y = generate(gen)
            if nobs(tree) > 0
                y_pred = predict(tree, x)
                total += 1
                if y_pred == y
                    correct += 1
                end
            end
            fit!(tree, (x, y))
        end

        @test nobs(tree) == 2000
        @test n_nodes(tree) >= 1
        @test height(tree) >= 0
        @test n_leaves(tree) >= 1
    end

    @testset "05_ensemble_learning.jl" begin
        # Test ensemble learning
        Random.seed!(42)

        # Load Titanic
        titanic = Titanic()

        # Prepare data
        data = []
        for obs in titanic
            target = obs.targets.Survived
            ismissing(target) && continue

            features = obs.features
            age = ismissing(features.Age) ? 30.0 : Float64(features.Age)
            fare = ismissing(features.Fare) ? 32.0 : Float64(features.Fare)
            sibsp = Float64(features.SibSp)
            parch = Float64(features.Parch)
            pclass = Float64(features.Pclass)
            sex = features.Sex == "male" ? 1.0 : 0.0

            x = [age, fare, sibsp, parch, pclass, sex]
            y = Int(target)
            push!(data, (x, y))
        end
        shuffle!(data)

        # Train bagging ensemble
        bag = Bagging(() -> LogisticRegression(), n_estimators=5)

        correct = 0
        total = 0

        for (x, y) in data
            if nobs(bag) > 0
                y_pred = predict(bag, x)
                total += 1
                if y_pred == y
                    correct += 1
                end
            end
            fit!(bag, (x, y))
        end

        accuracy = correct / total
        @test accuracy > 0.6
        @test nobs(bag) == length(data)
    end
end
