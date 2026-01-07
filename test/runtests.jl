using OnlineML
using Test
using Documenter

@testset "OnlineML.jl" begin
    # Unit tests - Core types
    @testset "Core Types" begin
        include("unit/test_types.jl")
    end

    # Unit tests - Learners
    @testset "Linear Models" begin
        include("unit/test_linear.jl")
    end

    @testset "Naive Bayes" begin
        include("unit/test_bayes.jl")
    end

    @testset "Decision Trees" begin
        include("unit/test_trees.jl")
    end

    @testset "Instance-Based" begin
        include("unit/test_instance.jl")
    end

    @testset "Metrics" begin
        include("unit/test_metrics.jl")
    end

    @testset "Anomaly Detection" begin
        include("unit/test_anomaly.jl")
    end

    @testset "Transform" begin
        include("unit/test_transform.jl")
    end

    @testset "Pipeline" begin
        include("unit/test_pipeline.jl")
    end

    @testset "Drift Detection" begin
        include("unit/test_drift.jl")
    end

    @testset "Generators" begin
        include("unit/test_generators.jl")
    end

    @testset "Evaluation" begin
        include("unit/test_evaluation.jl")
    end

    @testset "Streams" begin
        include("unit/test_streams.jl")
    end

    # Contract tests (wrapper equivalence)
    @testset "Contract: Linear" begin
        include("contract/test_linear.jl")
    end

    @testset "Contract: Cluster" begin
        include("contract/test_cluster.jl")
    end

    @testset "Contract: Transform" begin
        include("contract/test_transform.jl")
    end

    # Integration tests
    @testset "Integration: Pipeline" begin
        include("integration/test_pipeline.jl")
    end

    @testset "Integration: Ensemble" begin
        include("integration/test_ensemble.jl")
    end

    @testset "Integration: Evaluation" begin
        include("integration/test_evaluation.jl")
    end

    @testset "Integration: Examples" begin
        include("integration/test_examples.jl")
    end
    # include("integration/test_mlj.jl")

    # NamedTuple input tests (Feature 002)
    @testset "NamedTuple Input" begin
        using OnlineML.Linear: LogisticRegression, Perceptron
        using OnlineML.Cluster: StreamingKMeans
        using Tables
        include("namedtuple_test.jl")
    end

    # Code quality checks (Aqua.jl)
    @testset "Code Quality" begin
        include("Aqua.jl")
    end

    # Doctests - verify all docstring examples work correctly
    @testset "Doctests" begin
        DocMeta.setdocmeta!(OnlineML, :DocTestSetup, quote
            using OnlineML
            using OnlineML: fit!, predict, nobs, transform, fit_transform!, reset!, fit_predict!, iterate_table!, value, score, fit_score!
            using OnlineML.Linear: LogisticRegression, Perceptron, Regression, PassiveAggressive
            using OnlineML.Cluster: StreamingKMeans, centroids, cluster_sizes
            using OnlineML.Transform: StandardScaler, MinMaxScaler, MaxAbsScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder, MeanImputer, ModeImputer
            using OnlineML.Trees: HoeffdingTree, ExtremelyFastTree, HoeffdingAdaptiveTree
            using OnlineML.Bayes: GaussianNB, MultinomialNB, BernoulliNB
            using OnlineML.Instance: KNN
            using OnlineML.Metrics: Accuracy, Precision, Recall, F1Score, MAE, MSE, RMSE, R2
            using OnlineML.Drift: ADWIN, DDM, EDDM, PageHinkley, KSWIN, status
            using OnlineML.Anomaly: HalfSpaceTrees, RobustRandomCutForest
            using OnlineML.Ensemble: AdaptiveRandomForest, Bagging, LeveragingBagging
            using OnlineML.Pipeline: OnlinePipeline
            using OnlineML.Streams: SEAGenerator, AgrawalGenerator, SineGenerator, RandomRBFGenerator, LEDGenerator, HyperplaneGenerator, ConceptDriftStream, DataStream, batch_stream, take_stream, skip_stream, generate
            using OnlineML.Optim: Adam, Descent, Momentum
            using Random
            using DataFrames
        end; recursive=true)
        Documenter.doctest(OnlineML)
    end
end
