using Documenter
using OnlineML

# Configure DocTestSetup - automatically inject imports before every doctest
DocMeta.setdocmeta!(OnlineML, :DocTestSetup, quote
    using OnlineML
    using OnlineML: fit!, predict, nobs, transform, fit_transform!, reset!, fit_predict!, iterate_table!
    using OnlineML.Linear: LogisticRegression, Perceptron, Regression, PassiveAggressive
    using OnlineML.Cluster: StreamingKMeans, centroids, cluster_sizes
    using OnlineML.Transform: StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, TargetEncoder, MeanImputer, ModeImputer
    using OnlineML.Trees: HoeffdingTree
    using OnlineML.Bayes: GaussianNB, MultinomialNB, BernoulliNB
    using OnlineML.Instance: KNN
    using OnlineML.Metrics: Accuracy, Precision, Recall, F1Score, MAE, MSE, RMSE, R2
    using OnlineML.Drift: ADWIN, DDM, EDDM, KSWIN, PageHinkley, status
    using OnlineML.Anomaly: HalfSpaceTrees
    using OnlineML.Ensemble: AdaptiveRandomForest, Bagging, DriftAwareBagging, LeveragingBagging
    using OnlineML.Pipeline: OnlinePipeline
    using OnlineML.Streams: SEAGenerator, AgrawalGenerator, SineGenerator, RandomRBFGenerator, LEDGenerator, HyperplaneGenerator, ConceptDriftStream, DataStream, batch_stream, take_stream, skip_stream, generate
    using OnlineML.Optim: Adam, Descent, Momentum
    using Random
    using DataFrames
end; recursive=true)

makedocs(
    sitename = "OnlineML.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        repolink = "https://github.com/s-celles/OnlineML.jl",
    ),
    remotes = nothing,
    warnonly = true,
    modules = [OnlineML],
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Incremental Contract" => "incremental_contract.md",
        "Model Capabilities" => "model_capabilities.md",
        "Migration from River" => "migration_from_river.md",
        "AI Transparency" => "ai_transparency.md",
        "API Reference" => [
            "api/linear.md",
            "api/trees.md",
            "api/ensemble.md",
            "api/instance.md",
            "api/bayes.md",
            "api/cluster.md",
            "api/anomaly.md",
            "api/transform.md",
            "api/drift.md",
            "api/metrics.md",
            "api/streams.md",
            "api/pipeline.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/s-celles/OnlineML.jl.git",
)
