module OnlineML

using OnlineStatsBase
using OnlineStats
using Optimisers
using NearestNeighbors
using Distances
using Tables
using StatsBase
using Random
using DataStructures

# Core types and interfaces
include("types.jl")

# Submodules
include("Optim/Optim.jl")
include("Linear/Linear.jl")
include("Drift/Drift.jl")  # Must come before Trees (HAT uses ADWIN)
include("Trees/Trees.jl")
include("Instance/Instance.jl")
include("Bayes/Bayes.jl")
include("Cluster/Cluster.jl")
include("Anomaly/Anomaly.jl")
include("Transform/Transform.jl")
include("Streams/Streams.jl")  # Must come before Metrics for evaluation functions
include("Metrics/Metrics.jl")
include("Pipeline/Pipeline.jl")
include("Ensemble/Ensemble.jl")

# Re-export OnlineStatsBase interface (reset! is our own)
export fit!, value, nobs, merge!
export reset!

# Re-export core types
export Learner, UnsupervisedLearner, Transformer, Detector, Metric
export DriftStatus, NoDrift, Warning, DriftDetected

# Re-export interface methods
export predict, predict_proba, transform, fit_transform!, inverse_transform
export fit_predict!, score, score_one, is_anomaly, fit_score!
export fit_batch!
export learnapi
export update!, status, detected_drift, detected_warning
export dict_argmax, generate

# Submodules (for convenience access via using OnlineML.Linear, etc.)
export Optim, Linear, Trees, Instance, Bayes, Cluster, Anomaly
export Transform, Drift, Metrics, Streams, Pipeline, Ensemble

"""
    learnapi(learner)

Return a LearnAPI-compatible learner configuration for a supported OnlineML
learner. Methods are provided by the LearnAPI package extension.
"""
function learnapi end

end # module OnlineML
