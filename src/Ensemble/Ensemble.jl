"""
    Ensemble

Online ensemble methods for machine learning.

# Exports
- `Bagging` - Online bagging with Poisson resampling
- `HighRatePoissonBagging` - Bagging with a configurable Poisson rate
- `DriftAwareBagging` - Poisson bagging with per-learner drift adaptation
"""
module Ensemble

using OnlineStatsBase
using Random
using ..OnlineML: Learner, dict_argmax
import ..OnlineML: predict, predict_proba, reset!
import ..OnlineML.Drift: detected_drift, detected_warning

include("bagging.jl")
include("leveraging.jl")
include("arf.jl")

export Bagging, HighRatePoissonBagging, LeveragingBagging
export DriftAwareBagging, DriftAwareEstimator
export AdaptiveRandomForest, ARFEstimator
export total_drifts, per_learner_drifts, per_tree_drifts

end # module Ensemble
