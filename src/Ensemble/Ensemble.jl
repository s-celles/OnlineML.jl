"""
    Ensemble

Online ensemble methods for machine learning.

# Exports
- `Bagging` - Online bagging with Poisson resampling
- `LeveragingBagging` - Enhanced bagging with higher λ
- `AdaptiveRandomForest` - Drift-aware random forest
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

export Bagging, LeveragingBagging
export AdaptiveRandomForest, ARFEstimator
export total_drifts, per_tree_drifts

end # module Ensemble
