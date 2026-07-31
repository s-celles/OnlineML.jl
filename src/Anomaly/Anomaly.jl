"""
    Anomaly

Online anomaly detection algorithms.

# Exports
- `HalfSpaceTrees` - Half-Space Trees for streaming anomaly detection
- `GaussianProjectionDetector` - Gaussian random-projection detector
- `LODA` - Compatibility alias for `GaussianProjectionDetector`
- `RandomCutForestApproximation` - Experimental random-cut approximation
- `RobustRandomCutForest` - Compatibility alias
"""
module Anomaly

using OnlineStatsBase
using OnlineStats
using Random
using DataStructures
using ..OnlineML: UnsupervisedLearner
import ..OnlineML: score, score_one, reset!, fit_score!, is_anomaly

include("half_space_trees.jl")
include("loda.jl")
include("cut_tree.jl")
include("rrcf.jl")

export HalfSpaceTrees, GaussianProjectionDetector, LODA
export RandomCutForestApproximation, RobustRandomCutForest

end # module Anomaly
