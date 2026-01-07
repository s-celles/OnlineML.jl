"""
    Anomaly

Online anomaly detection algorithms.

# Exports
- `HalfSpaceTrees` - Half-Space Trees for streaming anomaly detection
- `LODA` - Lightweight Online Detector of Anomalies
- `RobustRandomCutForest` - RRCF for streaming anomaly detection
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

export HalfSpaceTrees, LODA, RobustRandomCutForest

end # module Anomaly
