"""
    Instance

Instance-based online learning models.

# Exports
- `KNN` - K-Nearest Neighbors with sliding window
"""
module Instance

using OnlineStatsBase
using NearestNeighbors
using Distances
using ..OnlineML: Learner, dict_argmax
import ..OnlineML: predict, predict_proba, reset!

include("knn.jl")

export KNN

end # module Instance
