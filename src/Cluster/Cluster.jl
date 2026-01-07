"""
    Cluster

Online clustering algorithms.

# Exports
- `StreamingKMeans` - Streaming K-Means clustering
"""
module Cluster

using OnlineStatsBase
using OnlineStats: Mean
using ..OnlineML: UnsupervisedLearner
import ..OnlineML: predict, reset!

include("kmeans.jl")

export StreamingKMeans, centroids, cluster_sizes

end # module Cluster
