"""
    Trees

Online decision tree models.

# Exports
- `HoeffdingTree` - Streaming decision tree using VFDT algorithm
- `ExtremelyFastTree` - EFDT with split re-evaluation
- `HoeffdingAdaptiveTree` - Drift-aware decision tree with ADWIN
"""
module Trees

using OnlineStatsBase
using OnlineStats: Variance, CountMap, mean, var
using ..OnlineML: Learner, dict_argmax
import ..OnlineML: predict, predict_proba, reset!

include("nodes.jl")
include("hoeffding_tree.jl")
include("efdt_nodes.jl")
include("extremely_fast_tree.jl")
include("adaptive_node.jl")
include("hoeffding_adaptive_tree.jl")

export HoeffdingTree, ExtremelyFastTree, HoeffdingAdaptiveTree
export n_nodes, n_leaves, height, n_replacements

end # module Trees
