"""
    Pipeline

Composable pipelines for online machine learning.

# Exports
- `Pipeline` - Sequential composition of transformers and learner
- `FeatureUnion` - Parallel feature concatenation with + operator
- `ColumnTransformer` - Apply different transformers to different columns
- `|>` operator overload for building pipelines
"""
module Pipeline

using OnlineStatsBase
using ..OnlineML: Learner, Transformer
import ..OnlineML: transform, predict, predict_proba, reset!

include("online_pipeline.jl")
include("feature_union.jl")
include("column_transformer.jl")

export OnlinePipeline, FeatureUnion, ColumnTransformer

end # module Pipeline
