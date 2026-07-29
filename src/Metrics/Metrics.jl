"""
    Metrics

Online performance metrics for evaluation.

# Exports
- `Accuracy` - Online accuracy
- `Precision` - Online precision
- `Recall` - Online recall
- `F1Score` - Online F1 score
- `MAE` - Mean Absolute Error
- `MSE` - Mean Squared Error
- `RMSE` - Root Mean Squared Error
- `R2` - R² coefficient of determination
- `RollingMetric` - Rolling window metric
- `progressive_val_score` - Prequential evaluation
- `holdout_score` - Holdout evaluation
"""
module Metrics

using OnlineStatsBase
using OnlineStats: Mean
using ..OnlineML: Metric, predict, nobs, fit!
using ..OnlineML.Streams: generate
import ..OnlineML: reset!

include("classification.jl")
include("regression.jl")
include("rolling.jl")
include("evaluation.jl")

export Accuracy, Precision, Recall, F1Score
export MAE, MSE, RMSE, R2
export RollingMetric
export progressive_val_score, holdout_score, interleaved_test_then_train

end # module Metrics
