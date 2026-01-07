"""
    Linear

Linear online learning models.

# Exports
- `Regression` - Online linear regression (wraps OnlineStats.LinReg)
- `LogisticRegression` - Online logistic regression with gradient optimization
- `Perceptron` - Perceptron classifier
- `PassiveAggressive` - Passive-Aggressive classifier (PA-I, PA-II)
"""
module Linear

using OnlineStatsBase
using OnlineStats
using OnlineStats: LinReg
using Optimisers
using ..OnlineML: Learner
import ..OnlineML: predict, predict_proba, reset!

include("regression.jl")
include("logistic.jl")
include("perceptron.jl")
include("passive_aggressive.jl")

export Regression, LogisticRegression, Perceptron, PassiveAggressive

end # module Linear
