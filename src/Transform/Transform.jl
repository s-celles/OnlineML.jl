"""
    Transform

Online feature transformation and preprocessing.

# Exports
- `StandardScaler` - Online standardization (z-score normalization)
- `MinMaxScaler` - Online min-max scaling
- `MaxAbsScaler` - Online max absolute value scaling
- `OneHotEncoder` - Online one-hot encoding
- `OrdinalEncoder` - Online ordinal encoding
- `TargetEncoder` - Online target encoding with smoothing
- `MeanImputer` - Impute missing with running mean
- `ModeImputer` - Impute missing with running mode
"""
module Transform

using OnlineStatsBase
using OnlineStats
using OnlineStats: Variance, Extrema, mean, var
using ..OnlineML: Transformer
import ..OnlineML: transform, fit_transform!, inverse_transform, reset!

include("scalers.jl")
include("encoders.jl")
include("imputers.jl")
include("max_abs_scaler.jl")

export StandardScaler, MinMaxScaler, MaxAbsScaler
export OneHotEncoder, OrdinalEncoder, TargetEncoder
export MeanImputer, ModeImputer, mode

end # module Transform
