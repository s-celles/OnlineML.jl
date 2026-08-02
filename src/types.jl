# Core abstract types and interfaces for OnlineML.jl
# All types inherit from OnlineStat{T} per OnlineStatsBase interface

using OnlineStatsBase: OnlineStat, fit!, value, nobs, _fit!, _merge!
using Tables

# =============================================================================
# Drift Status Enum (T007)
# =============================================================================

"""
    DriftStatus

Enumeration representing the current drift detection status.

- `NoDrift`: No distribution change detected
- `Warning`: Potential drift detected, monitoring closely
- `DriftDetected`: Significant distribution change detected

Note: The enum value is named `DriftDetected` (not `Drift`) to avoid
name collision with the `Drift` submodule.
"""
@enum DriftStatus NoDrift Warning DriftDetected

# =============================================================================
# Abstract Types (T008-T012)
# =============================================================================

"""
    Learner{X,Y} <: OnlineStat{Tuple{X,Y}}

Abstract type for supervised learning models.

Type parameters:
- `X`: Feature type (e.g., `AbstractVector{Float64}`)
- `Y`: Target type (e.g., `Float64` for regression, `Int` for classification)

Required interface:
- `fit!(l::Learner, x, y)` - Update model with observation
- `predict(l::Learner, x) -> Y` - Generate prediction
- `value(l::Learner)` - Return model state
- `nobs(l::Learner) -> Int` - Observation count
- `reset!(l::Learner)` - Reinitialize state
"""
abstract type Learner{X,Y} <: OnlineStat{Tuple{X,Y}} end

"""
    UnsupervisedLearner{X} <: OnlineStat{X}

Abstract type for unsupervised models (clustering, anomaly detection).

Type parameters:
- `X`: Feature type

Required interface:
- `fit!(u::UnsupervisedLearner, x)` - Update model
- `predict(u::UnsupervisedLearner, x)` - Cluster assignment or anomaly score
- `value(u::UnsupervisedLearner)` - Model state
- `nobs(u::UnsupervisedLearner) -> Int` - Observation count
- `reset!(u::UnsupervisedLearner)` - Reinitialize
"""
abstract type UnsupervisedLearner{X} <: OnlineStat{X} end

"""
    Transformer{X} <: OnlineStat{X}

Abstract type for feature preprocessing transformers.

Type parameters:
- `X`: Input/output feature type

Required interface:
- `fit!(t::Transformer, x)` - Update statistics
- `transform(t::Transformer, x) -> X` - Apply transformation
- `fit_transform!(t::Transformer, x) -> X` - Update and transform
- `value(t::Transformer)` - Learned parameters
- `nobs(t::Transformer) -> Int` - Observation count
- `reset!(t::Transformer)` - Reinitialize
"""
abstract type Transformer{X} <: OnlineStat{X} end

"""
    Detector <: OnlineStat{Float64}

Abstract type for drift detection algorithms.

Required interface:
- `update!(d::Detector, value) -> DriftStatus` - Process value, return status
- `detected_drift(d::Detector) -> Bool` - Check drift flag
- `detected_warning(d::Detector) -> Bool` - Check warning flag
- `reset!(d::Detector)` - Reinitialize
"""
abstract type Detector <: OnlineStat{Float64} end

"""
    Metric{T} <: OnlineStat{Tuple{T,T}}

Abstract type for evaluation metrics.

Type parameters:
- `T`: Type of predictions and ground truth

Required interface:
- `fit!(m::Metric, y_true, y_pred)` - Update metric
- `value(m::Metric) -> Float64` - Current score
- `nobs(m::Metric) -> Int` - Observation count
- `reset!(m::Metric)` - Reinitialize
"""
abstract type Metric{T} <: OnlineStat{Tuple{T,T}} end

# =============================================================================
# Default Interface Methods (T013)
# =============================================================================

"""
    predict(learner, x)

Generate a prediction for input x. Must be implemented by concrete types.
"""
function predict end

"""
    predict_proba(learner, x) -> Dict{Y, Float64}

Return class probability estimates. For classifiers only.
"""
function predict_proba end

"""
    reset!(model)

Reset the model to its initial state, forgetting all learned data.
Must be implemented by concrete types.
"""
function reset! end

"""
    transform(t::Transformer, x)

Apply learned transformation to input without updating state.
"""
function transform end

"""
    fit_transform!(t::Transformer, x)

Update transformer with observation and return transformed result.
"""
function fit_transform!(t::Transformer, x)
    fit!(t, x)
    return transform(t, x)
end

"""
    inverse_transform(t::Transformer, x)

Reverse the transformation. Only available for invertible transformers.
"""
function inverse_transform end

"""
    update!(d::Detector, value) -> DriftStatus

Process a new value and return the current drift status.
"""
function update! end

"""
    detected_drift(d::Detector) -> Bool

Check if drift has been detected.
"""
function detected_drift end

"""
    detected_warning(d::Detector) -> Bool

Check if warning state has been triggered.
"""
function detected_warning end

"""
    score(detector, x) -> Float64

Return anomaly score for an observation.
"""
function score end

"""
    score_one(detector, x) -> Float64

Return anomaly score in [0, 1] range for a single observation.
"""
function score_one end

"""
    status(detector) -> DriftStatus

Return the current drift status of a detector.
"""
function status end

"""
    is_anomaly(detector, x; threshold=0.5) -> Bool

Check if observation is anomalous based on threshold.
"""
function is_anomaly(detector, x; threshold=0.5)
    return score_one(detector, x) > threshold
end

"""
    fit_score!(detector, x) -> Float64

Update detector with observation and return anomaly score.
"""
function fit_score!(detector, x)
    score = score_one(detector, x)
    fit!(detector, x)
    return score
end

# =============================================================================
# Helper Functions (T014)
# =============================================================================

"""
    fit_predict!(learner, x, y)

Test-then-train: predict before learning (prequential evaluation).
Returns the prediction made BEFORE fitting.

# Example
```jldoctest
julia> model = LogisticRegression()
LogisticRegression: n=0 | value=(weights = Float64[], bias = 0.0)

julia> y_pred = fit_predict!(model, [1.0, 2.0], 1)
0

julia> nobs(model)
1
```
"""
function fit_predict!(learner::Learner, x, y)
    y_pred = predict(learner, x)
    fit!(learner, (x, y))  # Pass as tuple for OnlineStatsBase
    return y_pred
end

"""
    fit_predict!(learner, x)

For unsupervised learners: predict cluster/score before updating.
"""
function fit_predict!(learner::UnsupervisedLearner, x)
    pred = predict(learner, x)
    fit!(learner, x)
    return pred
end

"""
    fit_batch!(learner, observations)

Consume `observations` exactly once, in iteration order, and update `learner` for
each observation. For a supervised [`Learner`](@ref), every element must be an
`(x, y)` tuple. For an [`UnsupervisedLearner`](@ref), every element is treated as
one feature observation.

Updates are committed observation by observation. If iteration or fitting fails,
successfully processed observations remain committed; this function does not copy
or roll back learner state. The return value is `learner` itself.

# Example
```jldoctest
julia> model = LogisticRegression();

julia> fit_batch!(model, (([x], x > 0 ? 1 : 0) for x in (-1.0, 1.0)));

julia> nobs(model)
2
```
"""
function fit_batch!(learner::Learner, observations)
    for observation in observations
        if !(observation isa Tuple) || length(observation) != 2
            throw(ArgumentError("a supervised batch observation must be an (x, y) tuple"))
        end
        fit!(learner, observation)
    end
    return learner
end

function fit_batch!(learner::UnsupervisedLearner, observations)
    for observation in observations
        fit!(learner, observation)
    end
    return learner
end

# =============================================================================
# NamedTuple Input Support (Feature 002)
# =============================================================================

"""
    namedtuple_to_features(nt::NamedTuple, target::Symbol) -> (Vector{Float64}, target_value)

Extract feature vector and target value from a NamedTuple.
Internal helper - not exported.

Features are extracted in NamedTuple field order, excluding the target column.
"""
function namedtuple_to_features(nt::NamedTuple, target::Symbol)
    names = propertynames(nt)

    # T005: Validate target column exists
    if target ∉ names
        throw(ArgumentError("target column :$target not found in NamedTuple with fields $names"))
    end

    # Extract feature columns (exclude target)
    feature_names = filter(n -> n != target, names)

    # T006: Validate at least one feature remains
    if isempty(feature_names)
        throw(ArgumentError("no feature columns remain after excluding target :$target"))
    end

    # Build feature vector
    x = Float64[getproperty(nt, n) for n in feature_names]
    y = getproperty(nt, target)

    return (x, y)
end

"""
    namedtuple_to_vector(nt::NamedTuple; exclude::Vector{Symbol}=Symbol[]) -> Vector{Float64}

Extract all numeric fields from a NamedTuple as a feature vector.
Internal helper - not exported.

Features are extracted in NamedTuple field order, excluding specified columns.
"""
function namedtuple_to_vector(nt::NamedTuple; exclude::Vector{Symbol}=Symbol[])
    names = propertynames(nt)

    # Extract feature columns (exclude specified)
    feature_names = filter(n -> n ∉ exclude, names)

    # T006: Validate at least one feature remains
    if isempty(feature_names)
        throw(ArgumentError("no feature columns remain after excluding $exclude"))
    end

    # Build feature vector
    return Float64[getproperty(nt, n) for n in feature_names]
end

"""
    fit!(learner::Learner, nt::NamedTuple, target::Symbol)

Fit a supervised learner with a single observation from a NamedTuple.

Features are extracted in NamedTuple field order, excluding the target column.

# Arguments
- `learner::Learner`: Any supervised learner (LogisticRegression, HoeffdingTree, etc.)
- `nt::NamedTuple`: A row of data with named fields
- `target::Symbol`: The name of the target/label field

# Returns
- The updated learner

# Errors
- `ArgumentError`: If target column not found in NamedTuple
- `ArgumentError`: If no feature columns remain after excluding target

# Example
```jldoctest
julia> model = LogisticRegression()
LogisticRegression: n=0 | value=(weights = Float64[], bias = 0.0)

julia> fit!(model, (; x1=1.0, x2=2.0, y=1), :y);

julia> nobs(model)
1

julia> # Streaming multiple observations (same feature structure)
       for row in [(; x1=1.0, x2=0.5, y=0), (; x1=2.0, x2=1.5, y=1), (; x1=3.0, x2=2.5, y=0)]
           fit!(model, row, :y)
       end

julia> nobs(model)
4
```
"""
function OnlineStatsBase.fit!(learner::Learner, nt::NamedTuple, target::Symbol)
    x, y = namedtuple_to_features(nt, target)
    fit!(learner, (x, y))
    return learner
end

"""
    fit!(learner::UnsupervisedLearner, nt::NamedTuple; exclude::Vector{Symbol}=Symbol[])

Fit an unsupervised learner with a single observation from a NamedTuple.

All numeric fields are used as features, except those in the exclude list.

# Arguments
- `learner::UnsupervisedLearner`: Any unsupervised learner (StreamingKMeans, HalfSpaceTrees, etc.)
- `nt::NamedTuple`: A row of data with named fields
- `exclude::Vector{Symbol}`: Field names to exclude from features (default: empty)

# Returns
- The updated learner

# Errors
- `ArgumentError`: If no feature columns remain after exclusions

# Example
```jldoctest
julia> model = StreamingKMeans(k=3)
StreamingKMeans: n=0 | value=Vector{Float64}[]

julia> fit!(model, (; x1=1.0, x2=2.0));

julia> nobs(model)
1

julia> # With exclusion
       fit!(model, (; x1=1.0, x2=2.0, id=1); exclude=[:id]);

julia> nobs(model)
2
```
"""
function OnlineStatsBase.fit!(learner::UnsupervisedLearner, nt::NamedTuple; exclude::Vector{Symbol}=Symbol[])
    x = namedtuple_to_vector(nt; exclude=exclude)
    fit!(learner, x)
    return learner
end

# =============================================================================
# Tables.jl Integration (T123 - placed here for convenience method)
# =============================================================================

"""
    fit!(learner::Learner, table, target::Symbol)

Convenience method to fit a learner on a Tables.jl source.
Iterates through all rows, extracting features and target.

# Example
```jldoctest
julia> import DataFrames

julia> df = DataFrames.DataFrame(x1=[1.0, 2.0, 3.0], x2=[4.0, 5.0, 6.0], y=[0, 1, 0])
3×3 DataFrame
 Row │ x1       x2       y
     │ Float64  Float64  Int64
─────┼─────────────────────────
   1 │     1.0      4.0      0
   2 │     2.0      5.0      1
   3 │     3.0      6.0      0

julia> model = LogisticRegression()
LogisticRegression: n=0 | value=(weights = Float64[], bias = 0.0)

julia> fit!(model, df, :y);

julia> nobs(model)
3
```
"""
function OnlineStatsBase.fit!(learner::Learner, table, target::Symbol)
    rows = Tables.rows(table)
    cols = Tables.columnnames(table)
    feature_cols = filter(c -> c != target, cols)

    for row in rows
        y = getproperty(row, target)
        x = [Float64(getproperty(row, c)) for c in feature_cols]
        fit!(learner, (x, y))
    end
    return learner
end

# Note: fit! for UnsupervisedLearner with Tables.jl is handled separately
# to avoid ambiguity with the vector-based fit!.
# Use iterate_table! helper instead for table-based fitting.

"""
    iterate_table!(learner::UnsupervisedLearner, table; exclude=Symbol[])

Convenience method to fit an unsupervised learner on a Tables.jl source.

# Example
```jldoctest
julia> model = StreamingKMeans(k=3)
StreamingKMeans: n=0 | value=Vector{Float64}[]

julia> import DataFrames

julia> df = DataFrames.DataFrame(x1=[1.0, 2.0, 3.0], x2=[4.0, 5.0, 6.0])
3×2 DataFrame
 Row │ x1       x2
     │ Float64  Float64
─────┼──────────────────
   1 │     1.0      4.0
   2 │     2.0      5.0
   3 │     3.0      6.0

julia> iterate_table!(model, df);

julia> nobs(model)
3
```
"""
function iterate_table!(learner::UnsupervisedLearner, table; exclude::Vector{Symbol}=Symbol[])
    rows = Tables.rows(table)
    cols = Tables.columnnames(table)
    feature_cols = filter(c -> c ∉ exclude, cols)

    for row in rows
        x = [getproperty(row, c) for c in feature_cols]
        fit!(learner, x)
    end
    return learner
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    dict_argmax(d::AbstractDict) -> key

Return the key with the maximum value in a dictionary.
Returns 0 if the dictionary is empty.
"""
function dict_argmax(d::AbstractDict)
    isempty(d) && return 0
    max_key = first(keys(d))
    max_val = d[max_key]
    for (k, v) in d
        if v > max_val
            max_val = v
            max_key = k
        end
    end
    return max_key
end

"""
    generate(generator) -> (x, y)

Generate a sample from a synthetic data generator.
Returns a tuple of (features, target).

This is a generic interface implemented by generators in the Streams module.
"""
function generate end
