# FeatureUnion - parallel feature extraction and concatenation

"""
    FeatureUnion <: OnlineStat{Any}

Concatenates outputs from multiple transformers horizontally.
Useful for applying different transformations to the same input.

# Example
```julia
union = FeatureUnion(StandardScaler(), MinMaxScaler())
fit!(union, [1.0, 2.0])
transform(union, [3.0, 4.0])  # Concatenated output from both scalers
```

# Operator
```julia
union = StandardScaler() + MinMaxScaler()  # Equivalent to FeatureUnion
```
"""
mutable struct FeatureUnion <: OnlineStat{Any}
    transformers::Vector{Any}
    n::Int

    function FeatureUnion(transformers...)
        new(collect(Any, transformers), 0)
    end
end

# + operator for building FeatureUnion (constrained to OnlineStat to avoid ambiguities)
Base.:+(t1::OnlineStat, t2::OnlineStat) = FeatureUnion(t1, t2)
Base.:+(fu::FeatureUnion, t::OnlineStat) = FeatureUnion(fu.transformers..., t)

function OnlineStatsBase._fit!(fu::FeatureUnion, x)
    for t in fu.transformers
        fit!(t, x)
    end
    fu.n += 1
    return fu
end

OnlineStatsBase.value(fu::FeatureUnion) = [OnlineStatsBase.value(t) for t in fu.transformers]
OnlineStatsBase.nobs(fu::FeatureUnion) = fu.n

function reset!(fu::FeatureUnion)
    for t in fu.transformers
        if hasmethod(reset!, Tuple{typeof(t)})
            reset!(t)
        end
    end
    fu.n = 0
    return fu
end

function transform(fu::FeatureUnion, x)
    outputs = [transform(t, x) for t in fu.transformers]
    return vcat(outputs...)
end

export FeatureUnion
