# ColumnTransformer - apply different transformers to different column subsets

"""
    ColumnTransformer <: OnlineStat{Any}

Apply different transformers to different columns of the input.

# Parameters
- `transformers` - Vector of (name, transformer, columns) tuples
  - `name` - String identifier for this transformer
  - `transformer` - The transformer to apply
  - `columns` - Either:
    - Integer indices: `[1, 2]` for columns 1 and 2
    - Range: `1:3` for columns 1-3
    - Symbol: `:all` for all columns

# Example
```julia
ct = ColumnTransformer([
    ("scaler", StandardScaler(), [1, 2]),
    ("encoder", OneHotEncoder(), [3])
])
fit!(ct, [1.0, 2.0, :a])
transform(ct, [3.0, 4.0, :b])
```
"""
mutable struct ColumnTransformer <: OnlineStat{Any}
    transformers::Vector{Tuple{String, Any, Any}}  # (name, transformer, columns)
    n::Int

    function ColumnTransformer(transformers::Vector)
        typed_transformers = [(String(t[1]), t[2], t[3]) for t in transformers]
        new(typed_transformers, 0)
    end
end

function OnlineStatsBase._fit!(ct::ColumnTransformer, x)
    for (name, transformer, cols) in ct.transformers
        subset = select_columns(x, cols)
        fit!(transformer, subset)
    end
    ct.n += 1
    return ct
end

OnlineStatsBase.value(ct::ColumnTransformer) = [(name, OnlineStatsBase.value(t)) for (name, t, _) in ct.transformers]
OnlineStatsBase.nobs(ct::ColumnTransformer) = ct.n

function reset!(ct::ColumnTransformer)
    for (_, transformer, _) in ct.transformers
        if hasmethod(reset!, Tuple{typeof(transformer)})
            reset!(transformer)
        end
    end
    ct.n = 0
    return ct
end

function transform(ct::ColumnTransformer, x)
    outputs = []
    for (name, transformer, cols) in ct.transformers
        subset = select_columns(x, cols)
        result = transform(transformer, subset)
        if result isa AbstractVector
            append!(outputs, result)
        else
            push!(outputs, result)
        end
    end
    return collect(Float64, outputs)
end

# Helper to select columns from input
function select_columns(x::AbstractVector, cols::AbstractVector{<:Integer})
    return [x[i] for i in cols]
end

function select_columns(x::AbstractVector, cols::UnitRange)
    return x[cols]
end

function select_columns(x::AbstractVector, cols::Symbol)
    if cols == :all
        return x
    else
        error("Unknown column selector: $cols")
    end
end

function select_columns(x::AbstractVector, col::Integer)
    return x[col]
end

export ColumnTransformer
