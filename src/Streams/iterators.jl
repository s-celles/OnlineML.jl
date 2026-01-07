# Data stream iterators

using Tables

"""
    DataStream{T}

Iterator wrapper for any data source that provides (x, y) pairs.

# Example
```jldoctest
julia> X = rand(3, 2);

julia> y = [0, 1, 0];

julia> stream = DataStream(X, y);

julia> length(stream)
3
```
"""
struct DataStream{T}
    source::T
    n::Union{Int, Nothing}
    target_col::Union{Symbol, Nothing}
    feature_cols::Union{Vector{Symbol}, Nothing}
end

# Default constructor
DataStream(source, n) = DataStream(source, n, nothing, nothing)

# Constructor from arrays
function DataStream(X::AbstractMatrix, y::AbstractVector)
    DataStream((X, y), size(X, 1), nothing, nothing)
end

# Constructor from generator (keyword argument version)
function DataStream(generator; n::Union{Int, Nothing}=nothing,
                    target::Union{Symbol, Nothing}=nothing,
                    features::Union{Vector{Symbol}, Nothing}=nothing)
    # If target is specified, treat as Tables.jl source
    if target !== nothing
        return _datastream_from_table(generator, target, features)
    end

    # Otherwise, treat as generator
    DataStream(generator, n, nothing, nothing)
end

# Internal function for Tables.jl-compatible source
function _datastream_from_table(table, target::Symbol, features::Union{Vector{Symbol}, Nothing})
    if !Tables.istable(table)
        throw(ArgumentError("Source must be a Tables.jl-compatible table"))
    end

    cols = Tables.columnnames(table)
    if !(target in cols)
        throw(ArgumentError("Target column :$target not found in table"))
    end

    # Determine feature columns
    feature_cols = if features !== nothing
        features
    else
        Symbol[c for c in cols if c != target]
    end

    n_rows = Tables.rowcount(table)
    n = n_rows !== nothing ? n_rows : nothing

    DataStream(table, n, target, feature_cols)
end

# Iterator interface for array data
function Base.iterate(ds::DataStream{Tuple{M,V}}) where {M<:AbstractMatrix, V<:AbstractVector}
    size(ds.source[1], 1) == 0 && return nothing
    X, y = ds.source
    return ((X[1, :], y[1]), 2)
end

function Base.iterate(ds::DataStream{Tuple{M,V}}, state::Int) where {M<:AbstractMatrix, V<:AbstractVector}
    X, y = ds.source
    state > size(X, 1) && return nothing
    return ((X[state, :], y[state]), state + 1)
end

Base.length(ds::DataStream{Tuple{M,V}}) where {M<:AbstractMatrix, V<:AbstractVector} = size(ds.source[1], 1)

# Iterator interface for Tables.jl sources
# Check if DataStream has a target column (Tables.jl source)
_is_table_stream(ds::DataStream) = ds.target_col !== nothing

function Base.iterate(ds::DataStream{T}) where {T}
    # Handle Tables.jl sources
    if _is_table_stream(ds)
        rows = Tables.rows(ds.source)
        row_iter = iterate(rows)
        row_iter === nothing && return nothing
        row, row_state = row_iter
        x, y = _extract_features_target(row, ds.feature_cols, ds.target_col)
        return ((x, y), (rows, row_state))
    end

    # Handle generators (fallback for non-tuple, non-table sources)
    ds.n !== nothing && ds.n <= 0 && return nothing
    x, y = generate(ds.source)
    return ((x, y), 1)
end

function Base.iterate(ds::DataStream{T}, state) where {T}
    # Handle Tables.jl sources
    if _is_table_stream(ds)
        rows, row_state = state
        row_iter = iterate(rows, row_state)
        row_iter === nothing && return nothing
        row, new_row_state = row_iter
        x, y = _extract_features_target(row, ds.feature_cols, ds.target_col)
        return ((x, y), (rows, new_row_state))
    end

    # Handle generators (state is Int for generators)
    if ds.n !== nothing && state >= ds.n
        return nothing
    end
    x, y = generate(ds.source)
    return ((x, y), state + 1)
end

# Helper to extract features and target from a row
function _extract_features_target(row, feature_cols::Vector{Symbol}, target_col::Symbol)
    features = Float64[getproperty(row, col) for col in feature_cols]
    target = getproperty(row, target_col)
    return (features, target)
end

Base.length(ds::DataStream{G}) where {G} = ds.n

# Utility function
"""
    iterate_stream(source; n=nothing)

Create an iterator over a data stream.

# Arguments
- `source` - Data source (matrix+vector, generator, or any iterable)
- `n` - Optional limit on number of iterations
"""
function iterate_stream(source; n::Union{Int, Nothing}=nothing)
    return DataStream(source, n)
end


# =============================================================================
# Stream Operations
# =============================================================================

"""
    batch_stream(stream, batch_size)

Create batched iterator that yields (X_batch, y_batch) tuples.

# Example
```jldoctest
julia> gen = SineGenerator();

julia> stream = DataStream(gen, n=20);

julia> batched = batch_stream(stream, 10);

julia> length(batched)
2
```
"""
struct BatchedStream{S}
    stream::S
    batch_size::Int
end

batch_stream(stream, batch_size::Int) = BatchedStream(stream, batch_size)
batch_stream(stream; batch_size::Int) = BatchedStream(stream, batch_size)

function Base.iterate(bs::BatchedStream)
    X_batch = Vector{Vector{Float64}}()
    y_batch = Vector{Any}()

    stream_state = nothing
    for i in 1:bs.batch_size
        result = stream_state === nothing ? iterate(bs.stream) : iterate(bs.stream, stream_state)
        result === nothing && break
        (x, y), stream_state = result
        push!(X_batch, x)
        push!(y_batch, y)
    end

    isempty(X_batch) && return nothing

    X_matrix = reduce(hcat, X_batch)'
    return ((X_matrix, y_batch), stream_state)
end

function Base.iterate(bs::BatchedStream, state)
    state === nothing && return nothing

    X_batch = Vector{Vector{Float64}}()
    y_batch = Vector{Any}()

    stream_state = state
    for i in 1:bs.batch_size
        result = iterate(bs.stream, stream_state)
        result === nothing && break
        (x, y), stream_state = result
        push!(X_batch, x)
        push!(y_batch, y)
    end

    isempty(X_batch) && return nothing

    X_matrix = reduce(hcat, X_batch)'
    return ((X_matrix, y_batch), stream_state)
end

Base.length(bs::BatchedStream) = bs.stream.n !== nothing ? div(bs.stream.n, bs.batch_size) : nothing


"""
    take_stream(stream, n)

Take first n elements from stream.

# Example
```jldoctest
julia> gen = SineGenerator();

julia> stream = DataStream(gen, n=100);

julia> first_10 = take_stream(stream, 10);

julia> length(first_10)
10
```
"""
struct TakeStream{S}
    stream::S
    n::Int
end

take_stream(stream, n::Int) = TakeStream(stream, n)

function Base.iterate(ts::TakeStream)
    ts.n <= 0 && return nothing
    result = iterate(ts.stream)
    result === nothing && return nothing
    item, stream_state = result
    return (item, (stream_state, 1))
end

function Base.iterate(ts::TakeStream, state)
    stream_state, count = state
    count >= ts.n && return nothing

    result = iterate(ts.stream, stream_state)
    result === nothing && return nothing
    item, new_stream_state = result
    return (item, (new_stream_state, count + 1))
end

Base.length(ts::TakeStream) = ts.n


"""
    skip_stream(stream, n)

Skip first n elements from stream.

# Example
```jldoctest
julia> gen = SineGenerator();

julia> stream = DataStream(gen, n=100);

julia> after_10 = skip_stream(stream, 10);

julia> length(after_10)
90
```
"""
struct SkipStream{S}
    stream::S
    n::Int
end

skip_stream(stream, n::Int) = SkipStream(stream, n)

function Base.iterate(ss::SkipStream)
    # Skip n elements
    state = nothing
    for i in 1:ss.n
        result = state === nothing ? iterate(ss.stream) : iterate(ss.stream, state)
        result === nothing && return nothing
        _, state = result
    end

    # Return next element after skipping
    result = state === nothing ? iterate(ss.stream) : iterate(ss.stream, state)
    result === nothing && return nothing
    return result
end

function Base.iterate(ss::SkipStream, state)
    return iterate(ss.stream, state)
end

Base.length(ss::SkipStream) = ss.stream.n !== nothing ? max(0, ss.stream.n - ss.n) : nothing
