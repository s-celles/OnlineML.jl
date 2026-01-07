# MaxAbsScaler - Scale features by maximum absolute value

"""
    MaxAbsScaler{T<:AbstractFloat} <: Transformer{AbstractVector{T}}

Online feature scaler that divides by maximum absolute value.

Scales features to the range [-1, 1] by dividing by the maximum
absolute value seen for each feature. Useful for data that is
already centered at zero.

# Example
```jldoctest
julia> scaler = MaxAbsScaler()
MaxAbsScaler: n=0 | features=0

julia> fit!(scaler, [2.0, -4.0, 1.0]);

julia> transform(scaler, [2.0, -4.0, 1.0])
3-element Vector{Float64}:
  1.0
 -1.0
  1.0

julia> nobs(scaler)
1
```

# Reference
Scales each feature by its maximum absolute value, similar to
sklearn.preprocessing.MaxAbsScaler but for streaming data.
"""
mutable struct MaxAbsScaler{T<:AbstractFloat} <: Transformer{AbstractVector{T}}
    max_abs::Vector{T}
    n::Int

    function MaxAbsScaler{T}() where {T<:AbstractFloat}
        new{T}(T[], 0)
    end
end

MaxAbsScaler() = MaxAbsScaler{Float64}()

# OnlineStatsBase interface
function OnlineStatsBase._fit!(scaler::MaxAbsScaler{T}, x::AbstractVector) where {T}
    # Ensure max_abs has correct size
    if length(scaler.max_abs) < length(x)
        append!(scaler.max_abs, zeros(T, length(x) - length(scaler.max_abs)))
    end

    # Update max absolute values
    for (i, xi) in enumerate(x)
        if i <= length(scaler.max_abs)
            scaler.max_abs[i] = max(scaler.max_abs[i], abs(T(xi)))
        end
    end

    scaler.n += 1
    return scaler
end

OnlineStatsBase.value(scaler::MaxAbsScaler) = scaler.max_abs
OnlineStatsBase.nobs(scaler::MaxAbsScaler) = scaler.n

function reset!(scaler::MaxAbsScaler{T}) where {T}
    empty!(scaler.max_abs)
    scaler.n = 0
    return scaler
end

"""
    transform(scaler::MaxAbsScaler, x::AbstractVector) -> Vector

Transform a feature vector by dividing by maximum absolute values.
Features with max_abs of zero are left unchanged.
"""
function transform(scaler::MaxAbsScaler{T}, x::AbstractVector) where {T}
    if isempty(scaler.max_abs)
        return collect(T, x)
    end

    result = similar(x, T)
    for i in eachindex(x)
        if i <= length(scaler.max_abs) && scaler.max_abs[i] > 0
            result[i] = T(x[i]) / scaler.max_abs[i]
        else
            result[i] = T(x[i])
        end
    end
    return result
end

"""
    fit_transform!(scaler::MaxAbsScaler, x::AbstractVector) -> Vector

Update the scaler with x and return the transformed vector.
"""
function fit_transform!(scaler::MaxAbsScaler{T}, x::AbstractVector) where {T}
    fit!(scaler, x)
    return transform(scaler, x)
end

"""
    inverse_transform(scaler::MaxAbsScaler, x::AbstractVector) -> Vector

Inverse transform by multiplying by maximum absolute values.
"""
function inverse_transform(scaler::MaxAbsScaler{T}, x::AbstractVector) where {T}
    if isempty(scaler.max_abs)
        return collect(T, x)
    end

    result = similar(x, T)
    for i in eachindex(x)
        if i <= length(scaler.max_abs)
            result[i] = T(x[i]) * scaler.max_abs[i]
        else
            result[i] = T(x[i])
        end
    end
    return result
end

# Inspection methods
n_features(scaler::MaxAbsScaler) = length(scaler.max_abs)
max_abs_values(scaler::MaxAbsScaler) = copy(scaler.max_abs)

function Base.show(io::IO, scaler::MaxAbsScaler)
    print(io, "MaxAbsScaler: n=$(nobs(scaler)) | features=$(n_features(scaler))")
end

export MaxAbsScaler
