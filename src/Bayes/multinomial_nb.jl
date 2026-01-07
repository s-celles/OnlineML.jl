# Multinomial Naive Bayes

"""
    MultinomialNB{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}

Online Multinomial Naive Bayes classifier for count data.

# Parameters
- `alpha::Float64 = 1.0` - Laplace smoothing parameter

# Example
```jldoctest
julia> model = MultinomialNB(alpha=1.0)
MultinomialNB: n=0 | value=Dict{Int64, Vector{Float64}}()

julia> fit!(model, ([1.0, 2.0], 1));

julia> predict(model, [1.0, 2.0])
1

julia> nobs(model)
1
```
"""
mutable struct MultinomialNB{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}
    feature_counts::Dict{Int, Vector{T}}  # class -> feature counts
    class_counts::CountMap{Int}
    alpha::T
    n::Int

    function MultinomialNB{T}(; alpha::Real=1.0) where {T<:AbstractFloat}
        new{T}(Dict{Int, Vector{T}}(), CountMap{Int}(), T(alpha), 0)
    end
end

MultinomialNB(; kwargs...) = MultinomialNB{Float64}(; kwargs...)

function OnlineStatsBase._fit!(nb::MultinomialNB{T}, xy::Tuple) where {T}
    x, y = xy
    x = collect(T, x)
    y = Int(y)

    fit!(nb.class_counts, y)

    if !haskey(nb.feature_counts, y)
        nb.feature_counts[y] = zeros(T, length(x))
    end

    counts = nb.feature_counts[y]
    while length(counts) < length(x)
        push!(counts, zero(T))
    end

    for (i, xi) in enumerate(x)
        counts[i] += xi
    end

    nb.n += 1
    return nb
end

OnlineStatsBase.value(nb::MultinomialNB) = nb.feature_counts
OnlineStatsBase.nobs(nb::MultinomialNB) = nb.n

function reset!(nb::MultinomialNB{T}) where {T}
    empty!(nb.feature_counts)
    nb.class_counts = CountMap{Int}()
    nb.n = 0
    return nb
end

function predict(nb::MultinomialNB, x::AbstractVector)
    probs = predict_proba(nb, x)
    isempty(probs) && return 0
    return dict_argmax(probs)
end

function predict_proba(nb::MultinomialNB{T}, x::AbstractVector) where {T}
    if nb.n == 0 || isempty(nb.feature_counts)
        return Dict{Int, T}()
    end

    x = collect(T, x)
    log_probs = Dict{Int, T}()
    total_count = T(nb.n)

    for (class, counts) in nb.feature_counts
        class_count = T(value(nb.class_counts)[class])
        log_prior = log(class_count / total_count)

        # Feature count total for this class
        total_features = sum(counts) + nb.alpha * length(counts)

        log_likelihood = zero(T)
        for (i, xi) in enumerate(x)
            if i <= length(counts)
                feature_prob = (counts[i] + nb.alpha) / total_features
                if feature_prob > 0 && xi > 0
                    log_likelihood += xi * log(feature_prob)
                end
            end
        end

        log_probs[class] = log_prior + log_likelihood
    end

    # Softmax
    max_log = maximum(values(log_probs))
    probs = Dict{Int, T}()
    total = zero(T)

    for (class, lp) in log_probs
        p = exp(lp - max_log)
        probs[class] = p
        total += p
    end

    if total > 0
        for class in keys(probs)
            probs[class] /= total
        end
    end

    return probs
end

classes(nb::MultinomialNB) = collect(keys(nb.feature_counts))

function class_prior(nb::MultinomialNB{T}) where {T}
    nb.n == 0 && return Dict{Int, T}()
    counts = value(nb.class_counts)
    total = T(nb.n)
    return Dict(k => T(v) / total for (k, v) in counts)
end

