# Bernoulli Naive Bayes

"""
    BernoulliNB{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}

Online Bernoulli Naive Bayes classifier for binary features.

# Parameters
- `alpha::Float64 = 1.0` - Laplace smoothing parameter

# Example
```jldoctest
julia> model = BernoulliNB(alpha=1.0)
BernoulliNB: n=0 | value=Dict{Int64, Vector{Float64}}()

julia> fit!(model, ([1.0, 0.0], 1));

julia> predict(model, [1.0, 0.0])
1

julia> nobs(model)
1
```
"""
mutable struct BernoulliNB{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}
    feature_counts::Dict{Int, Vector{T}}  # class -> binary feature counts
    class_counts::CountMap{Int}
    alpha::T
    n::Int

    function BernoulliNB{T}(; alpha::Real=1.0) where {T<:AbstractFloat}
        new{T}(Dict{Int, Vector{T}}(), CountMap{Int}(), T(alpha), 0)
    end
end

BernoulliNB(; kwargs...) = BernoulliNB{Float64}(; kwargs...)

function OnlineStatsBase._fit!(nb::BernoulliNB{T}, xy::Tuple) where {T}
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
        if xi > 0
            counts[i] += one(T)
        end
    end

    nb.n += 1
    return nb
end

OnlineStatsBase.value(nb::BernoulliNB) = nb.feature_counts
OnlineStatsBase.nobs(nb::BernoulliNB) = nb.n

function reset!(nb::BernoulliNB{T}) where {T}
    empty!(nb.feature_counts)
    nb.class_counts = CountMap{Int}()
    nb.n = 0
    return nb
end

function predict(nb::BernoulliNB, x::AbstractVector)
    probs = predict_proba(nb, x)
    isempty(probs) && return 0
    return dict_argmax(probs)
end

function predict_proba(nb::BernoulliNB{T}, x::AbstractVector) where {T}
    if nb.n == 0 || isempty(nb.feature_counts)
        return Dict{Int, T}()
    end

    x = collect(T, x)
    log_probs = Dict{Int, T}()
    total_count = T(nb.n)

    for (class, counts) in nb.feature_counts
        class_count_val = value(nb.class_counts)[class]
        class_count = T(class_count_val)
        log_prior = log(class_count / total_count)

        log_likelihood = zero(T)
        for (i, xi) in enumerate(x)
            if i <= length(counts)
                # P(x_i=1 | class)
                p1 = (counts[i] + nb.alpha) / (class_count + 2 * nb.alpha)
                if xi > 0
                    log_likelihood += log(p1)
                else
                    log_likelihood += log(one(T) - p1)
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

classes(nb::BernoulliNB) = collect(keys(nb.feature_counts))

function class_prior(nb::BernoulliNB{T}) where {T}
    nb.n == 0 && return Dict{Int, T}()
    counts = value(nb.class_counts)
    total = T(nb.n)
    return Dict(k => T(v) / total for (k, v) in counts)
end
