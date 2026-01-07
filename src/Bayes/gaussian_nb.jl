# Gaussian Naive Bayes

"""
    GaussianNB{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}

Online Gaussian Naive Bayes classifier.

Maintains per-class, per-feature Gaussian distributions using online
mean and variance estimation.

# Example
```jldoctest
julia> model = GaussianNB()
GaussianNB: n=0 | value=Dict{Int64, Vector{Variance}}()

julia> fit!(model, ([1.0, 2.0], 1));

julia> predict(model, [1.0, 2.0])
1

julia> nobs(model)
1
```
"""
mutable struct GaussianNB{T<:AbstractFloat} <: Learner{AbstractVector{T}, Int}
    class_stats::Dict{Int, Vector{Variance}}  # class -> per-feature stats
    class_counts::CountMap{Int}
    n::Int

    function GaussianNB{T}() where {T<:AbstractFloat}
        new{T}(Dict{Int, Vector{Variance}}(), CountMap{Int}(), 0)
    end
end

GaussianNB() = GaussianNB{Float64}()

# OnlineStatsBase interface
function OnlineStatsBase._fit!(nb::GaussianNB{T}, xy::Tuple) where {T}
    x, y = xy
    x = collect(T, x)
    y = Int(y)

    # Update class counts
    fit!(nb.class_counts, y)

    # Update per-class, per-feature statistics
    if !haskey(nb.class_stats, y)
        nb.class_stats[y] = [Variance(T) for _ in 1:length(x)]
    end

    stats = nb.class_stats[y]

    # Expand if needed (dynamic feature discovery)
    while length(stats) < length(x)
        push!(stats, Variance(T))
    end

    for (i, xi) in enumerate(x)
        fit!(stats[i], xi)
    end

    nb.n += 1
    return nb
end

OnlineStatsBase.value(nb::GaussianNB) = nb.class_stats
OnlineStatsBase.nobs(nb::GaussianNB) = nb.n

function reset!(nb::GaussianNB{T}) where {T}
    empty!(nb.class_stats)
    nb.class_counts = CountMap{Int}()
    nb.n = 0
    return nb
end

# Learner interface
function predict(nb::GaussianNB, x::AbstractVector)
    probs = predict_proba(nb, x)
    if isempty(probs)
        return 0
    end
    return dict_argmax(probs)
end

function predict_proba(nb::GaussianNB{T}, x::AbstractVector) where {T}
    if nb.n == 0 || isempty(nb.class_stats)
        return Dict{Int, T}()
    end

    x = collect(T, x)
    log_probs = Dict{Int, T}()
    total_count = T(nb.n)

    for (class, stats) in nb.class_stats
        # Log prior
        class_count = T(value(nb.class_counts)[class])
        log_prior = log(class_count / total_count)

        # Log likelihood
        log_likelihood = zero(T)
        for (i, xi) in enumerate(x)
            if i <= length(stats) && nobs(stats[i]) > 0
                μ = mean(stats[i])
                σ² = var(stats[i])
                if σ² > 0
                    # Gaussian log-likelihood
                    log_likelihood += -0.5 * (log(2π * σ²) + (xi - μ)^2 / σ²)
                end
            end
        end

        log_probs[class] = log_prior + log_likelihood
    end

    # Convert to probabilities (softmax)
    if isempty(log_probs)
        return Dict{Int, T}()
    end

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

# Inspection methods
"""
    classes(nb::GaussianNB) -> Vector{Int}

Return the classes seen during training.
"""
classes(nb::GaussianNB) = collect(keys(nb.class_stats))

"""
    class_prior(nb::GaussianNB) -> Dict{Int, Float64}

Return the prior probability of each class.
"""
function class_prior(nb::GaussianNB{T}) where {T}
    if nb.n == 0
        return Dict{Int, T}()
    end
    counts = value(nb.class_counts)
    total = T(nb.n)
    return Dict(k => T(v) / total for (k, v) in counts)
end

