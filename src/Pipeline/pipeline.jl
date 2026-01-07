# Composable pipelines

"""
    OnlinePipeline{T<:Tuple} <: Learner{AbstractVector, Any}

Sequential composition of transformers and a final learner.

Transformers are applied in order, with each transforming the input
before passing to the next component.

# Example
```jldoctest
julia> pipe = OnlinePipeline(StandardScaler(), LogisticRegression())
OnlinePipeline: n=0 | value=(StandardScaler: n=0 | value=Variance[], LogisticRegression: n=0 | value=(weights = Float64[], bias = 0.0))

julia> fit!(pipe, ([1.0, 2.0], 1));

julia> predict(pipe, [1.0, 2.0])
1

julia> nobs(pipe)
1
```
"""
struct OnlinePipeline{T<:Tuple} <: Learner{AbstractVector, Any}
    components::T
    n::Base.RefValue{Int}

    function OnlinePipeline(components...)
        # Validate: all but last must be transformers, last must be learner
        if length(components) < 1
            error("Pipeline requires at least one component")
        end

        new{typeof(components)}(components, Ref(0))
    end
end

# OnlineStatsBase interface
function OnlineStatsBase._fit!(pipe::OnlinePipeline, xy::Tuple)
    x, y = xy

    # Transform through all but last component
    for i in 1:(length(pipe.components) - 1)
        component = pipe.components[i]
        fit!(component, x)
        x = transform(component, x)
    end

    # Fit final component (learner) - pass as tuple
    fit!(pipe.components[end], (x, y))

    pipe.n[] += 1
    return pipe
end

OnlineStatsBase.value(pipe::OnlinePipeline) = pipe.components
OnlineStatsBase.nobs(pipe::OnlinePipeline) = pipe.n[]

function reset!(pipe::OnlinePipeline)
    for component in pipe.components
        reset!(component)
    end
    pipe.n[] = 0
    return pipe
end

# Learner interface
function predict(pipe::OnlinePipeline, x)
    # Transform through all but last component
    for i in 1:(length(pipe.components) - 1)
        x = transform(pipe.components[i], x)
    end

    # Predict with final component
    return predict(pipe.components[end], x)
end

function predict_proba(pipe::OnlinePipeline, x)
    # Transform through all but last component
    for i in 1:(length(pipe.components) - 1)
        x = transform(pipe.components[i], x)
    end

    # Predict with final component
    return predict_proba(pipe.components[end], x)
end

# Pipe operator for building pipelines
function Base.:(|>)(a::Union{Transformer, Learner}, b::Union{Transformer, Learner})
    OnlinePipeline(a, b)
end

function Base.:(|>)(pipe::OnlinePipeline, component::Union{Transformer, Learner})
    OnlinePipeline(pipe.components..., component)
end

# Utility functions
components(pipe::OnlinePipeline) = pipe.components
n_components(pipe::OnlinePipeline) = length(pipe.components)
