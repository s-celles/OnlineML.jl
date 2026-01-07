# Tree node types for Hoeffding Trees

using OnlineStats: Variance, CountMap, Mean

"""
    TreeNode

Abstract type for tree nodes.
"""
abstract type TreeNode end

"""
    SufficientStats{T<:AbstractFloat}

Sufficient statistics for evaluating splits at a leaf node.
Tracks per-class, per-attribute Gaussian statistics.
"""
mutable struct SufficientStats{T<:AbstractFloat}
    count::Int
    gaussians::Vector{Variance}  # Per-attribute Gaussian estimator

    function SufficientStats{T}(n_features::Int) where {T}
        new{T}(0, [Variance(T) for _ in 1:n_features])
    end
end

SufficientStats(n_features::Int) = SufficientStats{Float64}(n_features)

function update!(ss::SufficientStats{T}, x::AbstractVector{T}) where {T}
    ss.count += 1
    for (i, xi) in enumerate(x)
        if i <= length(ss.gaussians)
            fit!(ss.gaussians[i], xi)
        else
            push!(ss.gaussians, Variance(T))
            fit!(ss.gaussians[end], xi)
        end
    end
    return ss
end

"""
    LeafNode{T} <: TreeNode

Leaf node containing sufficient statistics per class.
"""
mutable struct LeafNode{T<:AbstractFloat} <: TreeNode
    stats::Dict{Int, SufficientStats{T}}  # class -> statistics
    class_counts::CountMap{Int}
    n::Int
    depth::Int

    function LeafNode{T}(depth::Int=0) where {T}
        new{T}(Dict{Int, SufficientStats{T}}(), CountMap{Int}(), 0, depth)
    end
end

LeafNode(depth::Int=0) = LeafNode{Float64}(depth)

function update!(leaf::LeafNode{T}, x::AbstractVector, y::Int) where {T}
    # Update class counts
    fit!(leaf.class_counts, y)

    # Update sufficient statistics for this class
    if !haskey(leaf.stats, y)
        leaf.stats[y] = SufficientStats{T}(length(x))
    end
    update!(leaf.stats[y], collect(T, x))

    leaf.n += 1
    return leaf
end

"""
Get the majority class at this leaf.
"""
function majority_class(leaf::LeafNode)
    if leaf.n == 0
        return 0
    end
    counts = value(leaf.class_counts)
    if isempty(counts)
        return 0
    end
    return dict_argmax(counts)
end

"""
Get class probabilities at this leaf.
"""
function class_probabilities(leaf::LeafNode{T}) where {T}
    if leaf.n == 0
        return Dict{Int, T}()
    end
    counts = value(leaf.class_counts)
    total = sum(values(counts))
    return Dict(k => T(v) / total for (k, v) in counts)
end

"""
    SplitNode{T} <: TreeNode

Internal split node.
"""
mutable struct SplitNode{T<:AbstractFloat} <: TreeNode
    attribute::Int
    threshold::T
    left::TreeNode
    right::TreeNode

    function SplitNode{T}(attribute::Int, threshold::T, left::TreeNode, right::TreeNode) where {T}
        new{T}(attribute, threshold, left, right)
    end
end

SplitNode(attribute::Int, threshold::Float64, left::TreeNode, right::TreeNode) =
    SplitNode{Float64}(attribute, threshold, left, right)

"""
Route an observation to the appropriate child.
"""
function route(node::SplitNode{T}, x::AbstractVector) where {T}
    if x[node.attribute] <= node.threshold
        return node.left
    else
        return node.right
    end
end

export TreeNode, SufficientStats, LeafNode, SplitNode
export update!, majority_class, class_probabilities, route
