# EFDT Node types for ExtremelyFastTree
# These extend the base node infrastructure to support split re-evaluation

"""
    EFDTNode

Abstract type for EFDT tree nodes that maintain statistics at split nodes.
"""
abstract type EFDTNode <: TreeNode end

"""
    EFDTLeaf{T<:AbstractFloat} <: EFDTNode

Leaf node for EFDT, same functionality as LeafNode but typed for EFDT.
"""
mutable struct EFDTLeaf{T<:AbstractFloat} <: EFDTNode
    stats::Dict{Int, SufficientStats{T}}  # class -> statistics
    class_counts::CountMap{Int}
    n::Int
    depth::Int

    function EFDTLeaf{T}(depth::Int=0) where {T}
        new{T}(Dict{Int, SufficientStats{T}}(), CountMap{Int}(), 0, depth)
    end
end

EFDTLeaf(depth::Int=0) = EFDTLeaf{Float64}(depth)

function update!(leaf::EFDTLeaf{T}, x::AbstractVector, y::Int) where {T}
    fit!(leaf.class_counts, y)
    if !haskey(leaf.stats, y)
        leaf.stats[y] = SufficientStats{T}(length(x))
    end
    update!(leaf.stats[y], collect(T, x))
    leaf.n += 1
    return leaf
end

function majority_class(leaf::EFDTLeaf)
    if leaf.n == 0
        return 0
    end
    counts = value(leaf.class_counts)
    if isempty(counts)
        return 0
    end
    return dict_argmax(counts)
end

function class_probabilities(leaf::EFDTLeaf{T}) where {T}
    if leaf.n == 0
        return Dict{Int, T}()
    end
    counts = value(leaf.class_counts)
    total = sum(values(counts))
    return Dict(k => T(v) / total for (k, v) in counts)
end

"""
    EFDTSplit{T<:AbstractFloat} <: EFDTNode

Split node for EFDT that also maintains sufficient statistics for re-evaluation.
Unlike standard SplitNode, this tracks statistics to allow reconsidering the split.
"""
mutable struct EFDTSplit{T<:AbstractFloat} <: EFDTNode
    attribute::Int
    threshold::T
    left::EFDTNode
    right::EFDTNode
    # Statistics for re-evaluation
    stats::Dict{Int, SufficientStats{T}}
    class_counts::CountMap{Int}
    n::Int
    depth::Int
    last_eval::Int  # Last observation count when re-evaluated

    function EFDTSplit{T}(
        attribute::Int,
        threshold::T,
        left::EFDTNode,
        right::EFDTNode,
        depth::Int=0
    ) where {T}
        new{T}(
            attribute, threshold, left, right,
            Dict{Int, SufficientStats{T}}(),
            CountMap{Int}(),
            0, depth, 0
        )
    end
end

function EFDTSplit(attribute::Int, threshold::Float64, left::EFDTNode, right::EFDTNode, depth::Int=0)
    EFDTSplit{Float64}(attribute, threshold, left, right, depth)
end

"""
Update statistics at a split node (for re-evaluation) and route to child.
"""
function update!(node::EFDTSplit{T}, x::AbstractVector, y::Int) where {T}
    # Update statistics at this node for potential re-evaluation
    fit!(node.class_counts, y)
    if !haskey(node.stats, y)
        node.stats[y] = SufficientStats{T}(length(x))
    end
    update!(node.stats[y], collect(T, x))
    node.n += 1
    return node
end

"""
Route an observation to the appropriate child.
"""
function route(node::EFDTSplit{T}, x::AbstractVector) where {T}
    if x[node.attribute] <= node.threshold
        return node.left
    else
        return node.right
    end
end

"""
Find the leaf node for an observation, passing through EFDT nodes.
"""
function find_leaf(node::EFDTLeaf, x)
    return node
end

function find_leaf(node::EFDTSplit, x)
    child = route(node, x)
    return find_leaf(child, x)
end

"""
Compute entropy for an EFDT leaf.
"""
function compute_entropy(leaf::EFDTLeaf{T}) where {T}
    probs = class_probabilities(leaf)
    if isempty(probs)
        return zero(T)
    end
    return -sum(p > 0 ? p * log2(p) : zero(T) for p in values(probs))
end

"""
Compute Gini impurity for an EFDT leaf.
"""
function compute_gini(leaf::EFDTLeaf{T}) where {T}
    if leaf.n == 0
        return zero(T)
    end
    probs = class_probabilities(leaf)
    if isempty(probs)
        return zero(T)
    end
    return one(T) - sum(p * p for p in values(probs))
end

export EFDTNode, EFDTLeaf, EFDTSplit
